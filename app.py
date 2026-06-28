import os
import pandas as pd
import numpy as np
import re
import hashlib
from flask import Flask, send_file, request, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PRECOMPUTED = False


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)

PROBABILITY_CAP = 0.99

# Explicit university-group calibration (rough mainstream perception tiers).
# These are used as soft baselines and blended with hensachi-based baseline.
UNIVERSITY_GROUP_BASELINES = {
    'top_elite': {'bunka': 650, 'rika': 640},
    'upper_private': {'bunka': 620, 'rika': 610},
    'upper_public': {'bunka': 610, 'rika': 600},
    'mid_private': {'bunka': 570, 'rika': 560},
    'mid_public': {'bunka': 580, 'rika': 570},
    'standard_private': {'bunka': 535, 'rika': 525},
    'standard_public': {'bunka': 545, 'rika': 535},
}

UNIVERSITY_GROUP_MAP = {
    '東京大学': 'top_elite', '京都大学': 'top_elite', '大阪大学': 'top_elite', '東北大学': 'top_elite',
    '名古屋大学': 'top_elite', '九州大学': 'top_elite', '北海道大学': 'top_elite', '一橋大学': 'top_elite',
    '東京工業大学': 'top_elite', '神戸大学': 'top_elite', '早稲田大学': 'top_elite', '慶應義塾大学': 'top_elite',

    '上智大学': 'upper_private', '国際基督教大学': 'upper_private', '東京理科大学': 'upper_private',
    '明治大学': 'upper_private', '青山学院大学': 'upper_private', '立教大学': 'upper_private',
    '中央大学': 'upper_private', '法政大学': 'upper_private', '同志社大学': 'upper_private',
    '立命館大学': 'upper_private', '関西大学': 'upper_private', '関西学院大学': 'upper_private',

    '筑波大学': 'upper_public', '横浜国立大学': 'upper_public', 'お茶の水女子大学': 'upper_public',
    '東京外国語大学': 'upper_public', '千葉大学': 'upper_public', '広島大学': 'upper_public',
    '金沢大学': 'upper_public', '岡山大学': 'upper_public', '東京都立大学': 'upper_public',
}

GROUP_HENSACHI_DEFAULT = {
    'top_elite': 67,
    'upper_private': 61,
    'upper_public': 61,
    'mid_private': 56,
    'mid_public': 57,
    'standard_private': 52,
    'standard_public': 53,
}


def normalize_text(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    text = re.sub(r'[\s\u3000\(\)\（\）\-\‐\–\—]+', '', text)
    return text.lower()


def is_school_name(value):
    text = str(value or '').strip()
    if not text:
        return False
    return bool(re.search(r'(大学院大学|短期大学|大学校|大学|学園大学|学院大学)$', text))


def normalize_official_bool(value):
    text = str(value or '').strip().lower()
    if not text:
        return None
    yes_set = {'是', 'yes', 'y', 'true', '1', '可', '有'}
    no_set = {'否', 'no', 'n', 'false', '0', '无', 'なし', '留学生なし', '一般選抜のみ'}
    if text in yes_set:
        return True
    if text in no_set:
        return False
    return None


def clean_official_faculty_text(raw, school=''):
    text = str(raw or '').strip()
    if not text:
        return ''
    if school and text.startswith(school):
        text = text[len(school):].strip()
    text = re.sub(r'^[\-_ー／/]+', '', text).strip()
    # Drop bracketed exam mode notes while keeping core faculty/department name.
    text = re.sub(r'（[^）]*）', '', text).strip()
    text = re.sub(r'\([^\)]*\)', '', text).strip()
    text = re.sub(r'(A方式|B方式|前期|後期)$', '', text).strip()
    text = re.sub(r'(?<=[一-龥ぁ-ゖァ-ヺー])\s+(?=[一-龥ぁ-ゖァ-ヺー])', '', text)
    return text


def clean_official_school_text(raw):
    text = str(raw or '').strip()
    if not text:
        return ''
    # Remove annotation notes while keeping the canonical school name.
    text = re.sub(r'（[^）]*）', '', text).strip()
    text = re.sub(r'\([^\)]*\)', '', text).strip()
    text = re.sub(r'\[[^\]]*\]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def looks_like_official_faculty(value):
    text = clean_official_faculty_text(value)
    if not text:
        return False

    deny_patterns = [
        r'https?://',
        r'募集要項',
        r'公開',
        r'準備中',
        r'文系のみ',
        r'理系のみ',
        r'一般選抜',
        r'留学生なし',
        r'外国人入試なし',
        r'のみ$',
        r'事前',
        r'URL',
    ]
    if any(re.search(pat, text) for pat in deny_patterns):
        return False

    return bool(re.search(r'(学部|学科|学域|学群|学院|課程|専攻|コース|類|系)$', text))


def split_school_and_faculty(raw):
    text = clean_official_school_text(raw)
    if not text:
        return '', ''

    m = re.match(r'^(.*?(?:大学院大学|短期大学|大学校|大学))(?:[\-_ー／/]?)(.+)$', text)
    if not m:
        return '', ''

    school = m.group(1).strip()
    faculty = m.group(2).strip()
    return school, faculty


def load_official_name_maps():
    school_map = {}
    faculty_map = {}
    department_map = {}
    school_faculty_index = {}

    official_files = [
        '2025年度 理科学部募集要項_国公立大学募集要項.csv',
        '2025年度 理科学部募集要項_私立大学募集要項.csv'
    ]

    for filename in official_files:
        file_path = project_path(filename)
        if not os.path.exists(file_path):
            continue

        try:
            official = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        except Exception:
            continue

        if official.empty:
            continue

        name_col = official.columns[0]
        owner_col = next((c for c in official.columns if '归属大学' in str(c)), None)
        intl_col = next((c for c in official.columns if '是否招收留学生' in str(c)), None)

        current_school = ''

        for _, row in official.iterrows():
            raw_name = clean_official_school_text(row.get(name_col, ''))
            if not raw_name:
                continue

            owner_school = clean_official_school_text(row.get(owner_col, '')) if owner_col else ''
            intl_flag = normalize_official_bool(row.get(intl_col, '')) if intl_col else None

            if owner_school and is_school_name(owner_school):
                current_school = owner_school
                school_map[normalize_text(owner_school)] = owner_school
                school_faculty_index.setdefault(owner_school, set())

            if is_school_name(raw_name):
                current_school = raw_name
                school_map[normalize_text(raw_name)] = raw_name
                school_faculty_index.setdefault(raw_name, set())
                continue

            school, faculty = split_school_and_faculty(raw_name)
            if school and is_school_name(school):
                current_school = school
                school_map[normalize_text(school)] = school
                school_faculty_index.setdefault(school, set())

            school_name = owner_school if (owner_school and is_school_name(owner_school)) else (school or current_school)
            faculty_name = faculty if faculty else raw_name
            faculty_name = clean_official_faculty_text(faculty_name, school_name)

            if not school_name or not faculty_name:
                continue

            if intl_flag is False:
                continue

            if not looks_like_official_faculty(faculty_name):
                continue

            normalized_value = normalize_text(faculty_name)
            if normalized_value:
                faculty_map[normalized_value] = faculty_name
                department_map[normalized_value] = faculty_name
                school_faculty_index.setdefault(school_name, set()).add(faculty_name)

    school_faculty_index = {
        school: sorted(list(faculties))
        for school, faculties in school_faculty_index.items()
        if len(faculties) > 0
    }

    return school_map, faculty_map, department_map, school_faculty_index


def map_official_name(value, mapping):
    if pd.isna(value):
        return ''
    raw = str(value).strip()
    normalized = normalize_text(raw)
    if not normalized:
        return raw
    if normalized in mapping:
        return mapping[normalized]

    # exact suffix match fallback
    for suffix in ['学部', '学科', '研究科', '学群', '学域', '系', 'コース', '学域', '専攻']:
        candidate = normalize_text(raw + suffix)
        if candidate in mapping:
            return mapping[candidate]

    # fuzzy matching based on simplified string containment
    candidates = [mapped for key, mapped in mapping.items() if normalized in key or key in normalized]
    if len(set(candidates)) == 1:
        return candidates[0]

    return raw


FACULTY_DISPLAY_MAP = {
    '工': '工学部',
    '経済': '経済学部',
    '経営': '経営学部',
    '文': '文学部',
    '商': '商学部',
    '法': '法学部',
    '社会': '社会学部',
    '国際': '国際学部',
    '理': '理学部',
    '教育': '教育学部',
    '医': '医学部',
    '薬': '薬学部',
    '歯': '歯学部',
    '情報': '情報学部',
    '理工': '理工学部',
    '先進理工': '先進理工学部',
    '基幹理工': '基幹理工学部',
    '創造理工': '創造理工学部',
    '教養': '教養学部',
}


def format_faculty_label(name):
    if pd.isna(name) or str(name).strip() == '':
        return str(name)
    label = str(name).strip()
    if label in FACULTY_DISPLAY_MAP:
        return FACULTY_DISPLAY_MAP[label]
    if re.search(r'(学部|研究科|学科|院|コース|系)$', label):
        return label
    if label.endswith('学'):
        return label + '部'
    if label.endswith('科'):
        return label
    return label + '学部'


def resolve_school_faculty_names(school, faculty):
    if EJU_DF is None:
        return school, faculty

    school_value = str(school or '').strip()
    faculty_value = str(faculty or '').strip()

    school_candidates = EJU_DF['受験校'].dropna().astype(str).tolist()
    faculty_candidates = []

    matched_school = school_value
    if school_value:
        for candidate in school_candidates:
            if normalize_text(candidate) == normalize_text(school_value):
                matched_school = candidate
                break
        else:
            for candidate in school_candidates:
                if normalize_text(school_value) in normalize_text(candidate) or normalize_text(candidate) in normalize_text(school_value):
                    matched_school = candidate
                    break

    if matched_school:
        faculty_candidates = EJU_DF[EJU_DF['受験校'] == matched_school]['学部／研究科'].dropna().astype(str).tolist()

    # Keep Tokyo University category-style selections as-is,
    # then rely on downstream contains matching to aggregate historical rows.
    tokyo_category_set = {'文科一類', '文科二類', '文科三類', '理科一類', '理科二類'}
    if matched_school == '東京大学' and faculty_value in tokyo_category_set:
        return matched_school, faculty_value

    matched_faculty = faculty_value
    if faculty_value and faculty_candidates:
        target_norm = normalize_text(faculty_value)
        faculty_norms = [normalize_text(candidate) for candidate in faculty_candidates]

        for candidate in faculty_candidates:
            cand_norm = normalize_text(candidate)
            if cand_norm == target_norm:
                matched_faculty = candidate
                break
            if target_norm and cand_norm and (
                target_norm in cand_norm or cand_norm in target_norm or
                any(token in cand_norm for token in target_norm.split()) or
                any(token in target_norm for token in cand_norm.split())
            ):
                matched_faculty = candidate
                break

        if matched_faculty == faculty_value:
            for candidate in faculty_candidates:
                cand_norm = normalize_text(candidate)
                if any(token in cand_norm for token in ['法学','文学','経済','工学','理学','医学','教育','商学','社会','国際','情報','薬学','歯学','看護','経営','教養','政治','心理','環境','生命']):
                    if any(token in target_norm for token in ['法学','文学','経済','工学','理学','医学','教育','商学','社会','国際','情報','薬学','歯学','看護','経営','教養','政治','心理','環境','生命']):
                        if any(token in cand_norm for token in target_norm.split()) or any(token in target_norm for token in cand_norm.split()):
                            matched_faculty = candidate
                            break

    return matched_school, matched_faculty

# Cache of pre-computed statistics
# Key: (school_name, faculty_name, track)
# Value: dict of stats
STATS_CACHE = {}

# Unique school-faculty index for UI dropdowns
SCHOOL_FACULTY_INDEX = {}

# Official school-faculty index parsed from university guideline CSVs
OFFICIAL_SCHOOL_FACULTY_INDEX = {}

# Official faculty name map parsed from university guideline CSVs
OFFICIAL_FACULTY_MAP = {}

# Global dataframe reference
EJU_DF = None

# Manual/dynamic school info maps to reduce default-hensachi fallbacks.
SCHOOL_INFO_OVERRIDES = {}
SCHOOL_INFO_DYNAMIC = {}

CAPITAL_REGION_KEYWORDS = [
    '東京', '神奈川', '横浜', '川崎', '千葉', '埼玉', '首都', '都立'
]
KANTO_REGION_KEYWORDS = [
    '茨城', '栃木', '群馬', '前橋', '宇都宮', '高崎', 'つくば'
]
KANSAI_REGION_KEYWORDS = [
    '大阪', '京都', '神戸', '兵庫', '奈良', '滋賀', '和歌山', '関西'
]

CAPITAL_REGION_SCHOOL_HINTS = {
    '早稲田大学', '慶應義塾大学', '上智大学', '学習院大学', '東京理科大学',
    '明治大学', '青山学院大学', '立教大学', '中央大学', '法政大学',
    '日本大学', '東洋大学', '専修大学', '駒澤大学', '明治学院大学',
    '國學院大學', '武蔵大学', '武蔵野大学', '芝浦工業大学', '工学院大学',
    '東京電機大学', '東京都市大学', '成蹊大学', '成城大学', '亜細亜大学'
}
KANSAI_REGION_SCHOOL_HINTS = {
    '関西大学', '関西学院大学', '同志社大学', '立命館大学', '龍谷大学',
    '京都産業大学', '近畿大学', '甲南大学', '京都橘大学', '佛教大学',
    '大阪経済大学', '大阪工業大学', '京都女子大学', '神戸学院大学'
}

MAJOR_DIRECTION_KEYWORDS = {
    'bunka_literature': ['文学', '人文', '日本語', '史学', '哲学', '言語', '文化'],
    'bunka_politics': ['政治', '法学', '政策', '公共', '国際関係', '行政'],
    'bunka_business': ['経営', '商学', '会計', 'ビジネス', 'マネジメント'],
    'bunka_economics': ['経済', '金融'],
    'bunka_liberal_arts': ['教養', '国際教養', 'リベラルアーツ', '総合文化', '文化構想'],
    'bunka_education': ['教育', '教員', '学校教育', '教育学'],
    'bunka_language': ['外国語', '英語', '中国語', 'フランス語', 'ドイツ語', 'スペイン語', '日本語教育', '言語教育', '通訳', '翻訳'],
    'rika_mechanical': ['機械', '機械工', 'メカトロ', 'ロボティクス'],
    'rika_electrical': ['電気', '電子', '通信', '制御', '電機'],
    'rika_math': ['数学', '数理', '統計'],
    'rika_architecture': ['建築', '都市工', '環境工学', '社会環境'],
    'rika_info': ['情報', '情報理工', '情報工学', '計算機', 'コンピュータ', 'AI', '知能'],
    'rika_medicine': ['医学', '医', '保健', '看護', '医療'],
    'rika_pharmacy': ['薬学', '薬'],
    'rika_dentistry': ['歯学', '歯'],
    'rika_agri_bio': ['農学', '農', '生物', '生命', 'バイオ', '応用生物', '生命科学']
}

def get_school_type(school_name):
    public_keywords = ['国立', '県立', '市立', '公立', '都立', '府立', '大学院大学', '札幌医科大学', '秋田公立美術大学', '国際教養大学', '公立はこだて未来大学', '兵庫県立大学']
    nationals = [
        '東京大学', '京都大学', '大阪大学', '東北大学', '名古屋大学', '九州大学', '北海道大学', 
        '一橋大学', '東京工業大学', '神戸大学', '筑波大学', '横浜国立大学', '千葉大学', '金沢大学', 
        '岡山大学', '広島大学', '電気通信大学', '東京外国語大学', '東京学芸大学', '東京芸術大学', 
        '東京農工大学', '東京医科歯科大学', 'お茶の水女子大学', '信州大学', '新潟大学', '熊本大学', 
        '埼玉大学', '静岡大学', '宇都宮大学', '茨城大学', '群馬大学', '滋賀大学', '山形大学', 
        '岩手大学', '琉球大学', '福井大学', '香川大学', '徳島大学', '愛媛大学', '高知大学', 
        '長崎大学', '大分大学', '宮崎大学', '鹿児島大学', '富山大学', '鳥取大学', '島根大学', '山口大学'
    ]
    if any(kw in school_name for kw in public_keywords) or any(nat in school_name for nat in nationals):
        return "国公立"
    return "私立"


def get_school_region_bucket(school_name):
    name = str(school_name or '').strip()
    if not name:
        return 'other'

    if name in CAPITAL_REGION_SCHOOL_HINTS:
        return 'capital'
    if name in KANSAI_REGION_SCHOOL_HINTS:
        return 'kansai'

    if any(k in name for k in CAPITAL_REGION_KEYWORDS):
        return 'capital'
    if any(k in name for k in KANSAI_REGION_KEYWORDS):
        return 'kansai'
    if any(k in name for k in KANTO_REGION_KEYWORDS):
        return 'kanto'
    return 'other'


def passes_region_filter(school_name, region_filter):
    region = str(region_filter or 'all').strip()
    if region in {'', 'all'}:
        return True

    bucket = get_school_region_bucket(school_name)

    if region == 'capital':
        return bucket == 'capital'
    if region == 'kanto':
        return bucket in {'capital', 'kanto'}
    if region == 'kansai':
        return bucket == 'kansai'
    if region == 'outside_capital':
        return bucket != 'capital'
    return True


def passes_school_type_filter(school_name, school_type_filter):
    type_filter = str(school_type_filter or 'all').strip()
    if type_filter in {'', 'all'}:
        return True

    school_type = get_school_type(school_name)
    if type_filter == 'public':
        return school_type == '国公立'
    if type_filter == 'private':
        return school_type == '私立'
    return True


def passes_major_direction_filter(track, faculty_name, major_direction_filter):
    direction = str(major_direction_filter or 'all').strip()
    if direction in {'', 'all'}:
        return True

    if direction.startswith('bunka_') and track != 'bunka':
        return False
    if direction.startswith('rika_') and track != 'rika':
        return False

    text = str(faculty_name or '').strip()
    keywords = MAJOR_DIRECTION_KEYWORDS.get(direction, [])
    if not keywords:
        return True
    return any(k in text for k in keywords)

def get_school_rank_tier(school_name):
    info = get_school_info(school_name)
    return f"{info['tier']} (偏差值: {info['hensachi']}, QS: {info['qs'] if info['qs'] < 9999 else '-'})"


def tier_from_hensachi(hensachi):
    h = float(hensachi or 50)
    if h >= 65:
        return 'S级'
    if h >= 60:
        return 'A级'
    if h >= 55:
        return 'B级'
    return 'C级'


def get_university_group(school_name, hensachi, school_type):
    name = str(school_name or '').strip()
    if name in UNIVERSITY_GROUP_MAP:
        return UNIVERSITY_GROUP_MAP[name]

    h = float(hensachi or 50)
    if h >= 64:
        return 'upper_public' if school_type == '国公立' else 'upper_private'
    if h >= 56:
        return 'mid_public' if school_type == '国公立' else 'mid_private'
    return 'standard_public' if school_type == '国公立' else 'standard_private'


def expected_score_floor(school_name, hensachi=None, track=None):
    # Backward-compatible shim for older call patterns during hot-reload.
    # Legacy pattern: expected_score_floor(hensachi, track)
    if track is None and isinstance(hensachi, str) and hensachi in {'bunka', 'rika'}:
        track = hensachi
        hensachi = school_name
        school_name = ''

    if track is None:
        track = 'bunka'

    if hensachi is None:
        hensachi = get_school_info(str(school_name or '')).get('hensachi', 50)

    # Cognition-aware floor: avoid implausibly low recommendations caused by noisy historical outliers.
    h = float(hensachi or 50)
    if track == 'rika':
        hensachi_floor = 490 + (h - 50) * 8.0
    else:
        hensachi_floor = 500 + (h - 50) * 8.0

    school_type = get_school_type(school_name)
    group_name = get_university_group(school_name, h, school_type)
    group_floor = UNIVERSITY_GROUP_BASELINES[group_name][track]

    # Soft calibration: do not fully override historical/hensachi baseline,
    # but keep floors aligned with mainstream university-group perception.
    baseline = max(hensachi_floor, group_floor - 25.0)
    return float(np.clip(baseline, 430, 730))


def load_hensachi_overrides(path='university_hensachi_overrides.csv'):
    resolved_path = path if os.path.isabs(path) else project_path(path)
    if not os.path.exists(resolved_path):
        return {}
    try:
        df = pd.read_csv(resolved_path, dtype=str, keep_default_na=False)
    except Exception:
        return {}

    if df.empty:
        return {}

    school_col = next((c for c in df.columns if str(c).strip() in {'school', '学校', '学校名', 'university'}), df.columns[0])
    hensachi_col = next((c for c in df.columns if 'hensachi' in str(c).lower() or '偏差值' in str(c) or '偏差値' in str(c)), None)
    qs_col = next((c for c in df.columns if str(c).strip().lower() == 'qs'), None)
    tier_col = next((c for c in df.columns if str(c).strip().lower() in {'tier', '等级', 'ランク'}), None)
    source_col = next((c for c in df.columns if str(c).strip().lower() in {'source', '来源', '出典'}), None)

    overrides = {}
    for _, row in df.iterrows():
        school = str(row.get(school_col, '')).strip()
        if not school or not hensachi_col:
            continue
        try:
            hensachi = float(str(row.get(hensachi_col, '')).strip())
        except Exception:
            continue

        try:
            qs = int(float(str(row.get(qs_col, '')).strip())) if qs_col and str(row.get(qs_col, '')).strip() else 9999
        except Exception:
            qs = 9999

        tier = str(row.get(tier_col, '')).strip() if tier_col else ''
        if not tier:
            tier = tier_from_hensachi(hensachi)

        source = str(row.get(source_col, '')).strip() if source_col else ''
        if not source:
            source = 'override_csv'

        overrides[school] = {
            'hensachi': int(round(hensachi)),
            'qs': qs,
            'tier': tier,
            'source': source
        }
    return overrides


def build_dynamic_school_info_from_cache(stats_cache):
    school_scores = {}
    for (school, _faculty, track_name), stats in stats_cache.items():
        if pd.isna(stats.get('median_score')):
            continue
        if int(stats.get('total_records', 0)) < 5:
            continue

        score = float(stats['median_score'])
        if track_name == 'rika':
            inferred_h = 50.0 + (score - 490.0) / 8.0
        else:
            inferred_h = 50.0 + (score - 500.0) / 8.0

        school_scores.setdefault(school, []).append(inferred_h)

    dynamic = {}
    for school, values in school_scores.items():
        h = float(np.median(values))
        school_type = get_school_type(school)
        group_name = get_university_group(school, h, school_type)
        group_default = GROUP_HENSACHI_DEFAULT.get(group_name, 52)

        # Softly pull inferred value toward group perception while keeping data-driven center.
        h = max(h, group_default - 1.0)
        h = float(np.clip(h, 42.0, 72.0))

        dynamic[school] = {
            'hensachi': int(round(h)),
            'qs': 9999,
            'tier': tier_from_hensachi(h),
            'source': 'dynamic_cache'
        }
    return dynamic

def get_school_info(school_name):
    UNIVERSITY_DATA = {
        '東京大学': {'hensachi': 72, 'qs': 32, 'tier': 'S级'},
        '京都大学': {'hensachi': 70, 'qs': 50, 'tier': 'S级'},
        '大阪大学': {'hensachi': 67, 'qs': 86, 'tier': 'S级'},
        '東京工業大学': {'hensachi': 67, 'qs': 84, 'tier': 'S级'},
        '東北大学': {'hensachi': 65, 'qs': 107, 'tier': 'A级'},
        '名古屋大学': {'hensachi': 65, 'qs': 152, 'tier': 'A级'},
        '九州大学': {'hensachi': 63, 'qs': 167, 'tier': 'A级'},
        '北海道大学': {'hensachi': 63, 'qs': 173, 'tier': 'A级'},
        '慶應義塾大学': {'hensachi': 68, 'qs': 188, 'tier': 'S级'},
        '早稲田大学': {'hensachi': 68, 'qs': 181, 'tier': 'S级'},
        '筑波大学': {'hensachi': 62, 'qs': 377, 'tier': 'A级'},
        '一橋大学': {'hensachi': 69, 'qs': 400, 'tier': 'S级'},
        '神戸大学': {'hensachi': 62, 'qs': 400, 'tier': 'A级'},
        '千葉大学': {'hensachi': 58, 'qs': 490, 'tier': 'B级'},
        '横浜国立大学': {'hensachi': 58, 'qs': 600, 'tier': 'A级'},
        '上智大学': {'hensachi': 65, 'qs': 700, 'tier': 'A级'},
        '東京理科大学': {'hensachi': 60, 'qs': 700, 'tier': 'A级'},
        '明治大学': {'hensachi': 62, 'qs': 800, 'tier': 'B级'},
        '青山学院大学': {'hensachi': 60, 'qs': 800, 'tier': 'B级'},
        '立教大学': {'hensachi': 60, 'qs': 800, 'tier': 'B级'},
        '中央大学': {'hensachi': 58, 'qs': 800, 'tier': 'B级'},
        '法政大学': {'hensachi': 58, 'qs': 800, 'tier': 'B级'},
        '関西大学': {'hensachi': 55, 'qs': 1000, 'tier': 'B级'},
        '関西学院大学': {'hensachi': 55, 'qs': 1000, 'tier': 'B级'},
        '同志社大学': {'hensachi': 60, 'qs': 800, 'tier': 'B级'},
        '立命館大学': {'hensachi': 55, 'qs': 1000, 'tier': 'B级'},
        '広島大学': {'hensachi': 57, 'qs': 500, 'tier': 'B级'},
        '金沢大学': {'hensachi': 56, 'qs': 600, 'tier': 'B级'},
        '岡山大学': {'hensachi': 56, 'qs': 600, 'tier': 'B级'},
        '東京都立大学': {'hensachi': 58, 'qs': 700, 'tier': 'B级'},
        'お茶の水女子大学': {'hensachi': 60, 'qs': 800, 'tier': 'A级'},
        '東京外国語大学': {'hensachi': 65, 'qs': 800, 'tier': 'A级'},
        '国際基督教大学': {'hensachi': 65, 'qs': 800, 'tier': 'A级'},
        '学習院大学': {'hensachi': 58, 'qs': 1000, 'tier': 'B级'}
    }
    for key, data in UNIVERSITY_DATA.items():
        if key in school_name:
            return data

    name = str(school_name or '').strip()

    for key, data in SCHOOL_INFO_OVERRIDES.items():
        if key == name or (key and key in name):
            return data

    for key, data in SCHOOL_INFO_DYNAMIC.items():
        if key == name or (key and key in name):
            return data

    school_type = get_school_type(name)
    group_name = get_university_group(name, 50, school_type)
    fallback_h = GROUP_HENSACHI_DEFAULT.get(group_name, 52)
    return {'hensachi': int(fallback_h), 'qs': 9999, 'tier': tier_from_hensachi(fallback_h), 'source': 'fallback_group'}


def display_official_faculty_name(school, faculty):
    raw = str(faculty or '').strip()
    school_name = str(school or '').strip()
    if not raw:
        return raw

    tokyo_category_set = {'文科一類', '文科二類', '文科三類', '理科一類', '理科二類'}
    if school_name == '東京大学' and raw in tokyo_category_set:
        return raw

    # Remove school prefix and route/class prefixes used in historical dataset
    cleaned = raw
    if school_name and school_name in cleaned:
        cleaned = cleaned.replace(school_name, '').strip()
    cleaned = re.sub(r'^(文科|理科)[一二三四五六七八九十]+類[-ー]?', '', cleaned).strip()

    # Prefer the right-most segment if the name is joined by '-'
    if any(ch in cleaned for ch in ['-', '‐', '–', '—']):
        segments = [seg.strip() for seg in re.split(r'[-‐–—]', cleaned) if seg.strip()]
        if len(segments) > 1:
            preferred = [seg for seg in segments if re.search(r'(学部|研究科|学科|専攻|コース|系)$', seg)]
            cleaned = preferred[-1] if preferred else segments[-1]

    normalized = normalize_text(cleaned)
    if normalized and normalized in OFFICIAL_FACULTY_MAP:
        return OFFICIAL_FACULTY_MAP[normalized]

    return format_faculty_label(cleaned)


def build_school_faculty_options(school, faculties):
    # Special display rule requested for University of Tokyo:
    # show only broad recruitment categories.
    if school == '東京大学':
        allowed = ['文科一類', '文科二類', '文科三類', '理科一類', '理科二類']
        present = set()
        for faculty in faculties:
            text = str(faculty or '').strip()
            m = re.search(r'(文科[一二三]類|理科[一二]類)', text)
            if m:
                present.add(m.group(1))

        ordered = [name for name in allowed if name in present]
        return [{'value': name, 'label': name} for name in ordered]

    options = []
    seen_labels = set()
    for faculty in faculties:
        label = display_official_faculty_name(school, faculty)
        if not label or label in {'学部', '研究科'}:
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        options.append({'value': faculty, 'label': label})

    return options


def validate_track_faculty_compatibility(school, faculty, track):
    school_name = str(school or '').strip()
    faculty_name = str(faculty or '').strip()
    track_name = str(track or '').strip()

    if school_name != '東京大学':
        return True, None

    if faculty_name.startswith('文科') and track_name == 'rika':
        return False, '东京大学文科一类/二类/三类仅面向文科生，当前理科方向不可报考。'

    if faculty_name.startswith('理科') and track_name == 'bunka':
        return False, '东京大学理科一类/二类仅面向理科生，当前文科方向不可报考。'

    return True, None

def load_and_precompute():
    global EJU_DF, STATS_CACHE, SCHOOL_FACULTY_INDEX, OFFICIAL_SCHOOL_FACULTY_INDEX, OFFICIAL_FACULTY_MAP, SCHOOL_INFO_OVERRIDES, SCHOOL_INFO_DYNAMIC
    print("Loading data and precomputing stats...")
    
    data_path = project_path('cleaned_eju_data.csv')
    if not os.path.exists(data_path):
        print("cleaned_eju_data.csv not found! Please run clean_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Use cleaned display fields if present, otherwise normalize names against official 2025 wording
    school_map, faculty_map, department_map, official_school_faculty_index = load_official_name_maps()
    if '学部／研究科_display' in df.columns and '学科／専攻_display' in df.columns:
        df['学部／研究科'] = df['学部／研究科_display'].fillna(df['学部／研究科'])
        df['学科／専攻'] = df['学科／専攻_display'].fillna(df['学科／専攻'])
    elif school_map or faculty_map or department_map:
        df['受験校'] = df['受験校'].apply(lambda v: map_official_name(v, school_map))
        df['学部／研究科'] = df['学部／研究科'].apply(lambda v: map_official_name(v, faculty_map))
        df['学科／専攻'] = df['学科／専攻'].apply(lambda v: map_official_name(v, department_map))
    
    # Pre-fill max values of June and Nov
    df['日本語'] = np.maximum(df['日本語_1'].fillna(0), df['日本語_2'].fillna(0))
    df['記述'] = np.maximum(df['記述_1'].fillna(0), df['記述_2'].fillna(0))
    df['综合'] = np.maximum(df['総合_1'].fillna(0), df['総合_2'].fillna(0))
    df['数１'] = np.maximum(df['数１_1'].fillna(0), df['数１_2'].fillna(0))
    df['数２'] = np.maximum(df['数２_1'].fillna(0), df['数２_2'].fillna(0))
    df['物理'] = np.maximum(df['物理_1'].fillna(0), df['物理_2'].fillna(0))
    df['化学'] = np.maximum(df['化学_1'].fillna(0), df['化学_2'].fillna(0))
    df['生物'] = np.maximum(df['生物_1'].fillna(0), df['生物_2'].fillna(0))
    df['iBT'] = df['iBT'].fillna(0)
    df['TOEIC'] = df['TOEIC'].fillna(0)
    
    df['is_bunka'] = df['综合'] > 0
    df['is_rika'] = (df['物理'] > 0) | (df['化学'] > 0) | (df['生物'] > 0)
    
    # Flags for complete scores (avoid missing partial data dragging down medians)
    df['has_full_bunka'] = (df['日本語'] > 0) & (df['综合'] > 0) & (df['数１'] > 0)
    df['has_full_rika'] = (df['日本語'] > 0) & df['is_rika'] & (df['数２'] > 0)
    
    df['文科EJU总分'] = df['日本語'] + df['综合'] + df['数１']
    
    science_sum = df[['物理', '化学', '生物']].fillna(0).values
    science_sum.sort(axis=1)
    df['理科EJU总分'] = df['日本語'] + science_sum[:, -1] + science_sum[:, -2] + df['数２']
    
    EJU_DF = df
    
    # Standardize string fields
    EJU_DF['受験校'] = EJU_DF['受験校'].fillna('未知大学').str.strip()
    EJU_DF['学部／研究科'] = EJU_DF['学部／研究科'].fillna('全学部').str.strip()
    EJU_DF['学科／専攻'] = EJU_DF['学科／専攻'].fillna('全学科').str.strip()

    # Normalize faculty labels to official display form so index values are consistent
    EJU_DF['学部／研究科'] = EJU_DF['学部／研究科'].apply(lambda v: format_faculty_label(v))

    # 1. Build SCHOOL_FACULTY_INDEX
    # { 'Tokyo U': ['Faculty of Letters', 'Faculty of Economics', ...], ... }
    SCHOOL_FACULTY_INDEX = {}
    schools = EJU_DF['受験校'].unique()
    for school in sorted(schools):
        faculties = EJU_DF[EJU_DF['受験校'] == school]['学部／研究科'].unique()
        display_faculties = []
        for faculty in faculties:
            display_faculty = faculty_map.get(normalize_text(faculty), format_faculty_label(faculty))
            display_faculties.append(display_faculty)

        display_school = school_map.get(normalize_text(school), school)
        SCHOOL_FACULTY_INDEX[display_school] = sorted({str(f) for f in display_faculties})

    # Prefer official names from recruitment guideline CSV for dropdown display
    OFFICIAL_SCHOOL_FACULTY_INDEX = {
        school: faculties
        for school, faculties in sorted(official_school_faculty_index.items(), key=lambda item: item[0])
    }
    OFFICIAL_FACULTY_MAP = faculty_map.copy()
        
    # 2. Build STATS_CACHE for each unique (school, faculty, track)
    grouped_bunka = EJU_DF[EJU_DF['is_bunka']].groupby(['受験校', '学部／研究科'])
    grouped_rika = EJU_DF[EJU_DF['is_rika']].groupby(['受験校', '学部／研究科'])
    
    def process_groups(grouped_obj, track_name, score_col, full_flag_col):
        for (school, faculty), group in grouped_obj:
            # Filter out incomplete scores to prevent medians from being dragged down
            full_group = group[group[full_flag_col]]
            
            passed = full_group[full_group['合否'] == 1][score_col].dropna()
            failed = full_group[full_group['合否'] == 0][score_col].dropna()
            
            total_records = len(group) # Total records including partial
            passed_count = len(group[group['合否'] == 1])
            failed_count = len(group[group['合否'] == 0])
            
            if len(passed) > 0:
                passed_arr = np.array(passed, dtype=float)
                passed_arr.sort()

                q10 = float(np.percentile(passed_arr, 10))
                q25 = float(np.percentile(passed_arr, 25))
                q50 = float(np.percentile(passed_arr, 50))
                q75 = float(np.percentile(passed_arr, 75))
                q90 = float(np.percentile(passed_arr, 90))

                robust_center = (q25 + q50 + q75) / 3.0 if len(passed_arr) >= 8 else q50

                school_info = get_school_info(school)
                score_floor = expected_score_floor(school, school_info.get('hensachi', 50), track_name)

                # Keep recommendations aligned with common perception of school difficulty
                # while still respecting observed data when sample size is sufficient.
                floor_tolerance = 35.0 if len(passed_arr) >= 15 else 20.0
                median_score = max(robust_center, score_floor - floor_tolerance)

                iqr = max(1.0, q75 - q25)
                std_score = max(18.0, iqr / 1.349)
                if std_score == 0:
                    std_score = 20.0
                min_passed = max(float(np.min(passed_arr)), score_floor - 85.0)
                max_passed = float(np.max(passed_arr))
                
                if len(passed) >= 2:
                    target_q1 = max(q10, score_floor - 45.0)
                    target_q3 = max(q90, score_floor - 5.0)
                else:
                    target_q1 = float(passed.iloc[0])
                    target_q3 = float(passed.iloc[0])
                
                # Auxiliary check using highest failed score
                if len(failed) > 0:
                    max_failed = float(failed.max())
                    # If median is somehow lower than the highest failed score significantly, 
                    # we can shift the median up slightly to be safe
                    if max_failed > median_score:
                        median_score = (median_score + max_failed) / 2
            else:
                median_score = np.nan
                std_score = np.nan
                min_passed = np.nan
                max_passed = np.nan
                target_q1 = np.nan
                target_q3 = np.nan
                
            # Average English scores for accepted students
            passed_ibt = group[(group['合否'] == 1) & (group['iBT'] > 0)]['iBT']
            passed_toeic = group[(group['合否'] == 1) & (group['TOEIC'] > 0)]['TOEIC']
            passed_kijutsu = group[(group['合否'] == 1) & (group['記述'] > 0)]['記述']
            
            avg_ibt = float(passed_ibt.mean()) if len(passed_ibt) > 0 else None
            avg_toeic = float(passed_toeic.mean()) if len(passed_toeic) > 0 else None
            avg_kijutsu = float(passed_kijutsu.mean()) if len(passed_kijutsu) > 0 else None
                
            STATS_CACHE[(school, faculty, track_name)] = {
                'total_records': total_records,
                'passed_count': passed_count,
                'failed_count': failed_count,
                'median_score': median_score,
                'std_score': std_score,
                'min_passed': min_passed,
                'max_passed': max_passed,
                'target_range_q1': target_q1,
                'target_range_q3': target_q3,
                'avg_ibt': avg_ibt,
                'avg_toeic': avg_toeic,
                'avg_kijutsu': avg_kijutsu
            }
            
    process_groups(grouped_bunka, 'bunka', '文科EJU总分', 'has_full_bunka')
    process_groups(grouped_rika, 'rika', '理科EJU总分', 'has_full_rika')

    # Refresh school-info sources after cache is built.
    SCHOOL_INFO_OVERRIDES = load_hensachi_overrides()
    SCHOOL_INFO_DYNAMIC = build_dynamic_school_info_from_cache(STATS_CACHE)
    
    print(f"Precomputed {len(STATS_CACHE)} school-faculty-track profiles.")


def get_prediction_for_profile(user_score, ibt_score, toeic_score, stats, school_info=None, school_name='', faculty_name=''):
    def deterministic_jitter(seed_text):
        digest = hashlib.md5(seed_text.encode('utf-8')).hexdigest()
        value = int(digest[:8], 16) / float(0xFFFFFFFF)
        return (value - 0.5) * 0.012

    def calibrate_high_band(prob_value, z_value, score_value, stats_value, school_meta, school, faculty):
        capped_prob = float(np.clip(prob_value, 0.01, PROBABILITY_CAP))
        if capped_prob < 0.90:
            return capped_prob

        q3 = stats_value.get('target_range_q3')
        median = stats_value.get('median_score')
        anchor = q3 if pd.notnull(q3) else median
        margin = max(0.0, float(score_value) - float(anchor)) if pd.notnull(anchor) else 0.0

        z_safe = max(0.0, float(z_value))
        records = float(stats_value.get('total_records', 0) or 0)

        z_score = 1.0 - np.exp(-z_safe / 1.8)
        margin_score = 1.0 - np.exp(-margin / 45.0)
        sample_score = 1.0 - np.exp(-records / 55.0)
        confidence = float(np.clip(0.60 * z_score + 0.25 * margin_score + 0.15 * sample_score, 0.0, 1.0))

        # Map very-strong profiles into 90%-99% while preserving gradient.
        band_target = 0.90 + 0.09 * confidence

        hensachi = float((school_meta or {}).get('hensachi', 55) or 55)
        qs_raw = (school_meta or {}).get('qs', 9999)
        try:
            qs = int(float(qs_raw))
        except Exception:
            qs = 9999

        # Harder schools receive a few percentage points penalty.
        hensachi_penalty = max(0.0, (hensachi - 58.0) * 0.0018)
        if qs <= 100:
            qs_penalty = 0.015
        elif qs <= 200:
            qs_penalty = 0.010
        elif qs <= 500:
            qs_penalty = 0.006
        elif qs <= 1000:
            qs_penalty = 0.003
        else:
            qs_penalty = 0.0

        seed = f'{school}|{faculty}|{int(float(score_value)//5)}|{int(hensachi)}'
        jitter = deterministic_jitter(seed)

        adjusted = band_target - hensachi_penalty - qs_penalty + jitter
        # Keep within high band and avoid over-inflating beyond raw model output too much.
        adjusted = min(adjusted, capped_prob + 0.02)
        return float(np.clip(adjusted, 0.90, PROBABILITY_CAP))

    # Calculate base probability on EJU score
    median = stats['median_score']
    std = stats['std_score']
    
    if pd.isna(median) or pd.isna(std):
        return 0.5, "由于缺乏合格数据，我们无法精确评估概率"
        
    z = (user_score - median) / std
    
    # 结合JASSO分布和偏差值的修正模型
    # 假设EJU成绩呈正态分布，将绝对分数差距转换为相对竞争力
    base_prob = 1.0 / (1.0 + np.exp(-(1.5 * z + 0.6)))
    
    # 获取大学层级和偏差值带来的竞争热度惩罚/奖励
    if school_info:
        hensachi = school_info.get('hensachi', 50)
        # 偏差值越高，容错率越低，曲线更陡峭
        steepness = 1.0 + (hensachi - 50) / 20.0
        
        # 模拟“留学生录取名额”与“实际入学者数”的报录比压力 (Heuristic)
        # 偏差值高的学校通常名额少、竞争激烈，实际入学者/录取名额比率高，导致录取率下降
        # 偏差值低的学校可能为了招满学生，实际发出的offer多于名额
        competitiveness_penalty = (hensachi - 50) / 100.0 # -0.1 to 0.2
        
        adjusted_z = z * steepness - competitiveness_penalty
        prob = 1.0 / (1.0 + np.exp(-(1.5 * adjusted_z + 0.6)))
    else:
        prob = base_prob
    
    # Adjust for English if historical data indicates English requirements
    english_discount = 1.0
    if stats['avg_ibt'] and stats['avg_ibt'] > 40:
        # If school requires/takes TOEFL and user entered a low score or 0
        req_ibt = stats['avg_ibt']
        if ibt_score == 0:
            if req_ibt > 70: # High TOEFL requirement
                english_discount = 0.5 # cut probability in half if no TOEFL
            else:
                english_discount = 0.8
        else:
            # Scale down if TOEFL score is lower than average accepted
            ibt_ratio = ibt_score / req_ibt
            if ibt_ratio < 0.8:
                english_discount = max(0.5, ibt_ratio)
                
    elif stats['avg_toeic'] and stats['avg_toeic'] > 400:
        req_toeic = stats['avg_toeic']
        if toeic_score == 0:
            if req_toeic > 750:
                english_discount = 0.5
            else:
                english_discount = 0.8
        else:
            toeic_ratio = toeic_score / req_toeic
            if toeic_ratio < 0.8:
                english_discount = max(0.5, toeic_ratio)
                
    prob = prob * english_discount
    
    # Holistic admissions: EJU is important but not sufficient (interview/school exam are common).
    prob = prob * 0.96
    prob = np.clip(prob, 0.01, PROBABILITY_CAP)
    prob = calibrate_high_band(prob, z, user_score, stats, school_info or {}, school_name, faculty_name)
    
    return float(prob), None


def build_target_school_detail(school, faculty, track):
    resolved_school, resolved_faculty = resolve_school_faculty_names(school, faculty)
    school_mask = EJU_DF['受験校'] == resolved_school
    faculty_mask = EJU_DF['学部／研究科'] == resolved_faculty
    track_mask = EJU_DF['is_bunka'] if track == 'bunka' else EJU_DF['is_rika']

    sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
    if len(sub_df) == 0 and '学部／研究科_raw' in EJU_DF.columns:
        faculty_mask = EJU_DF['学部／研究科_raw'] == faculty
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]

    if len(sub_df) == 0:
        short_faculty = faculty
        if '学部' in short_faculty and '学部' not in resolved_faculty:
            short_faculty = short_faculty.replace('学部', '')
        faculty_mask = EJU_DF['学部／研究科'].astype(str).str.contains(short_faculty, na=False, regex=False)
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]

    if len(sub_df) == 0:
        sub_df = EJU_DF[school_mask & track_mask]

    score_col = '文科EJU总分' if track == 'bunka' else '理科EJU总分'
    full_flag_col = 'has_full_bunka' if track == 'bunka' else 'has_full_rika'

    valid_sub_df = sub_df[sub_df[full_flag_col] == True]
    scores_passed = valid_sub_df[valid_sub_df['合否'] == 1][score_col].dropna().tolist()
    scores_failed = valid_sub_df[valid_sub_df['合否'] == 0][score_col].dropna().tolist()

    cache_key = (resolved_school, resolved_faculty, track)
    stats = STATS_CACHE.get(cache_key, None)
    if stats is None:
        stats = STATS_CACHE.get((school, faculty, track), None)

    stats_data = {}
    if stats:
        stats_data = {
            'total_records': int(stats['total_records']),
            'passed_count': int(stats['passed_count']),
            'failed_count': int(stats['failed_count']),
            'median_score': float(stats['median_score']) if pd.notnull(stats['median_score']) else None,
            'min_passed': float(stats['min_passed']) if pd.notnull(stats['min_passed']) else None,
            'max_passed': float(stats['max_passed']) if pd.notnull(stats['max_passed']) else None,
            'avg_ibt': float(stats['avg_ibt']) if stats['avg_ibt'] else None,
            'avg_toeic': float(stats['avg_toeic']) if stats['avg_toeic'] else None,
            'avg_kijutsu': float(stats['avg_kijutsu']) if stats['avg_kijutsu'] else None
        }
    else:
        stats_data = {
            'passed_count': len(scores_passed),
            'failed_count': len(scores_failed),
            'median_score': float(np.median(scores_passed)) if len(scores_passed) > 0 else None,
            'min_passed': float(np.min(scores_passed)) if len(scores_passed) > 0 else None,
            'max_passed': float(np.max(scores_passed)) if len(scores_passed) > 0 else None,
            'avg_ibt': None,
            'avg_toeic': None,
            'avg_kijutsu': None
        }

    if len(scores_passed) >= 2:
        target_q1 = float(np.percentile(scores_passed, 10))
        target_q3 = float(np.percentile(scores_passed, 90))
        stats_data['target_range_q1'] = target_q1
        stats_data['target_range_q3'] = target_q3
    elif len(scores_passed) == 1:
        stats_data['target_range_q1'] = float(scores_passed[0])
        stats_data['target_range_q3'] = float(scores_passed[0])
    else:
        stats_data['target_range_q1'] = None
        stats_data['target_range_q3'] = None

    accepted_rows = valid_sub_df[valid_sub_df['合否'] == 1]
    subject_list = []
    if track == 'bunka':
        subject_list = [
            ('日本語', 'EJU日语'),
            ('記述', '日语记述'),
            ('综合', '文科综合'),
            ('数１', '数学 课程1')
        ]
    else:
        subject_list = [
            ('日本語', 'EJU日语'),
            ('記述', '日语记述'),
            ('数２', '数学 课程2'),
            ('物理', '物理'),
            ('化学', '化学'),
            ('生物', '生物')
        ]

    subject_scores = []
    for col, label in subject_list:
        values = accepted_rows[col].dropna()
        values = values[values > 0]
        if len(values) > 0:
            median = float(values.median())
            avg = float(values.mean())
            p10 = float(np.percentile(values, 10)) if len(values) >= 2 else median
            p90 = float(np.percentile(values, 90)) if len(values) >= 2 else median
            subject_scores.append({
                'key': col,
                'name': label,
                'median': median,
                'avg': avg,
                'p10': p10,
                'p90': p90,
                'min': float(values.min()),
                'max': float(values.max()),
                'count': int(len(values))
            })
        else:
            subject_scores.append({
                'key': col,
                'name': label,
                'median': None,
                'avg': None,
                'p10': None,
                'p90': None,
                'min': None,
                'max': None,
                'count': 0
            })

    stats_data['subject_scores'] = subject_scores

    return {
        'school': school,
        'faculty': faculty,
        'track': track,
        'school_type': get_school_type(resolved_school),
        'school_rank': get_school_rank_tier(resolved_school),
        'stats': stats_data,
        'scores_passed': scores_passed,
        'scores_failed': scores_failed
    }


@app.route('/')
def home():
    return send_file('templates/index.html')

@app.route('/api/schools', methods=['GET'])
def get_schools():
    display_index = {}
    all_schools = sorted(set(SCHOOL_FACULTY_INDEX.keys()) | set(OFFICIAL_SCHOOL_FACULTY_INDEX.keys()))
    for school in all_schools:
        historical_faculties = SCHOOL_FACULTY_INDEX.get(school, [])
        official_faculties = OFFICIAL_SCHOOL_FACULTY_INDEX.get(school, [])
        # Keep official wording while preserving historical faculties that may include bunka options.
        source_faculties = list(dict.fromkeys(list(official_faculties) + list(historical_faculties)))
        display_index[school] = build_school_faculty_options(school, source_faculties)

    return jsonify({
        'schools': all_schools,
        'index': display_index
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    
    track = data.get('track', 'bunka') # 'bunka' or 'rika'
    ja = float(data.get('ja', 280))
    kijutsu = float(data.get('kijutsu', 0))
    math = float(data.get('math', 120))
    sub = float(data.get('sub', 120)) # Sogou for bunka, or sum of 2 science for rika
    ibt = float(data.get('ibt', 0))
    toeic = float(data.get('toeic', 0))
    
    # Calculate user EJU total
    user_score = ja + math + sub
    
    # Find precise target if provided
    target_school = data.get('target_school')
    target_faculty = data.get('target_faculty')
    mode = data.get('mode', 'prediction')
    region_filter = data.get('region_filter', 'all')
    school_type_filter = data.get('school_type_filter', 'all')
    major_direction_filter = data.get('major_direction_filter', 'all')

    target_res = None
    if target_school and target_faculty:
        is_compatible, incompatible_msg = validate_track_faculty_compatibility(target_school, target_faculty, track)
        if not is_compatible:
            target_res = {
                'error': True,
                'message': incompatible_msg
            }
        else:
            detail = build_target_school_detail(target_school, target_faculty, track)
            has_history = detail['stats'].get('passed_count', 0) + detail['stats'].get('failed_count', 0) > 0
            if has_history:
                if mode == 'target_setting':
                    target_res = {
                        'school': target_school,
                        'faculty': target_faculty,
                        'faculty_display': display_official_faculty_name(target_school, target_faculty),
                        'school_rank': detail['school_rank'],
                        'school_type': detail['school_type'],
                        'mode': 'target_setting',
                        'stats': detail['stats'],
                        'note': f"基于 {detail['stats']['passed_count'] + detail['stats']['failed_count']} 条历史样本，给出该目标学校学部的总分与各科目标设定区间"
                    }
                else:
                    resolved_school, resolved_faculty = resolve_school_faculty_names(target_school, target_faculty)
                    stats = STATS_CACHE.get((resolved_school, resolved_faculty, track))
                    if stats is None:
                        stats = STATS_CACHE.get((target_school, target_faculty, track))
                    if stats is None and len(detail.get('scores_passed', [])) > 0:
                        passed_scores = np.array(detail.get('scores_passed', []), dtype=float)
                        std_score = float(np.std(passed_scores)) if len(passed_scores) > 1 else 20.0
                        if std_score == 0:
                            std_score = 20.0
                        stats = {
                            'median_score': float(np.median(passed_scores)),
                            'std_score': std_score,
                            'avg_ibt': detail['stats'].get('avg_ibt'),
                            'avg_toeic': detail['stats'].get('avg_toeic')
                        }

                    if stats is not None:
                        prob, msg = get_prediction_for_profile(user_score, ibt, toeic, stats, get_school_info(target_school), target_school, target_faculty)

                        if prob >= 0.70:
                            tier = "保底"
                            tier_color = "green"
                        elif prob >= 0.40:
                            tier = "适中"
                            tier_color = "yellow"
                        else:
                            tier = "冲刺"
                            tier_color = "red"

                        target_res = {
                            'school': target_school,
                            'faculty': target_faculty,
                            'faculty_display': display_official_faculty_name(target_school, target_faculty),
                            'school_rank': detail['school_rank'],
                            'school_type': detail['school_type'],
                            'probability': prob,
                            'tier': tier,
                            'tier_color': tier_color,
                            'stats': detail['stats'],
                            'note': msg or f"基于该专业 {detail['stats']['passed_count'] + detail['stats']['failed_count']} 条历史录取数据分析"
                        }
                    else:
                        target_res = {
                            'error': True,
                            'message': "未找到该专业针对此文理科方向的历史录取数据，建议查看该校其他学部或使用推荐功能。"
                        }
            else:
                target_res = {
                    'error': True,
                    'message': "未找到该专业针对此文理科方向的历史录取数据，建议查看该校其他学部或使用推荐功能。"
                }
            
    # Calculate recommendations
    # Filter all STATS_CACHE keys matching the track
    recommendations = []
    for (school, faculty, cache_track), stats in STATS_CACHE.items():
        if cache_track == track and pd.notnull(stats['median_score']) and stats['total_records'] >= 5:
            if not passes_region_filter(school, region_filter):
                continue
            if not passes_school_type_filter(school, school_type_filter):
                continue
            if not passes_major_direction_filter(track, faculty, major_direction_filter):
                continue

            prob, _ = get_prediction_for_profile(user_score, ibt, toeic, stats, get_school_info(school), school, faculty)
            
            if prob >= 0.70:
                tier = "保底"
                tier_sort = 2
            elif prob >= 0.40:
                tier = "适中"
                tier_sort = 1
            elif prob >= 0.15:
                tier = "冲刺"
                tier_sort = 0
            else:
                continue # Probability too low to recommend
                
            school_type = get_school_type(school)
            school_rank = get_school_rank_tier(school)
            school_info = get_school_info(school)
            
            recommendations.append({
                'school': school,
                'faculty': faculty,
                'faculty_display': display_official_faculty_name(school, faculty),
                'probability': prob,
                'tier': tier,
                'tier_sort': tier_sort,
                'school_type': school_type,
                'school_rank': school_rank,
                'hensachi': school_info['hensachi'],
                'qs': school_info['qs'],
                'median_score': float(stats['median_score']),
                'total_records': int(stats['total_records']),
                'avg_kijutsu': float(stats['avg_kijutsu']) if stats['avg_kijutsu'] else None
            })
            
    # Sort recommendations by tier, then hensachi (descending), qs (ascending), then probability
    recommendations = sorted(recommendations, key=lambda x: (x['tier_sort'], x['hensachi'], -x['qs'], x['probability']), reverse=True)
    
    # Split into list
    safety = [r for r in recommendations if r['tier'] == "保底"]
    match = [r for r in recommendations if r['tier'] == "适中"]
    challenger = [r for r in recommendations if r['tier'] == "冲刺"]
    
    return jsonify({
        'user_score': user_score,
        'target_result': target_res,
        'applied_filters': {
            'region_filter': region_filter,
            'school_type_filter': school_type_filter,
            'major_direction_filter': major_direction_filter
        },
        'recommendations': {
            'safety': safety[:30],       # top 30 recommendations
            'match': match[:30],
            'challenger': challenger[:30],
            'total_found': len(recommendations)
        }
    })

@app.route('/api/school-detail', methods=['POST'])
def get_school_detail():
    data = request.json or {}
    school = data.get('school')
    faculty = data.get('faculty')
    track = data.get('track', 'bunka')
    
    if not school or not faculty:
        return jsonify({'error': 'Missing parameters'}), 400

    is_compatible, incompatible_msg = validate_track_faculty_compatibility(school, faculty, track)
    if not is_compatible:
        return jsonify({'error': incompatible_msg}), 400
        
    resolved_school, resolved_faculty = resolve_school_faculty_names(school, faculty)

    # Query raw rows
    school_mask = EJU_DF['受験校'] == resolved_school
    faculty_mask = EJU_DF['学部／研究科'] == resolved_faculty
    track_mask = EJU_DF['is_bunka'] if track == 'bunka' else EJU_DF['is_rika']
    
    sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
    if len(sub_df) == 0 and '学部／研究科_raw' in EJU_DF.columns:
        faculty_mask = EJU_DF['学部／研究科_raw'] == faculty
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]

    if len(sub_df) == 0:
        short_faculty = faculty
        if '学部' in short_faculty and '学部' not in resolved_faculty:
            short_faculty = short_faculty.replace('学部', '')
        faculty_mask = EJU_DF['学部／研究科'].astype(str).str.contains(short_faculty, na=False, regex=False)
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
    
    if len(sub_df) == 0:
        # Fallback to school level
        sub_df = EJU_DF[school_mask & track_mask]
        
    score_col = '文科EJU总分' if track == 'bunka' else '理科EJU总分'
    full_flag_col = 'has_full_bunka' if track == 'bunka' else 'has_full_rika'
    
    # Only return full valid scores for the detail chart
    valid_sub_df = sub_df[sub_df[full_flag_col] == True]
    
    scores_passed = valid_sub_df[valid_sub_df['合否'] == 1][score_col].dropna().tolist()
    scores_failed = valid_sub_df[valid_sub_df['合否'] == 0][score_col].dropna().tolist()
    
    # Get standard stats
    cache_key = (resolved_school, resolved_faculty, track)
    stats = STATS_CACHE.get(cache_key, None)
    if stats is None:
        stats = STATS_CACHE.get((school, faculty, track), None)
    
    stats_data = {}
    if stats:
        stats_data = {
            'passed_count': int(stats['passed_count']),
            'failed_count': int(stats['failed_count']),
            'median_score': float(stats['median_score']) if pd.notnull(stats['median_score']) else None,
            'min_passed': float(stats['min_passed']) if pd.notnull(stats['min_passed']) else None,
            'max_passed': float(stats['max_passed']) if pd.notnull(stats['max_passed']) else None,
            'avg_ibt': float(stats['avg_ibt']) if stats['avg_ibt'] else None,
            'avg_toeic': float(stats['avg_toeic']) if stats['avg_toeic'] else None,
            'avg_kijutsu': float(stats['avg_kijutsu']) if stats['avg_kijutsu'] else None
        }
    else:
        stats_data = {
            'passed_count': len(scores_passed),
            'failed_count': len(scores_failed),
            'median_score': float(np.median(scores_passed)) if len(scores_passed) > 0 else None,
            'min_passed': float(np.min(scores_passed)) if len(scores_passed) > 0 else None,
            'max_passed': float(np.max(scores_passed)) if len(scores_passed) > 0 else None,
            'avg_ibt': None,
            'avg_toeic': None,
            'avg_kijutsu': None
        }
        
    # Calculate target score range (10th to 90th percentile, covers 80% of passed students)
    if len(scores_passed) >= 2:
        target_q1 = float(np.percentile(scores_passed, 10))
        target_q3 = float(np.percentile(scores_passed, 90))
        stats_data['target_range_q1'] = target_q1
        stats_data['target_range_q3'] = target_q3
    elif len(scores_passed) == 1:
        stats_data['target_range_q1'] = float(scores_passed[0])
        stats_data['target_range_q3'] = float(scores_passed[0])
    else:
        stats_data['target_range_q1'] = None
        stats_data['target_range_q3'] = None

    # Calculate per-subject score targets for accepted students
    accepted_rows = valid_sub_df[valid_sub_df['合否'] == 1]
    subject_list = []
    if track == 'bunka':
        subject_list = [
            ('日本語', 'EJU日语'),
            ('記述', '日语记述'),
            ('综合', '文科综合'),
            ('数１', '数学 课程1')
        ]
    else:
        subject_list = [
            ('日本語', 'EJU日语'),
            ('記述', '日语记述'),
            ('数２', '数学 课程2'),
            ('物理', '物理'),
            ('化学', '化学'),
            ('生物', '生物')
        ]

    subject_scores = []
    for col, label in subject_list:
        values = accepted_rows[col].dropna()
        values = values[values > 0]
        if len(values) > 0:
            median = float(values.median())
            avg = float(values.mean())
            p10 = float(np.percentile(values, 10)) if len(values) >= 2 else median
            p90 = float(np.percentile(values, 90)) if len(values) >= 2 else median
            subject_scores.append({
                'key': col,
                'name': label,
                'median': median,
                'avg': avg,
                'p10': p10,
                'p90': p90,
                'min': float(values.min()),
                'max': float(values.max()),
                'count': int(len(values))
            })
        else:
            subject_scores.append({
                'key': col,
                'name': label,
                'median': None,
                'avg': None,
                'p10': None,
                'p90': None,
                'min': None,
                'max': None,
                'count': 0
            })

    stats_data['subject_scores'] = subject_scores
    
    return jsonify({
        'school': school,
        'faculty': faculty,
        'faculty_display': display_official_faculty_name(school, faculty),
        'track': track,
        'school_type': get_school_type(resolved_school),
        'school_rank': get_school_rank_tier(resolved_school),
        'stats': stats_data,
        'scores_passed': scores_passed,
        'scores_failed': scores_failed
    })


if __name__ == '__main__':
    if not APP_PRECOMPUTED:
        load_and_precompute()
        APP_PRECOMPUTED = True
    app.run(
        debug=os.environ.get('FLASK_DEBUG', '0') == '1',
        host='0.0.0.0',
        port=int(os.environ.get('PORT', '5001'))
    )


# Ensure stats cache is ready when served by WSGI servers like gunicorn.
if not APP_PRECOMPUTED:
    load_and_precompute()
    APP_PRECOMPUTED = True
