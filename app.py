import os
import pandas as pd
import numpy as np
import re
import hashlib
from urllib.parse import urlparse
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
    '東京工業大学': 'top_elite', '東京科学大学': 'top_elite', '神戸大学': 'top_elite', '早稲田大学': 'top_elite', '慶應義塾大学': 'top_elite',

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


SCHOOL_NAME_ALIASES = {
    normalize_text('首都大学東京'): '東京都立大学',
    normalize_text('首都大学东京'): '東京都立大学',
    normalize_text('大阪市立大学'): '大阪公立大学',
    normalize_text('大阪府立大学'): '大阪公立大学',
    normalize_text('東京工業大学'): '東京科学大学',
    normalize_text('東京医科歯科大学'): '東京科学大学',
    normalize_text('东京工业大学'): '東京科学大学',
    normalize_text('东京医科齿科大学'): '東京科学大学',
    normalize_text('早稻田大学'): '早稲田大学',
    normalize_text('庆应义塾大学'): '慶應義塾大学',
    normalize_text('庆应大学'): '慶應義塾大学',
    normalize_text('东京大学'): '東京大学',
    normalize_text('东北大学'): '東北大学',
    normalize_text('东京理科大学'): '東京理科大学',
    normalize_text('东京农工大学'): '東京農工大学',
    normalize_text('北京语言大学'): '北京語言大学',
}


def canonicalize_school_name(value):
    raw = str(value or '').strip()
    if not raw:
        return raw
    return SCHOOL_NAME_ALIASES.get(normalize_text(raw), raw)


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


def is_valid_http_url(value):
    text = clean_meta_text(value)
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    if parsed.scheme not in {'http', 'https'}:
        return False
    if not parsed.netloc:
        return False
    # Reject common placeholder pseudo-links like http://文系のみ
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uff00-\uffef]', parsed.netloc):
        return False
    if '.' not in parsed.netloc:
        return False
    return True


def clean_official_school_text(raw):
    text = str(raw or '').strip()
    if not text:
        return ''
    # Remove annotation notes while keeping the canonical school name.
    text = re.sub(r'（[^）]*）', '', text).strip()
    text = re.sub(r'\([^\)]*\)', '', text).strip()
    text = re.sub(r'\[[^\]]*\]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return canonicalize_school_name(text)


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

# Hard blacklist for known non-existent or invalid school-faculty pairs.
# This is a narrow safety valve for historical data noise.
SCHOOL_FACULTY_HARD_BLACKLIST = {
    '専修大学': {'理工学部'},
}


def is_blacklisted_school_faculty(school, faculty):
    school_name = canonicalize_school_name(school)
    blocked = SCHOOL_FACULTY_HARD_BLACKLIST.get(school_name, set())
    if not blocked:
        return False

    raw = str(faculty or '').strip()
    label = display_official_faculty_name(school_name, raw)
    return raw in blocked or label in blocked


def is_short_college_school(school):
    name = canonicalize_school_name(school)
    text = str(name or '').strip()
    if not text:
        return False
    return ('短期大学' in text) or ('大学短期大学部' in text)


EXCLUDED_SCHOOLS = {
    normalize_text('上海大学東京校'),
    normalize_text('北京語言大学東京校'),
    normalize_text('上海大学东京校'),
    normalize_text('北京语言大学东京校'),
}


SCHOOL_TARGET_FACULTY_ALLOWLIST = {
    '早稲田大学': {
        '政治経済学部',
        '法学部',
        '教育学部',
        '商学部',
        '社会科学部',
        '文化構想学部',
        '文学部',
        '基幹理工学部',
        '基幹理工学部学系1',
        '基幹理工学部学系2',
        '基幹理工学部学系4',
        '創造理工学部',
        '創造理工学部建築学科',
        '創造理工学部環境資源工学科',
        '創造理工学部社会環境工学科',
        '創造理工学部経営システム工学科',
        '創造理工学部総合機械工学科',
        '先進理工学部',
        '先進理工学部化学・生命化学科',
        '先進理工学部応用化学科',
        '先進理工学部応用物理学科',
        '先進理工学部物理学科',
        '先進理工学部生命医科学科',
        '先進理工学部電気・情報生命工学科',
        '人間科学部',
        '人間科学部人間情報科学科',
        '人間科学部人間環境科学科',
        '人間科学部健康福祉科学科',
        'スポーツ科学部',
        'ｽﾎﾟｰﾂ科学部',
    }
    ,
    '慶應義塾大学': {
        '医学部',
        '医学部医学科',
        '文学部',
        '経済学部',
        '法学部',
        '商学部',
        '理工学部',
    }
}


# Verified official supplements for cases where official CSV rows are grouped
# and cannot fully represent each faculty in the dropdown.
SCHOOL_TARGET_FACULTY_MANUAL_SUPPLEMENTS = {
    '早稲田大学': [
        '政治経済学部',
        '法学部',
        '教育学部',
        '商学部',
        '社会科学部',
        '文化構想学部',
        '文学部',
    ],
    '慶應義塾大学': [
        '文学部',
        '経済学部',
        '法学部',
        '商学部',
    ],
}


def is_allowed_target_faculty_for_school(school, faculty):
    school_name = canonicalize_school_name(school)
    allowed = SCHOOL_TARGET_FACULTY_ALLOWLIST.get(school_name)
    if not allowed:
        return True

    raw = str(faculty or '').strip()
    label = display_official_faculty_name(school_name, raw)
    return (raw in allowed) or (label in allowed)


def apply_manual_target_faculty_supplements(index):
    updated = {school: list(faculties) for school, faculties in (index or {}).items()}

    for school_name, supplements in SCHOOL_TARGET_FACULTY_MANUAL_SUPPLEMENTS.items():
        bucket = updated.setdefault(school_name, [])
        for faculty_name in supplements:
            if faculty_name not in bucket:
                bucket.append(faculty_name)
        updated[school_name] = sorted(list(dict.fromkeys(bucket)))

    return updated


def is_excluded_school(school):
    text = str(canonicalize_school_name(school) or '').strip()
    if not text:
        return False
    return normalize_text(text) in EXCLUDED_SCHOOLS


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

    school_value = canonicalize_school_name(school)
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

# Official recruitment metadata parsed from guideline CSVs
OFFICIAL_PROGRAM_META = {}
OFFICIAL_SCHOOL_META = {}
TARGET_SCHOOL_FACULTY_INDEX = {}
OFFICIAL_REGION_OPTIONS = []
OFFICIAL_APPLICATION_MONTH_OPTIONS = []

# Phase-2 manual patches from verified official guideline content.
# Applied only when a faculty-level official row is missing.
MANUAL_PROGRAM_META_TOKEN_OVERRIDES = {
    ('武蔵野大学', '文学部'): {
        'eju_subjects': '日语, 文科综合',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '法学部'): {
        'eju_subjects': '日语, 文科综合',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', 'ｳｪﾙﾋﾞｰｲﾝｸﾞ学部'): {
        'eju_subjects': '日语, 文科综合',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（文系口径）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '言語文化学部'): {
        'eju_subjects': '日语, 文科综合（文系口径）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('東洋大学', '健康ｽﾎﾟｰﾂ科学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '健康ｽﾎﾟｰﾂ科'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '福祉社会ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '文学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('長崎県立大学', '地域創造学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長崎県立大学', '情報ｼｽﾃﾑ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長崎県立大学', '看護栄養学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長崎県立大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('順天堂大学', '健康ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('順天堂大学', '国際教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('順天堂大学', 'ｽﾎﾟｰﾂ健康科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('順天堂大学', 'ｽﾎﾟｰﾂ健康科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('高知県立大学', '健康栄養学科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知県立大学', '文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知県立大学', '看護学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知県立大学', '社会福祉学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鹿児島大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鹿児島大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鹿児島大学', '法文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鹿児島大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('三重大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('三重大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('三重大学', '生物資源学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都府立大学', '公共政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都府立大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都府立大学', '生命環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('佐賀大学', '地域ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('佐賀大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('佐賀大学', '芸術地域ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('同志社女子大学', '介護学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('同志社女子大学', '学芸学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('同志社女子大学', '現代社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('名城大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('名城大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('名城大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪工業大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪工業大学', '情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪工業大学', 'ﾛﾎﾞﾃｨｸｽ&ﾃﾞｻﾞｲﾝ工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('山形大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山形大学', '地域教育文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山形大学', '社会文化創造学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山梨大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山梨大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山梨大学', '生命環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島市立大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島市立大学', '情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島市立大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟県立大学', '人間生活学科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟県立大学', '国際地域学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟県立大学', '国際経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東北芸術工科大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東北芸術工科大学', '芸術工学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東北芸術工科大学', 'ﾃﾞｻﾞｲﾝ工学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('滋賀県立大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('滋賀県立大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('滋賀県立大学', '環境科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福岡女子大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福岡女子大学', '人間科学部人間環境科学科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福岡女子大学', '国際文理学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('茨城大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('茨城大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('茨城大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('お茶の水女子大学', '人間文化創成科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('お茶の水女子大学', '文教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名桜大学', '人間健康学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名桜大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('和歌山大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('和歌山大学', 'ｼｽﾃﾑ工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('国際教養大学', '国際教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('国際教養大学', 'ｸﾞﾛｰﾊﾞﾙｺﾐｭﾆｹｰｼｮﾝ実践学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大分大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大分大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大和大学', '政治経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大和大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('宮城大学', '事業構想学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宮城大学', '食産業学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岐阜大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岐阜大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山県立大学', '情報系工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山県立大学', 'ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛媛大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛媛大学', '法文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京理科大学', '理（第一部）学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東京理科大学', '理（第二部）学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('沖縄県立芸術大学', '美術工芸学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('沖縄県立芸術大学', '造形芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('群馬大学', '保健学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('群馬大学', '医学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長崎大学', '情報ﾃﾞｰﾀ科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長崎大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高崎経済大学', '地域政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高崎経済大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鳥取大学', '地域学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鳥取大学', '持続性社会創生科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('九州歯科大学', '歯学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医療系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都工芸繊維大学', '工芸科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('公立諏訪東京理科大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('公立鳥取環境大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('前橋工科大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北見工業大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('国際基督教大学', 'ｱｰﾂｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪教育大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良教育大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山県立大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山陽小野田市立山口東京理科大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('崇城大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帯広畜産大学', '共同獣医学部畜産学科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('徳島大学', '融合学域スマート創成科学類学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知教育大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('旭川医科大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('札幌市立大学', 'ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京女子大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東京学芸大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('津田塾大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('滋賀大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本県立大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('石川県立大学', '生物資源環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神奈川県立保健福祉大学', '保健福祉学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸市外国語大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福岡教育大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('秋田大学', '情報ﾃﾞｰﾀ科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('足利大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('長岡技術科学大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('長野県立大学', 'ｸﾞﾛｰﾊﾞﾙﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('鹿屋体育大学', '体育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東洋大学', '法学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '社会学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '社会福祉学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '経営学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '経済学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '国際学部'): {
        'eju_subjects': '日语（学科要件あり）',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', '国際観光学部'): {
        'eju_subjects': '日语（学科要件あり）',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東洋大学', 'ﾗｲﾌﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语',
        'application_window': '一期: 2025/09/01 - 2025/09/03；二期: 2025/10/24 - 2025/10/28',
        'application_months': [9, 10],
        'faculty_pdf': 'https://www.toyo.ac.jp/nyushi/content/dam/toyowebstyle/admission/admission-data/international-student/japan/international_entry-exam_2027.pdf',
    },
    ('東京大学', 'PEAK'): {
        'eju_subjects': 'PEAK英语学位项目（EJU科目要件以年度要项为准）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '人文社会系'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '公共政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '文科一類'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '文科二類'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '文科三類'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '法学政治学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '総合文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '医学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '学際情報学部'): {
        'eju_subjects': '日语, 数学コース2（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '工学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '情報理工学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '新領域創成科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理工学部理学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理科(類不明)'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理科一類'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理科二類'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理科三類'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '理科（＊）類'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('東京大学', '農学生命科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-12月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12],
    },
    ('京都大学', 'Kyoto iUP'): {
        'eju_subjects': 'Kyoto iUP要件（EJU/JLPT/英语外部考试等は募集要項指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '人間環境学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '公共政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '経営管理学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '総合生存学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', 'ｱｼﾞｱｱﾌﾘｶ地域研究学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('京都大学', 'ｴﾈﾙｷﾞｰ科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東海大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '体育学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '健康学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '児童教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '国際文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '多文化社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '政治経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '教養学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東海大学', '観光学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '産業社会学部現代社会学科人間福祉専攻学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '神学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '経営戦略学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西学院大学', '言語ｺﾐｭﾆｹｰｼｮﾝ文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '21世紀社会ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '人工知能科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '異文化ｺﾐｭﾆｹｰｼｮﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '社会ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', '観光学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', 'ｺﾐｭﾆﾃｨ福祉学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', 'ｽﾎﾟｰﾂｳｴﾙﾈｽ学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立教大学', 'ﾋﾞｼﾞﾈｽﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '人間健康学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '外国語教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '政策創造学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '東ｱｼﾞｱ文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '環境都市工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', 'ｼｽﾃﾑ理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('関西大学', 'ﾋﾞｼﾞﾈｽﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪大学', 'こども学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '医学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '国際公共政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '子ども教育学科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪大学', '連合小児発達学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '医歯薬学総合学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医歯薬系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '歯学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医歯薬系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '環境理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '環境生命自然科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '社会文化科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', 'ｸﾞﾛｰﾊﾞﾙﾃﾞｨｽｶﾊﾞﾘｰﾌﾟﾛｸﾞﾗﾑ学部'): {
        'eju_subjects': '英语学位项目（EJU/英语要件は募集要項指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岡山大学', 'ﾍﾙｽｼｽﾃﾑ統合科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('日本大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '危機管理学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '国際関係学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '新聞学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '生物資源科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本大学', 'ｽﾎﾟｰﾂ科'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('筑波大学', '人文文化学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '人文社会ﾋﾞｼﾞﾈｽ科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '人間学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '人間総合科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '体育専門学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '理工情報生命学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '生命環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '社会国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('筑波大学', '芸術専門学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('青山学院大学', '会計ﾌﾟﾛﾌｪｯｼｮﾝ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '国際政治経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '国際ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '地球社会共生学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '教育人間科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '教育人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('青山学院大学', '総合文化政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '会計専門職学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '先端数理科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '国際日本学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '情報ｺﾐｭﾆｹｰｼｮﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '政治経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '教養ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治大学', 'ｸﾞﾛｰﾊﾞﾙﾋﾞｼﾞﾈｽ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '健康ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
        'school_exam': '书类选考 + 面试（第二次选考）',
        'school_exam_detail': '第二次选考为面试（商学部：线下面试，三田校区口径，日程以当年度募集要项为准）',
    },
    ('慶應義塾大学', '政策ﾒﾃﾞｨｱ学部'): {
        'eju_subjects': '日语（または英语重視型）・学科指定',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
        'school_exam': '书类选考 + 面试（第二次选考）',
        'school_exam_detail': '第二次选考为面试（文学部：线上面试口径，日程以当年度募集要项为准）',
    },
    ('慶應義塾大学', '法務学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
        'school_exam': '书类选考',
        'school_exam_detail': '仅书类选考（无校内考/面试）',
    },
    ('慶應義塾大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '経営管理学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
        'school_exam': '书类选考',
        'school_exam_detail': '仅书类选考（无校内考/面试）',
    },
    ('慶應義塾大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('慶應義塾大学', 'ﾒﾃﾞｨｱﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语（または英语重視型）・学科指定',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '総合社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '臨床心理学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', '言語文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('文教大学', 'ﾘﾊﾋﾞﾘﾃｰｼｮﾝ科学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '国際人間科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '国際文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸大学', 'ｼｽﾃﾑ情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', 'ISP（総合科学ﾌﾟﾛｸﾞﾗﾑ）学部'): {
        'eju_subjects': '英语学位项目（EJU/英语要件は募集要項指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', 'MJSP（現代日本学ﾌﾟﾛｸﾞﾗﾑ）学部'): {
        'eju_subjects': '日语・英语要件あり（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '保健科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（保健系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '公共政策学教育学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '国際広報ﾒﾃﾞｨｱ観光学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '生命科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北海道大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '医学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '国際開発学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '多元数理科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('国士舘大学', '21世紀ｱｼﾞｱ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '体育学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '政治学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '政経学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', '総合知的財産法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('国士舘大学', 'ｸﾞﾛｰﾊﾞﾙｱｼﾞｱ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '国際関係学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', 'ｱｼﾞｱ地域学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', 'ｽﾎﾟｰﾂ健康科'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大東文化大学', 'ｽﾎﾟｰﾂ健康科学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('城西大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
        'guideline_url': 'https://admission.josai.ac.jp/examination/schedule/international-students-52542/',
    },
    ('専修大学', '人間学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '国際ｺﾐｭﾆｹｰｼｮﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('専修大学', 'ﾈｯﾄﾜｰｸ情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '実学社会起業ｲﾉﾍﾞｰｼｮﾝ学位ﾌﾟﾛｸﾞﾗﾑ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '文芸学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '総合文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('近畿大学', '総合社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '仏教学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '国際文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', 'ｸﾞﾛｰﾊﾞﾙﾒﾃﾞｨｱ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('駒沢大学', 'ｸﾞﾛｰﾊﾞﾙﾒﾃﾞｨｱｽﾀ ﾃﾞｨｰｽﾞ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '健康科学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '国際英語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '現代ﾋﾞｼﾞﾈｽ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '発達教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '看護学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（看護系口径）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都橘大学', '総合心理学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '先端情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '国際関係学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '現代社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('京都産業大学', '総合生命科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大阪公立大学', '地域保健学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（保健系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '現代ｼｽﾃﾑ科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '現代ｼｽﾃﾑ科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '生命環境科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大阪公立大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '人間社会科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '先進理工系科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '医系科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '統合生命科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('広島大学', '総合科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('成蹊大学', '国際観光学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '法学政治学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '経済経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', '芸術学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', 'ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('成蹊大学', 'ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('東京都立大学', '人文社会学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '人間健康科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（保健系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '法学政治学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '経済経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', '都市環境科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京都立大学', 'ｼｽﾃﾑﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '社会文化科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '自然科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('熊本大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('神戸学院大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '現代社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', '総合ﾘﾊﾋﾞﾘﾃｰｼｮﾝ学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神戸学院大学', 'ｸﾞﾛｰﾊﾞﾙｺﾐｭﾆｹｰｼｮﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('一橋大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '国際公共政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '経営管理学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', '言語社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('一橋大学', 'ｿｰｼｬﾙﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('中京大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '現代社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中京大学', 'ｽﾎﾟｰﾂ科'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('摂南大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('新潟大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '経済科'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '歯学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医歯系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('新潟大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('昭和女子大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '人間社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '環境ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '生活機構学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', '生活科学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('昭和女子大学', 'ｸﾞﾛｰﾊﾞﾙﾋﾞｼﾞﾈｽ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('横浜国立大学', 'GBEEP学部'): {
        'eju_subjects': '英语学位项目（EJU/英语要件は募集要項指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '先進実践学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '国際社会科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '環境情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜国立大学', '都市ｲﾉﾍﾞｰｼｮﾝ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('甲南大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', '自然科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', 'ﾌﾛﾝﾃｨｱｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('甲南大学', 'ﾏﾈｼﾞﾒﾝﾄ創造学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '人間科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '国際日本学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '歴史民俗資料学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('神奈川大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '政策学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('龍谷大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('北九州市立大学', '国際環境工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', '社会ｼｽﾃﾑ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('北九州市立大学', 'ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('埼玉大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '人文社会芸術総合学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '人間発達科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '芸術文化学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('富山大学', '都市ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('帝京大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('帝京大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含は要項準拠）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '建築学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明星大学', 'ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('横浜市立大学', '国際商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', '国際教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', '国際総合科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', '国際ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', '生命ﾅﾉｼｽﾃﾑ科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', '都市社会文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('横浜市立大学', 'ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('立正大学', '仏教学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('立正大学', 'ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '国際教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '現代社会文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('追手門学院大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '国際関係学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '応用生物学域学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '応用生物学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('中部大学', '経営情報学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '国際教養学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('南山大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('宇都宮大学', '共同教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宇都宮大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宇都宮大学', '地域創生科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宇都宮大学', '地域ﾃﾞｻﾞｲﾝ科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宇都宮大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宇都宮大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '共同獣医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '融合学域スマート創成科学類学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('山口大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知学院大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('愛知学院大学', '心身科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('愛知学院大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('愛知学院大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('愛知学院大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('愛知学院大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('香川大学', '創発科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('香川大学', '創造工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('香川大学', '地域ﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('香川大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('香川大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('香川大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '土佐さきがけﾌﾟﾛｸﾞﾗﾑ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '総合人間自然科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('高知大学', '農林海洋科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉大学', '人文公共学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉大学', '情報ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉大学', '法政経学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('大妻女子大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大妻女子大学', '人間関係学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大妻女子大学', '家政学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大妻女子大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('大妻女子大学', '比較文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('島根大学', '人間社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('島根大学', '材料ｴﾈﾙｷﾞｰ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('島根大学', '法文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('島根大学', '統合理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('島根大学', '自然科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('弘前大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('弘前大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('弘前大学', '地域共創科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('弘前大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('弘前大学', '農学生命科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('日本女子大学', '人間社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本女子大学', '国際文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本女子大学', '家政学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本女子大学', '建築ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('日本女子大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治学院大学', '国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治学院大学', '教育協働学科教育コミュニティ支援専攻心理科学コース'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治学院大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治学院大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('明治学院大学', '社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('杏林大学', '国際協力学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('杏林大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('杏林大学', '総合政策学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('杏林大学', '保健学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('杏林大学', '医学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('県立広島大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('県立広島大学', '地域創生学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('県立広島大学', '生命環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('県立広島大学', '生物資源科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('県立広島大学', '経営情報学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福井県立大学', '恐竜学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福井県立大学', '海洋生物資源学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福井県立大学', '生物資源学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福井県立大学', '看護福祉学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福井県立大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福岡大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('福岡大学', '商学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('福岡大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('福岡大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('福岡大学', 'ｽﾎﾟｰﾂ健康科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('金沢大学', '人間社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('金沢大学', '人間社会環境学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('金沢大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('金沢大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('金沢大学', '自然科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('電気通信大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('電気通信大学', '工学部情報通信工学科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('電気通信大学', '建築ﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('電気通信大学', '情報理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('電気通信大学', '総合情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('静岡大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('静岡大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('静岡大学', '情報学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('静岡大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('静岡大学', 'ｱｼﾞｱﾌﾞﾘｯｼﾞﾌﾟﾛｸﾞﾗﾑ学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('信州大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('信州大学', '医学系'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学系口径）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('信州大学', '経法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('信州大学', '総合人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('千葉工業大学', '情報変革科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('千葉工業大学', '情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('千葉工業大学', '社会ｼｽﾃﾑ科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('千葉工業大学', '社会ｼｽﾃﾑ科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('名古屋市立大学', '人文社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋市立大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋市立大学', '看護学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋市立大学', '経済学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('名古屋市立大学', '芸術工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '人間文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '人間文化総合科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '生活環境学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('奈良女子大学', '理学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('学習院大学', '人文学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('学習院大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('学習院大学', '国際社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('学習院大学', '国際文化交流学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('学習院大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('学習院大学', '政治学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出願時期: 例年9月-11月頃（当年度募集要項で最終確認）',
        'application_months': [9, 10, 11],
    },
    ('宮崎大学', '地域資源創成学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宮崎大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宮崎大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('宮崎大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '人文社会科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '人文社会科'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知県立大学', '外国語学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知県立大学', '情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知県立大学', '日本文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('愛知県立大学', '看護学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京外国語大学', '国際日本学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京外国語大学', '国際社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京外国語大学', '総合国際学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京外国語大学', '言語文化学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京海洋大学', '海洋工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京海洋大学', '海洋生命科'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京海洋大学', '海洋科学技術学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('東京海洋大学', '海洋資源環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('琉球大学', '人文社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('琉球大学', '国際地域創造学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('琉球大学', '地域共創学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('琉球大学', '教育学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福島大学', '人文社会学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福島大学', '地域ﾃﾞｻﾞｲﾝ科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福島大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('福島大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '理工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '総合科学部'): {
        'eju_subjects': '日语, 文科综合（学科指定）',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('岩手大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选',
        'application_window': '出願時期: 例年11月-翌1月頃（当年度募集要項で最終確認）',
        'application_months': [11, 12, 1],
    },
    ('法政大学', '人文科学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '人間環境学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '人間社会学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '公共政策学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '国際文化学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '地域創造ｲﾝｽﾃｨﾃｭｰﾄ学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '政治学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '政策創造学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '文学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '法務学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '法学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '現代福祉学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '社会学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '経営学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '経済学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', '連帯社会ｲﾝｽﾃｨﾃｭｰﾄ学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', 'ｲﾉﾍﾞｰｼｮﾝﾏﾈｼﾞﾒﾝﾄ学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', 'ｷｬﾘｱﾃﾞｻﾞｲﾝ学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', 'ｸﾞﾛｰﾊﾞﾙ教養学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', 'ｽﾎﾟｰﾂ健康学部'): {
        'eju_subjects': '日本留学試験（学部学科指定科目）',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },
    ('法政大学', 'ﾃﾞｻﾞｲﾝ工学部'): {
        'eju_subjects': '日语, 数学コース2',
        'application_window': '一期: 2025/09/09 - 2025/09/19；二期: 2025/11/04 - 2025/11/14',
        'application_months': [9, 11],
    },

    ('武蔵野大学', '経営学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', 'ｱﾝﾄﾚﾌﾟﾚﾅｰｼｯﾌﾟ学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', 'ﾃﾞｰﾀｻｲｴﾝｽ学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（物理/化学/生物）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '環境学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（理工口径）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（理工口径）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('武蔵野大学', 'ｸﾞﾛｰﾊﾞﾙ学部'): {
        'eju_subjects': 'GC/日本语沟通学科: EJU日语（记述除外）或JLPT；Global Business学科: 英语外部考试（TOEFL/IELTS/TOEIC）',
        'application_window': 'Ⅰ期: 2025/08/27 - 2025/09/05；Ⅱ期: 2025/10/14 - 2025/10/23；Ⅲ期: 2026/01/14 - 2026/01/23（邮送必着）',
        'application_months': [8, 9, 10, 1],
    },
    ('九州大学', '法学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '文学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '人文科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '経済学部'): {
        'eju_subjects': '经济・经营学科: 日语, 文科综合, 数学コース1；经济工学科: 日语, 数学コース2, 理科2选',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '歯学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（物理/化学/生物）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '薬学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（化学必含）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '農学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（物理/化学/生物）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '生物資源環境科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（物理/化学/生物）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '環境園芸学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（物理/化学/生物）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', 'ｼｽﾃﾑ情報科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（工学系口径）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', 'ｼｽﾃﾑ生命科学部'): {
        'eju_subjects': '日语, 数学コース2, 理科2选（医学・生命系口径）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '統合新領域学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1 或 日语, 数学コース2, 理科2选（共創学部口径）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('九州大学', '地球社会統合科学部'): {
        'eju_subjects': '日语, 文科综合, 数学コース1 或 日语, 数学コース2, 理科2选（共創学部口径）',
        'application_window': '出愿: 2025/11/04 - 2025/11/14（17:00必着）；Web入力・检定料支付: 2025/10/27 - 2025/11/07',
        'application_months': [10, 11],
    },
    ('中央大学', '法学部'): {
        'eju_subjects': '日语',
        'school_exam': '面试',
        'application_window': '2025/10/27 - 2025/10/31',
        'application_months': [10],
    },
    ('中央大学', '法務学部'): {
        'eju_subjects': '日语',
        'school_exam': '面试',
        'application_window': '2025/10/27 - 2025/10/31',
        'application_months': [10],
    },
    ('中央大学', '経済学部'): {
        'eju_subjects': 'A方式: 日语；B方式: 日语, 文科综合, 数学コース1/2',
        'school_exam': 'A方式: 小论文+面试；B方式: 书类选考',
        'application_window': '2025/11/13 - 2025/11/19（A/B共通）',
        'application_months': [11],
    },
    ('中央大学', '商学部'): {
        'eju_subjects': 'A方式: 日语；B方式: 日语, 文科综合, 数学コース1/2',
        'school_exam': 'A方式: 小论文+面试；B方式: 书类选考',
        'application_window': 'A方式: 2025/10/27 - 2025/10/31；B方式: 2025/11/13 - 2025/11/19',
        'application_months': [10, 11],
    },
    ('中央大学', '文学部'): {
        'eju_subjects': '日语',
        'school_exam': '面试',
        'application_window': '2025/11/13 - 2025/11/19',
        'application_months': [11],
    },
    ('中央大学', '総合政策学部'): {
        'eju_subjects': '日语',
        'school_exam': '小论文+面试',
        'application_window': '2025/11/13 - 2025/11/19',
        'application_months': [11],
    },
    ('中央大学', '国際経営学部'): {
        'eju_subjects': '日语（A方式）',
        'school_exam': '英语笔试+面试（A方式）',
        'application_window': '2025/10/27 - 2025/10/31（A方式）',
        'application_months': [10],
    },
    ('上智大学', '学部'): {
        'eju_subjects': 'EJU必須（2025年6月/2024年6月・11月実施が有効；学部学科別に指定科目・基準点あり）',
        'school_exam': '学科試問・面接（学部学科により実施）',
        'application_window': 'Web出願: 2025/07/23 10:00 - 2025/08/05 23:59；出願書類提出: 2025/08/06（消印有効）',
        'application_months': [7, 8],
    },
    ('明治学院大学', '経済'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'school_exam': '面试',
        'application_window': '一期: 2025/09/19 - 2025/09/26；二期: 2025/10/23 - 2025/10/29',
        'application_months': [9, 10],
    },
    ('明治学院大学', '経営'): {
        'eju_subjects': '日语, 文科综合, 数学コース1',
        'school_exam': '面试',
        'application_window': '一期: 2025/09/19 - 2025/09/26；二期: 2025/10/23 - 2025/10/29',
        'application_months': [9, 10],
    },
    ('九州工業大学', '工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科（物理・化学）',
        'school_exam': '个别学力检査（数学・理科）+ 面试',
        'application_window': '网络出愿登记: 2025/12/22 09:00 起；出愿材料提交截止: 2026/01/09 17:00（必着）',
        'application_months': [12, 1],
    },
    ('九州工業大学', '情報工学部'): {
        'eju_subjects': '日语, 数学コース2, 理科（物理・化学・生物から2科目）',
        'school_exam': '个别学力检査（数学・理科）+ 面试',
        'application_window': '网络出愿登记: 2025/12/22 09:00 起；出愿材料提交截止: 2026/01/09 17:00（必着）',
        'application_months': [12, 1],
    },
}

# Populate this fixed set with the schools that should display the
# "高度人才加分校" note in recommendation cards.
HIGH_TALENT_BONUS_SCHOOLS = {
    '愛知医科大学',
    '秋田大学',
    '青山学院大学',
    '千葉工業大学',
    '千葉大学',
    '中部大学',
    '中央大学',
    '獨協医科大学',
    '同志社大学',
    '愛媛大学',
    '藤田医科大学',
    '岐阜大学',
    '群馬大学',
    '浜松医科大学',
    '弘前大学',
    '広島大学',
    '一橋大学',
    '北海道大学',
    '法政大学',
    '兵庫医科大学',
    '茨城大学',
    '国際基督教大学',
    '東京慈恵会医科大学',
    '順天堂大学',
    '香川大学',
    '鹿児島大学',
    '神奈川大学',
    '金沢大学',
    '関西医科大学',
    '関西大学',
    '慶應義塾大学',
    '近畿大学',
    '北里大学',
    '神戸大学',
    '高知大学',
    '高知工科大学',
    '工学院大学',
    '熊本大学',
    '久留米大学',
    '関西学院大学',
    '京都工芸繊維大学',
    '京都府立医科大学',
    '京都大学',
    '九州工業大学',
    '九州大学',
    '明治大学',
    '名城大学',
    '三重大学',
    '室蘭工業大学',
    '長岡技術科学大学',
    '長崎大学',
    '名古屋市立大学',
    '名古屋工業大学',
    '名古屋大学',
    '日本大学',
    '新潟大学',
    '日本医科大学',
    'お茶の水女子大学',
    '大分大学',
    '岡山大学',
    '岡山理科大学',
    '沖縄科学技術大学院大学',
    '大阪工業大学',
    '大阪医科薬科大学',
    '大阪大学',
    '立教大学',
    '立命館アジア太平洋大学',
    '立命館大学',
    '龍谷大学',
    '佐賀大学',
    '埼玉医科大学',
    '埼玉大学',
    '札幌医科大学',
    '芝浦工業大学',
    '滋賀医科大学',
    '島根大学',
    '信州大学',
    '静岡大学',
    '創価大学',
    '上智大学',
    '帝京大学',
    '総合研究大学院大学',
    '電気通信大学',
    '東京大学',
    '東邦大学',
    '東北大学',
    '東海大学',
    '徳島大学',
    '東京都市大学',
    '東京電機大学',
    '東京医科歯科大学',
    '東京医科大学',
    '東京科学大学',
    '東京都立大学',
    '東京農業大学',
    '東京農工大学',
    '東京海洋大学',
    '東京理科大学',
    '東京工科大学',
    '鳥取大学',
    '富山県立大学',
    '東洋大学',
    '豊橋技術科学大学',
    '会津大学',
    '福井大学',
    '兵庫県立大学',
    '宮崎大学',
    '産業医科大学',
    '静岡県立大学',
    '琉球大学',
    '富山大学',
    '筑波大学',
    '山梨大学',
    '宇都宮大学',
    '和歌山県立医科大学',
    '早稲田大学',
    '山形大学',
    '山口大学',
    '横浜市立大学',
    '横浜国立大学',
}

# Global dataframe reference
EJU_DF = None

# Manual/dynamic school info maps to reduce default-hensachi fallbacks.
SCHOOL_INFO_OVERRIDES = {}
SCHOOL_INFO_DYNAMIC = {}

CAPITAL_REGION_KEYWORDS = [
    '東京', '神奈川', '横浜', '川崎', '千葉', '埼玉', '首都', '都立'
]
HOKKAIDO_REGION_KEYWORDS = ['北海道', '札幌', '函館', '旭川']
TOHOKU_REGION_KEYWORDS = ['東北', '青森', '岩手', '宮城', '仙台', '秋田', '山形', '福島']
KANTO_REGION_KEYWORDS = [
    '茨城', '栃木', '群馬', '前橋', '宇都宮', '高崎', 'つくば', '北関東'
]
CHUBU_REGION_KEYWORDS = [
    '中部', '東海', '北陸', '甲信越', '新潟', '富山', '石川', '福井',
    '山梨', '長野', '岐阜', '静岡', '愛知', '名古屋'
]
KINKI_REGION_KEYWORDS = [
    '大阪', '京都', '神戸', '兵庫', '奈良', '滋賀', '和歌山', '三重', '近畿', '関西'
]
CHUGOKU_REGION_KEYWORDS = ['中国', '鳥取', '島根', '岡山', '広島', '山口']
SHIKOKU_REGION_KEYWORDS = ['四国', '徳島', '香川', '愛媛', '高知']
KYUSHU_REGION_KEYWORDS = ['九州', '福岡', '佐賀', '長崎', '熊本', '大分', '宮崎', '鹿児島', '沖縄']

REGION_VALUE_ALIASES = {
    'all': 'all',
    'capital': 'capital',
    'hokkaido': 'hokkaido',
    'tohoku': 'tohoku',
    'kanto': 'kanto',
    'chubu': 'chubu',
    'kinki': 'kinki',
    'chugoku': 'chugoku',
    'shikoku': 'shikoku',
    'kyushu': 'kyushu',
    # backward compatibility for old UI values
    'capital_hokkaido': 'capital_hokkaido',
    'kansai': 'kinki',
}

CAPITAL_REGION_SCHOOL_HINTS = {
    '早稲田大学', '慶應義塾大学', '上智大学', '学習院大学', '東京理科大学',
    '明治大学', '青山学院大学', '立教大学', '中央大学', '法政大学',
    '日本大学', '東洋大学', '専修大学', '駒澤大学', '明治学院大学',
    '國學院大學', '武蔵大学', '武蔵野大学', '芝浦工業大学', '工学院大学',
    '東京電機大学', '東京都市大学', '成蹊大学', '成城大学', '亜細亜大学'
}
KINKI_REGION_SCHOOL_HINTS = {
    '関西大学', '関西学院大学', '同志社大学', '立命館大学', '龍谷大学',
    '京都産業大学', '近畿大学', '甲南大学', '京都橘大学', '佛教大学',
    '大阪経済大学', '大阪工業大学', '京都女子大学', '神戸学院大学'
}

MAJOR_DIRECTION_KEYWORDS = {
    'bunka_literature': ['文学', '人文', '日本語', '史学', '哲学', '言語', '文化'],
    'bunka_politics': ['政治', '法学', '政策', '公共', '国際関係', '行政'],
    'bunka_business': ['経営', '商学', '会計', 'ビジネス', 'マネジメント'],
    'bunka_economics': ['経済', '金融'],
    'bunka_sociology': ['社会', '社会学', '社会福祉', '福祉', 'メディア', 'コミュニティ', '人間科学'],
    'bunka_liberal_arts': ['教養', '国際教養', 'リベラルアーツ', '総合文化', '文化構想'],
    'bunka_education': ['教育', '教員', '学校教育', '教育学'],
    'bunka_language': ['外国語', '外語', '英語', '中国語', 'フランス語', 'ドイツ語', 'スペイン語', '日本語教育', '言語教育', '言語', '言語文化', '国際日本', '日本学', '通訳', '翻訳'],
    'rika_mechanical': ['機械', '機械工', 'メカトロ', 'ロボティクス'],
    'rika_electrical': ['電気', '電子', '通信', '制御', '電機'],
    'rika_math': [
        '数学', '数理', '数物', '計数', '統計', '応用数学', '数理情報', '情報数理',
        '理学部', '理工学部', '情報理工学部', '総合数理', '数理科学', '数理工学',
        'データサイエンス', 'ﾃﾞｰﾀｻｲｴﾝｽ', '情報ﾃﾞｰﾀｻｲｴﾝｽ', '経営工学', '管理工学'
    ],
    'rika_architecture': ['建築', '都市工', '環境工学', '社会環境'],
    'rika_info': ['情報', '情報理工', '情報工学', '計算機', 'コンピュータ', 'AI', '知能'],
    'rika_medicine': ['医学', '医', '保健', '看護', '医療'],
    'rika_pharmacy': ['薬学', '薬'],
    'rika_dentistry': ['歯学', '歯'],
    'rika_agri_bio': ['農学', '農', '生物', '生命', 'バイオ', '応用生物', '生命科学']
}

# Configurable hard excludes for track-faculty compatibility guard.
# Update these lists when business rules for bunka/rika faculty boundaries change.
TRACK_FACULTY_HARD_EXCLUDES = {
    'bunka': [
        '理工', '工学', '理学', '情報理工', '生命理工', '創造理工', '基幹理工', '先進理工',
        '農学', '農林', '獣医', '医学', '医療', '看護', '薬学', '歯学', '保健', '物質',
        '機械', '電気', '電子', '応用化学', '化学工', '建築', '土木', '材料', 'データサイエンス'
    ],
    'rika': [
        '文学', '人文', '法学', '経済', '経営', '商学', '教育', '外国語', '国際', '国際教養',
        '社会', '社会学', '政策', '観光', '心理', '福祉', '芸術', '文化', '神学'
    ]
}

# Allow technical faculties that include ambiguous keywords (e.g. 社会) but are still rika-compatible.
TRACK_FACULTY_ALLOW_OVERRIDES = {
    'rika': [
        '社会工学', '社会基盤', '社会環境'
    ]
}

RIKA_MATH_EXCLUDE_KEYWORDS = [
    '危機管理', '文理学部', '国際文理', '総合心理',
    '社会情報', '経営情報', '流通情報', '芸術情報', 'ﾋﾞｼﾞﾈｽ情報', 'ﾒﾃﾞｨｱ情報'
]

# Discipline-first and faculty-fallback matching for rika directions.
# If subject-level keywords are not found, we fall back to related faculty names.
RIKA_DIRECTION_FACULTY_FALLBACK = {
    'rika_mechanical': [
        '工学部', '理工学部', '先進理工学部', '創造理工学部', '基幹理工学部',
        '創域理工学部', '統合理工学部', '総合理工学部', 'ｼｽﾃﾑ理工学部', '産業理工学部'
    ],
    'rika_electrical': [
        '工学部', '理工学部', '情報理工学部', '先進理工学部', '創造理工学部',
        '基幹理工学部', '創域理工学部', '統合理工学部', '総合理工学部', 'ｼｽﾃﾑ理工学部'
    ],
    'rika_math': [
        '理学部', '理工学部', '情報理工学部', '総合数理学部'
    ]
}

def get_school_type(school_name):
    school_name = canonicalize_school_name(school_name)
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


def infer_region_bucket_from_text(value):
    text = str(value or '').strip()
    if not text:
        return 'other'

    if any(k in text for k in CAPITAL_REGION_KEYWORDS):
        return 'capital'
    if any(k in text for k in HOKKAIDO_REGION_KEYWORDS):
        return 'hokkaido'
    if any(k in text for k in TOHOKU_REGION_KEYWORDS):
        return 'tohoku'
    if any(k in text for k in KANTO_REGION_KEYWORDS):
        return 'kanto'
    if any(k in text for k in CHUBU_REGION_KEYWORDS):
        return 'chubu'
    if any(k in text for k in KINKI_REGION_KEYWORDS):
        return 'kinki'
    if any(k in text for k in CHUGOKU_REGION_KEYWORDS):
        return 'chugoku'
    if any(k in text for k in SHIKOKU_REGION_KEYWORDS):
        return 'shikoku'
    if any(k in text for k in KYUSHU_REGION_KEYWORDS):
        return 'kyushu'
    return 'other'


def get_school_region_bucket(school_name):
    school_name = canonicalize_school_name(school_name)
    official_region = get_official_program_meta(school_name).get('region')
    official_bucket = infer_region_bucket_from_text(official_region)
    if official_bucket != 'other':
        return official_bucket

    name = str(school_name or '').strip()
    if not name:
        return 'other'

    if name in CAPITAL_REGION_SCHOOL_HINTS:
        return 'capital'
    if name in KINKI_REGION_SCHOOL_HINTS:
        return 'kinki'

    return infer_region_bucket_from_text(name)


def passes_region_filter(school_name, region_filter):
    if isinstance(region_filter, list):
        regions = [str(item).strip() for item in region_filter if str(item).strip() and str(item).strip() != 'all']
        if not regions:
            return True
        return any(passes_region_filter(school_name, item) for item in regions)

    region = str(region_filter or 'all').strip()
    region = REGION_VALUE_ALIASES.get(region, region)
    if region in {'', 'all'}:
        return True

    bucket = get_school_region_bucket(school_name)

    if region == 'capital_hokkaido':
        return bucket in {'capital', 'hokkaido'}
    if region in {'capital', 'hokkaido', 'tohoku', 'kanto', 'chubu', 'kinki', 'chugoku', 'shikoku', 'kyushu'}:
        return bucket == region

    # Fail closed for unknown values to avoid leaking unfiltered results.
    return False


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

    matched = any(k in text for k in keywords)
    if not matched:
        fallback_keywords = RIKA_DIRECTION_FACULTY_FALLBACK.get(direction, [])
        if fallback_keywords and any(k in text for k in fallback_keywords):
            matched = True

    if not matched:
        return False

    if direction == 'rika_math' and any(k in text for k in RIKA_MATH_EXCLUDE_KEYWORDS):
        return False
    return True


def is_track_faculty_compatible(track, faculty_name):
    text = str(faculty_name or '').strip()
    if not text:
        return True

    allow_overrides = TRACK_FACULTY_ALLOW_OVERRIDES.get(str(track or '').strip(), [])
    if allow_overrides and any(keyword in text for keyword in allow_overrides):
        return True

    excludes = TRACK_FACULTY_HARD_EXCLUDES.get(str(track or '').strip(), [])
    if excludes:
        return not any(keyword in text for keyword in excludes)
    return True


def rebalance_tiers_by_hensachi(safety, match, challenger):
    """Apply a presentation-layer correction so displayed difficulty ordering follows common sense.

    Goal: challenger hensachi >= match hensachi >= safety hensachi as much as possible,
    while preserving the original probability model as the primary signal.
    """

    def sort_bucket(bucket):
        bucket.sort(key=lambda x: (x.get('hensachi', 0), x.get('median_score', 0), x.get('probability', 0)), reverse=True)

    def pick_overlap_index(src_bucket, dst_bucket, max_promotable_prob):
        if not src_bucket or not dst_bucket:
            return None
        dst_min_h = min(x.get('hensachi', 0) for x in dst_bucket)
        candidates = [
            (i, item) for i, item in enumerate(src_bucket)
            if item.get('hensachi', 0) > dst_min_h and item.get('probability', 0) <= max_promotable_prob
        ]
        if not candidates:
            return None
        # Move the most difficult but still borderline-probability item first.
        candidates.sort(key=lambda x: (x[1].get('hensachi', 0), -x[1].get('probability', 0)), reverse=True)
        return candidates[0][0]

    sort_bucket(safety)
    sort_bucket(match)
    sort_bucket(challenger)

    # Promote high-hensachi safety entries into match when boundaries overlap.
    guard = 0
    while safety and match and max(x.get('hensachi', 0) for x in safety) > min(x.get('hensachi', 0) for x in match):
        guard += 1
        if guard > 500:
            break
        # Probability first: only rebalance borderline safety entries.
        idx = pick_overlap_index(safety, match, 0.71)
        if idx is None:
            break
        item = safety.pop(idx)
        item['tier'] = '适中'
        item['tier_sort'] = 1
        match.append(item)
        sort_bucket(safety)
        sort_bucket(match)

    # Promote high-hensachi match entries into challenger when boundaries overlap.
    guard = 0
    while match and challenger and max(x.get('hensachi', 0) for x in match) > min(x.get('hensachi', 0) for x in challenger):
        guard += 1
        if guard > 500:
            break
        # Never push high-probability match entries into challenger just for hensachi ordering.
        idx = pick_overlap_index(match, challenger, 0.41)
        if idx is None:
            break
        item = match.pop(idx)
        item['tier'] = '冲刺'
        item['tier_sort'] = 0
        challenger.append(item)
        sort_bucket(match)
        sort_bucket(challenger)

    # Final internal sort for stable rendering.
    sort_bucket(safety)
    sort_bucket(match)
    sort_bucket(challenger)
    return safety, match, challenger


def enforce_medical_hierarchy_buckets(safety, match, challenger):
    """Final bucket constraint for medical majors.

    - 医学部 must be challenger
    - 獣医 must be at least match (not safety)
    - 看護/保健/医療 branches must not be challenger
    """
    new_safety = []
    new_match = []
    new_challenger = []

    for item in safety:
        level = medical_faculty_difficulty_level(item.get('faculty', ''))
        if level == 3:
            item['tier'] = '冲刺'
            item['tier_sort'] = 0
            new_challenger.append(item)
        elif level == 2:
            item['tier'] = '适中'
            item['tier_sort'] = 1
            new_match.append(item)
        else:
            new_safety.append(item)

    for item in match:
        level = medical_faculty_difficulty_level(item.get('faculty', ''))
        if level == 3:
            item['tier'] = '冲刺'
            item['tier_sort'] = 0
            new_challenger.append(item)
        else:
            new_match.append(item)

    for item in challenger:
        level = medical_faculty_difficulty_level(item.get('faculty', ''))
        if level == 1:
            item['tier'] = '适中'
            item['tier_sort'] = 1
            new_match.append(item)
        else:
            new_challenger.append(item)

    new_safety.sort(key=lambda x: (x.get('hensachi', 0), x.get('median_score', 0), x.get('probability', 0)), reverse=True)
    new_match.sort(key=lambda x: (x.get('hensachi', 0), x.get('median_score', 0), x.get('probability', 0)), reverse=True)
    new_challenger.sort(key=lambda x: (x.get('hensachi', 0), x.get('median_score', 0), x.get('probability', 0)), reverse=True)
    return new_safety, new_match, new_challenger


def enforce_probability_bucket_boundaries(safety, match, challenger):
    """Apply strict probability-to-bucket boundaries for non-medical majors.

    - >= 0.70 => 保底
    - >= 0.40 and < 0.70 => 适中
    - >= 0.15 and < 0.40 => 冲刺

    Medical hierarchy (医学/獣医等) keeps priority and is not overridden here.
    """
    merged = list(safety) + list(match) + list(challenger)
    new_safety = []
    new_match = []
    new_challenger = []

    for item in merged:
        faculty = item.get('faculty', '')
        med_level = medical_faculty_difficulty_level(faculty)
        prob = float(item.get('probability', 0) or 0)

        if med_level >= 2:
            if item.get('tier') == '冲刺':
                item['tier_sort'] = 0
                new_challenger.append(item)
            else:
                item['tier'] = '适中'
                item['tier_sort'] = 1
                new_match.append(item)
            continue

        if prob >= 0.70:
            item['tier'] = '保底'
            item['tier_sort'] = 2
            new_safety.append(item)
        elif prob >= 0.40:
            item['tier'] = '适中'
            item['tier_sort'] = 1
            new_match.append(item)
        else:
            item['tier'] = '冲刺'
            item['tier_sort'] = 0
            new_challenger.append(item)

    new_safety.sort(key=lambda x: (x.get('probability', 0), x.get('hensachi', 0), x.get('median_score', 0)), reverse=True)
    new_match.sort(key=lambda x: (x.get('probability', 0), x.get('hensachi', 0), x.get('median_score', 0)), reverse=True)
    new_challenger.sort(key=lambda x: (x.get('probability', 0), x.get('hensachi', 0), x.get('median_score', 0)), reverse=True)
    return new_safety, new_match, new_challenger


def classify_display_tier(probability, faculty_name=''):
    """Classify one item using the same post-processing path as recommendation list."""
    prob = float(probability or 0)
    if prob >= 0.70:
        tier = '保底'
        tier_sort = 2
    elif prob >= 0.40:
        tier = '适中'
        tier_sort = 1
    else:
        tier = '冲刺'
        tier_sort = 0

    item = {
        'tier': tier,
        'tier_sort': tier_sort,
        'faculty': str(faculty_name or ''),
        'probability': prob,
        'hensachi': 0,
        'median_score': 0,
    }
    safety = [item] if tier == '保底' else []
    match = [item] if tier == '适中' else []
    challenger = [item] if tier == '冲刺' else []

    safety, match, challenger = enforce_medical_hierarchy_buckets(safety, match, challenger)
    safety, match, challenger = enforce_probability_bucket_boundaries(safety, match, challenger)

    if safety:
        return '保底', 'green'
    if match:
        return '适中', 'yellow'
    return '冲刺', 'red'


def build_probability_stats_for_prediction(school, faculty, track, detail=None):
    """Build probability stats on the same basis as recommendation list.

    Priority:
    1) STATS_CACHE raw stats for resolved school/faculty
    2) detail passed-score fallback
    """
    prefer_detail_stats = bool(detail and detail.get('prefer_detail_stats'))

    cache_stats = None
    if not prefer_detail_stats:
        cache_stats = STATS_CACHE.get((school, faculty, track))
        if cache_stats is None:
            resolved_school, resolved_faculty = resolve_school_faculty_names(school, faculty)
            cache_stats = STATS_CACHE.get((resolved_school, resolved_faculty, track))

    if cache_stats is not None:
        return {
            'median_score': cache_stats.get('raw_median_score', cache_stats.get('median_score')),
            'std_score': cache_stats.get('raw_std_score', cache_stats.get('std_score')),
            'avg_ibt': cache_stats.get('avg_ibt'),
            'avg_toeic': cache_stats.get('avg_toeic'),
            'target_range_q3': cache_stats.get('target_range_q3'),
            'total_records': cache_stats.get('total_records'),
        }

    if detail and len(detail.get('scores_passed', [])) > 0:
        passed_scores = np.array(detail.get('scores_passed', []), dtype=float)
        std_score = float(np.std(passed_scores)) if len(passed_scores) > 1 else 20.0
        if std_score == 0:
            std_score = 20.0
        passed_count = int(detail['stats'].get('passed_count', 0) or 0)
        failed_count = int(detail['stats'].get('failed_count', 0) or 0)
        return {
            'median_score': float(np.median(passed_scores)),
            'std_score': std_score,
            'avg_ibt': detail['stats'].get('avg_ibt'),
            'avg_toeic': detail['stats'].get('avg_toeic'),
            'target_range_q3': detail['stats'].get('target_range_q3'),
            'total_records': passed_count + failed_count,
        }

    return None


def medical_faculty_difficulty_level(faculty_name):
    """Return medical difficulty level.

    3: 医学部 (hardest)
    2: 獣医 (close to medicine)
    1: 看護/保健/医療 branch
    0: non-medical
    """
    text = str(faculty_name or '').strip()
    if not text:
        return 0

    vet_keywords = ['獣医', '獣医学', '共同獣医学']
    branch_keywords = [
        '看護', '保健', '医療', 'リハビリ', '臨床', '検査', '助産',
        '放射線', '福祉', '栄養', '医療科学'
    ]

    is_vet = any(k in text for k in vet_keywords)
    is_branch = any(k in text for k in branch_keywords)
    is_med = '医学部' in text and not is_vet and not is_branch

    if is_med:
        return 3
    if is_vet:
        return 2
    if is_branch:
        return 1
    if '医' in text and not any(k in text for k in ['薬', '歯']):
        return 1
    return 0


def clean_meta_text(value):
    text = str(value or '').strip()
    return text if text and text.lower() != 'nan' else ''


def merge_meta_dict(base, update):
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if value in (None, '', []):
            continue
        if isinstance(value, bool):
            merged[key] = value
        elif isinstance(value, list):
            existing = merged.get(key, [])
            if not isinstance(existing, list):
                existing = [existing] if existing not in (None, '') else []
            merged[key] = sorted(list({*existing, *value}))
        elif key not in merged or merged[key] in (None, '', []):
            merged[key] = value
    return merged


def overlay_meta_dict(base, update):
    """Overlay non-empty update fields on top of base.

    Used when applying faculty-level metadata so it can override school-level defaults.
    """
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if value in (None, '', []):
            continue
        merged[key] = value
    return merged


def format_application_windows(row):
    periods = []
    for start_col, end_col, label in [
        ('出愿开始日期', '出愿结束日期', '一期'),
        ('2期出愿开始日期', '2期出愿结束日期', '二期'),
    ]:
        start = clean_meta_text(row.get(start_col, ''))
        end = clean_meta_text(row.get(end_col, ''))
        if start and end:
            periods.append(f'{label}: {start} - {end}')
        elif start:
            periods.append(f'{label}: {start} 起')
        elif end:
            periods.append(f'{label}: 截止 {end}')
    return '；'.join(periods)


def extract_month_from_date_text(value):
    text = clean_meta_text(value)
    if not text:
        return None
    match = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', text)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return month
    match = re.search(r'(\d{1,2})月', text)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            return month
    return None


def extract_application_months(row):
    months = set()
    for col in ['出愿开始日期', '2期出愿开始日期']:
        month = extract_month_from_date_text(row.get(col, ''))
        if month:
            months.add(month)
    return sorted(months)


def build_official_row_meta(row, region_col=None, intl_col=None, link_col=None, pdf_col=None,
                            eju_col=None, english_required_col=None, english_type_col=None,
                            jlpt_col=None, school_exam_col=None, school_exam_detail_col=None):
    english_required = normalize_official_bool(row.get(english_required_col, '')) if english_required_col else None
    jlpt_required = normalize_official_bool(row.get(jlpt_col, '')) if jlpt_col else None
    intl_accepts = normalize_official_bool(row.get(intl_col, '')) if intl_col else None

    row_meta = {
        'region': clean_meta_text(row.get(region_col, '')) if region_col else '',
        'intl_accepts': intl_accepts,
        'guideline_url': clean_meta_text(row.get(link_col, '')) if (link_col and is_valid_http_url(row.get(link_col, ''))) else '',
        'faculty_pdf': clean_meta_text(row.get(pdf_col, '')) if (pdf_col and is_valid_http_url(row.get(pdf_col, ''))) else '',
        'eju_subjects': clean_meta_text(row.get(eju_col, '')) if eju_col else '',
        'english_required': english_required,
        'english_type': clean_meta_text(row.get(english_type_col, '')) if english_type_col else '',
        'jlpt_required': jlpt_required,
        'school_exam': clean_meta_text(row.get(school_exam_col, '')) if school_exam_col else '',
        'school_exam_detail': clean_meta_text(row.get(school_exam_detail_col, '')) if school_exam_detail_col else '',
        'application_window': format_application_windows(row),
        'application_months': extract_application_months(row),
    }
    return {key: value for key, value in row_meta.items() if value not in (None, '', [])}


def load_official_program_meta():
    school_meta = {}
    program_meta = {}
    eligible_index = {}
    region_values = set()
    application_month_values = set()

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
        region_col = next((c for c in official.columns if '地区归属' in str(c)), None)
        link_col = next((c for c in official.columns if '募集要项链接' in str(c)), None)
        pdf_col = next((c for c in official.columns if '具体学部PDF' in str(c)), None)
        eju_col = next((c for c in official.columns if 'EJU必要科目' in str(c)), None)
        english_required_col = next((c for c in official.columns if '是否需要英语成绩' in str(c)), None)
        english_type_col = next((c for c in official.columns if '英语成绩类型' in str(c)), None)
        jlpt_col = next((c for c in official.columns if '是否必须JLPT成绩' in str(c)), None)
        school_exam_col = next((c for c in official.columns if '是否有校内考' in str(c)), None)
        school_exam_detail_col = next((c for c in official.columns if '第二次选考/校内考/面试 内容' in str(c)), None)

        current_school = ''

        for _, row in official.iterrows():
            raw_name = clean_official_school_text(row.get(name_col, ''))
            if not raw_name:
                continue

            owner_school = clean_official_school_text(row.get(owner_col, '')) if owner_col else ''
            row_meta = build_official_row_meta(
                row,
                region_col=region_col,
                intl_col=intl_col,
                link_col=link_col,
                pdf_col=pdf_col,
                eju_col=eju_col,
                english_required_col=english_required_col,
                english_type_col=english_type_col,
                jlpt_col=jlpt_col,
                school_exam_col=school_exam_col,
                school_exam_detail_col=school_exam_detail_col,
            )

            if row_meta.get('region'):
                region_values.add(row_meta['region'])
            if row_meta.get('application_months'):
                application_month_values.update(row_meta['application_months'])

            if owner_school and is_school_name(owner_school):
                current_school = owner_school
                school_meta[owner_school] = merge_meta_dict(school_meta.get(owner_school, {}), row_meta)

            if is_school_name(raw_name):
                current_school = raw_name
                school_meta[raw_name] = merge_meta_dict(school_meta.get(raw_name, {}), row_meta)
                continue

            school, faculty = split_school_and_faculty(raw_name)
            if school and is_school_name(school):
                current_school = school
                school_meta[school] = merge_meta_dict(school_meta.get(school, {}), row_meta)

            school_name = owner_school if (owner_school and is_school_name(owner_school)) else (school or current_school)
            faculty_name = faculty if faculty else raw_name
            faculty_name = clean_official_faculty_text(faculty_name, school_name)

            if not school_name or not faculty_name or not looks_like_official_faculty(faculty_name):
                continue

            program_key = (school_name, faculty_name)
            program_meta[program_key] = merge_meta_dict(program_meta.get(program_key, {}), row_meta)

            if row_meta.get('intl_accepts') is False:
                continue

            eligible_index.setdefault(school_name, set()).add(faculty_name)

    eligible_index = {
        school: sorted(list(faculties))
        for school, faculties in eligible_index.items()
        if faculties
    }
    return school_meta, program_meta, eligible_index, sorted(region_values), sorted(application_month_values)


def get_official_program_meta(school, faculty=''):
    school_name = canonicalize_school_name(school)
    faculty_name = str(faculty or '').strip()
    school_level_meta = dict(OFFICIAL_SCHOOL_META.get(school_name, {}))
    merged = dict(school_level_meta)
    has_faculty_match = False

    faculty_sensitive_fields = {
        'eju_subjects',
        'english_required',
        'english_type',
        'jlpt_required',
        'school_exam',
        'school_exam_detail',
        'application_window',
        'application_months',
    }

    if not faculty_name:
        return merged

    direct_keys = [
        (school_name, faculty_name),
        (school_name, display_official_faculty_name(school_name, faculty_name)),
    ]
    for key in direct_keys:
        if key in OFFICIAL_PROGRAM_META:
            merged = overlay_meta_dict(merged, OFFICIAL_PROGRAM_META[key])
            has_faculty_match = True
            break

    faculty_norm = normalize_text(display_official_faculty_name(school_name, faculty_name))
    if not has_faculty_match:
        for (meta_school, meta_faculty), meta in OFFICIAL_PROGRAM_META.items():
            if meta_school != school_name:
                continue
            meta_norm = normalize_text(meta_faculty)
            if faculty_norm == meta_norm or faculty_norm in meta_norm or meta_norm in faculty_norm:
                merged = overlay_meta_dict(merged, meta)
                has_faculty_match = True
                break

    # Manual token-based override for verified faculties where CSV has no explicit row yet.
    # Prefer the longest matching token so specific faculties (e.g., 情報工学部)
    # are not shadowed by broader tokens (e.g., 工学部).
    matched_override = None
    matched_token_len = -1
    for (ov_school, token), ov_meta in MANUAL_PROGRAM_META_TOKEN_OVERRIDES.items():
        if ov_school != school_name:
            continue
        if token and token in faculty_name and len(token) > matched_token_len:
            matched_override = ov_meta
            matched_token_len = len(token)

    if matched_override:
        # If no faculty row matched, drop school-level faculty-sensitive fields first
        # to avoid leaking unrelated interview/exam settings.
        overridden = dict(merged)
        if not has_faculty_match:
            for key in faculty_sensitive_fields:
                overridden.pop(key, None)
        for key, value in matched_override.items():
            if value not in (None, '', []):
                overridden[key] = value
        return overridden

    if has_faculty_match:
        return merged

    # No matched faculty-level row: avoid showing potentially wrong faculty-specific fields
    # copied from a different faculty at school level.
    for key in faculty_sensitive_fields:
        merged.pop(key, None)
    return merged


def passes_application_month_filter(school_name, faculty_name, application_month_filter):
    month_filter = str(application_month_filter or 'all').strip()
    if month_filter in {'', 'all'}:
        return True

    meta = get_official_program_meta(school_name, faculty_name)
    months = meta.get('application_months', [])
    if not months:
        return False

    try:
        target_month = int(month_filter)
    except Exception:
        return False

    return target_month in months

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

    def source_rank(value):
        text = str(value or '').strip().lower()
        if 'toshin' in text:
            return 3
        if 'override' in text:
            return 2
        if 'dynamic' in text:
            return 1
        return 0

    overrides = {}
    for _, row in df.iterrows():
        school = canonicalize_school_name(row.get(school_col, ''))
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

        candidate = {
            'hensachi': int(round(hensachi)),
            'qs': qs,
            'tier': tier,
            'source': source
        }

        if school in overrides:
            existing = overrides[school]
            existing_rank = source_rank(existing.get('source'))
            candidate_rank = source_rank(candidate.get('source'))
            if candidate_rank > existing_rank:
                overrides[school] = candidate
            elif candidate_rank == existing_rank and candidate.get('hensachi', 0) > existing.get('hensachi', 0):
                overrides[school] = candidate
        else:
            overrides[school] = candidate
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


def ielts_to_ibt(ielts_score):
    try:
        score = float(ielts_score or 0)
    except Exception:
        return 0.0

    if score <= 0:
        return 0.0
    if score >= 8.5:
        return 115.0
    if score >= 8.0:
        return 110.0
    if score >= 7.5:
        return 102.0
    if score >= 7.0:
        return 94.0
    if score >= 6.5:
        return 79.0
    if score >= 6.0:
        return 66.0
    if score >= 5.5:
        return 52.0
    if score >= 5.0:
        return 40.0
    if score >= 4.5:
        return 32.0
    return 0.0


def new_toefl_to_ibt(new_toefl_score):
    try:
        score = float(new_toefl_score or 0)
    except Exception:
        return 0.0

    if score <= 0:
        return 0.0
    if score >= 6.0:
        return 114.0
    if score >= 5.5:
        return 107.0
    if score >= 5.0:
        return 95.0
    if score >= 4.5:
        return 86.0
    if score >= 4.0:
        return 72.0
    if score >= 3.5:
        return 58.0
    if score >= 3.0:
        return 44.0
    if score >= 2.5:
        return 34.0
    if score >= 2.0:
        return 24.0
    if score >= 1.5:
        return 12.0
    if score >= 1.0:
        return 0.0
    return 0.0

def get_school_info(school_name):
    UNIVERSITY_DATA = {
        '東京大学': {'hensachi': 72, 'qs': 32, 'tier': 'S级'},
        '京都大学': {'hensachi': 70, 'qs': 50, 'tier': 'S级'},
        '大阪大学': {'hensachi': 67, 'qs': 86, 'tier': 'S级'},
        '東京工業大学': {'hensachi': 67, 'qs': 84, 'tier': 'S级'},
        '東京科学大学': {'hensachi': 67, 'qs': 84, 'tier': 'S级'},
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
    name = canonicalize_school_name(school_name)

    for key, data in UNIVERSITY_DATA.items():
        if key in name:
            return data

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

    # Keep faculty+track labels like 基幹理工学部学系1 as-is.
    if re.search(r'学系[0-9０-９ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]$', cleaned):
        return cleaned

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
        if is_blacklisted_school_faculty(school, faculty):
            continue
        label = display_official_faculty_name(school, faculty)
        if not label or label in {'学部', '研究科'}:
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        options.append({'value': faculty, 'label': label})

    return options


def split_target_faculty_selection(faculty):
    text = str(faculty or '').strip()
    if not text:
        return '', ''

    # e.g. 基幹理工学部学系1 / 先進理工学部化学・生命化学科
    m = re.match(r'^(.*?学部)(.+)$', text)
    if m:
        base = m.group(1).strip()
        tail = m.group(2).strip()
        if tail and re.search(r'(学科|専攻|コース|学系)', tail):
            return base, tail

    return text, ''


def expand_department_aliases(token):
    raw = str(token or '').strip()
    if not raw:
        return []

    aliases = {raw}
    m = re.search(r'学系\s*([0-9０-９ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])', raw)
    if m:
        numeral = m.group(1)
        numeral_map = {
            '1': ['1', '１', 'Ⅰ'],
            '2': ['2', '２', 'Ⅱ'],
            '3': ['3', '３', 'Ⅲ'],
            '4': ['4', '４', 'Ⅳ'],
        }
        canon = {
            '１': '1', '２': '2', '３': '3', '４': '4',
            'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4',
        }.get(numeral, numeral)
        for n in numeral_map.get(canon, [numeral]):
            aliases.add(re.sub(r'学系\s*[0-9０-９ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]', f'学系{n}', raw))

    return [normalize_text(x) for x in aliases if x]


def extend_waseda_target_faculties_from_history(index):
    extended = {school: list(faculties) for school, faculties in (index or {}).items()}

    school_name = '早稲田大学'
    if EJU_DF is None:
        return extended

    school_df = EJU_DF[EJU_DF['受験校'] == school_name]
    if school_df.empty:
        return extended

    base_df = school_df[school_df['学部／研究科'].astype(str).str.contains('基幹理工', na=False)]
    if base_df.empty:
        return extended

    dept_series = base_df['学科／専攻'].fillna('').astype(str)
    wanted = {
        '基幹理工学部学系1': [r'学系\s*(1|１|Ⅰ)'],
        '基幹理工学部学系2': [r'学系\s*(2|２|Ⅱ)'],
        '基幹理工学部学系4': [r'学系\s*(4|４|Ⅳ)'],
    }

    bucket = extended.setdefault(school_name, [])
    for target_name, patterns in wanted.items():
        hit = any(any(re.search(pat, val) for pat in patterns) for val in dept_series)
        if hit and target_name not in bucket:
            bucket.append(target_name)

    extended[school_name] = sorted(list(dict.fromkeys(bucket)))
    return extended


def is_target_faculty_candidate(school, faculty):
    if is_blacklisted_school_faculty(school, faculty):
        return False

    label = display_official_faculty_name(school, faculty)
    if not label:
        return False

    deny_keywords = ['研究科', '大学院', '院', '専攻科', '別科']
    if any(keyword in label for keyword in deny_keywords):
        return False

    return label not in {'学部', '研究科', '全学部', '全学科'}


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
    global EJU_DF, STATS_CACHE, SCHOOL_FACULTY_INDEX, OFFICIAL_SCHOOL_FACULTY_INDEX, OFFICIAL_FACULTY_MAP, SCHOOL_INFO_OVERRIDES, SCHOOL_INFO_DYNAMIC, OFFICIAL_PROGRAM_META, OFFICIAL_SCHOOL_META, TARGET_SCHOOL_FACULTY_INDEX, OFFICIAL_REGION_OPTIONS, OFFICIAL_APPLICATION_MONTH_OPTIONS
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

    # Merge legacy school names into canonical 2025 official naming.
    df['受験校'] = df['受験校'].apply(canonicalize_school_name)
    
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
    OFFICIAL_SCHOOL_META, OFFICIAL_PROGRAM_META, TARGET_SCHOOL_FACULTY_INDEX, OFFICIAL_REGION_OPTIONS, OFFICIAL_APPLICATION_MONTH_OPTIONS = load_official_program_meta()
    TARGET_SCHOOL_FACULTY_INDEX = apply_manual_target_faculty_supplements(TARGET_SCHOOL_FACULTY_INDEX)
    TARGET_SCHOOL_FACULTY_INDEX = extend_waseda_target_faculties_from_history(TARGET_SCHOOL_FACULTY_INDEX)
        
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
                raw_median_score = float(np.median(passed_arr))

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
                raw_std_score = float(np.std(passed_arr)) if len(passed_arr) > 1 else 20.0
                if raw_std_score == 0:
                    raw_std_score = 20.0
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
                raw_median_score = np.nan
                raw_std_score = np.nan
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
                'raw_median_score': raw_median_score,
                'raw_std_score': raw_std_score,
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


def get_prediction_for_profile(user_score, ibt_score, toeic_score, stats, school_info=None, school_name='', faculty_name='', track_name=''):
    def medical_faculty_score_shift(track_name, faculty_text):
        if track_name != 'rika':
            return 0.0

        level = medical_faculty_difficulty_level(faculty_text)
        if level == 3:
            return 95.0
        if level == 2:
            return 78.0
        if level == 1:
            return 28.0
        return 0.0

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
        
    faculty_shift = medical_faculty_score_shift(track_name, faculty_name)

    effective_median = median + faculty_shift
    z = (user_score - effective_median) / std
    
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
    selected_base_faculty, selected_department = split_target_faculty_selection(faculty)
    effective_faculty = selected_base_faculty or faculty

    if (school, effective_faculty, track) in STATS_CACHE:
        resolved_school, resolved_faculty = school, effective_faculty
    else:
        resolved_school, resolved_faculty = resolve_school_faculty_names(school, effective_faculty)
    school_mask = EJU_DF['受験校'] == resolved_school
    faculty_mask = EJU_DF['学部／研究科'] == resolved_faculty
    track_mask = EJU_DF['is_bunka'] if track == 'bunka' else EJU_DF['is_rika']
    matched_scope = 'exact'

    sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
    if len(sub_df) > 0 and selected_department:
        dept_aliases = set(expand_department_aliases(selected_department))
        dep_mask = sub_df['学科／専攻'].astype(str).apply(lambda x: normalize_text(x) in dept_aliases)
        filtered = sub_df[dep_mask]
        if len(filtered) > 0:
            sub_df = filtered
            matched_scope = 'department'

    if len(sub_df) == 0 and '学部／研究科_raw' in EJU_DF.columns:
        faculty_mask = EJU_DF['学部／研究科_raw'] == effective_faculty
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
        if len(sub_df) > 0:
            matched_scope = 'raw'

    if len(sub_df) == 0:
        short_faculty = effective_faculty
        if '学部' in short_faculty and '学部' not in resolved_faculty:
            short_faculty = short_faculty.replace('学部', '')
        faculty_mask = EJU_DF['学部／研究科'].astype(str).str.contains(short_faculty, na=False, regex=False)
        sub_df = EJU_DF[school_mask & faculty_mask & track_mask]
        if len(sub_df) > 0:
            matched_scope = 'contains'

    if len(sub_df) > 0 and selected_department and matched_scope != 'department':
        dept_aliases = set(expand_department_aliases(selected_department))
        dep_mask = sub_df['学科／専攻'].astype(str).apply(lambda x: normalize_text(x) in dept_aliases)
        filtered = sub_df[dep_mask]
        if len(filtered) > 0:
            sub_df = filtered
            matched_scope = 'department'

    if len(sub_df) == 0:
        sub_df = EJU_DF[school_mask & track_mask]
        matched_scope = 'school'

    score_col = '文科EJU总分' if track == 'bunka' else '理科EJU总分'
    full_flag_col = 'has_full_bunka' if track == 'bunka' else 'has_full_rika'

    valid_sub_df = sub_df[sub_df[full_flag_col] == True]
    scores_passed = valid_sub_df[valid_sub_df['合否'] == 1][score_col].dropna().tolist()
    scores_failed = valid_sub_df[valid_sub_df['合否'] == 0][score_col].dropna().tolist()

    cache_key = (resolved_school, resolved_faculty, track)
    stats = None if selected_department else STATS_CACHE.get(cache_key, None)
    if stats is None and not selected_department:
        stats = STATS_CACHE.get((school, effective_faculty, track), None)

    raw_median_score = float(np.median(scores_passed)) if len(scores_passed) > 0 else None
    raw_min_passed = float(np.min(scores_passed)) if len(scores_passed) > 0 else None
    raw_max_passed = float(np.max(scores_passed)) if len(scores_passed) > 0 else None

    stats_data = {}
    if stats:
        stats_data = {
            'total_records': int(stats['total_records']),
            'passed_count': int(stats['passed_count']),
            'failed_count': int(stats['failed_count']),
            # For single-school precise analysis, keep display stats on raw historical distribution
            # so median/min/max are consistent with q10-q90 interval and gap text.
            'median_score': raw_median_score,
            'min_passed': raw_min_passed,
            'max_passed': raw_max_passed,
            'model_median_score': float(stats['median_score']) if pd.notnull(stats['median_score']) else None,
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
        'matched_scope': matched_scope,
        'school_type': get_school_type(resolved_school),
        'school_rank': get_school_rank_tier(resolved_school),
        'official_meta': get_official_program_meta(resolved_school, resolved_faculty),
        'stats': stats_data,
        'scores_passed': scores_passed,
        'scores_failed': scores_failed,
        'prefer_detail_stats': bool(selected_department)
    }


@app.route('/')
def home():
    return send_file('templates/index.html')

@app.route('/api/schools', methods=['GET'])
def get_schools():
    display_index = {}
    all_schools_raw = sorted(set(SCHOOL_FACULTY_INDEX.keys()) | set(OFFICIAL_SCHOOL_FACULTY_INDEX.keys()))
    for school in all_schools_raw:
        if is_excluded_school(school):
            continue
        if is_short_college_school(school):
            continue
        historical_faculties = SCHOOL_FACULTY_INDEX.get(school, [])
        official_faculties = OFFICIAL_SCHOOL_FACULTY_INDEX.get(school, [])
        # Keep official wording while preserving historical faculties that may include bunka options.
        source_faculties = list(dict.fromkeys(list(official_faculties) + list(historical_faculties)))
        options = build_school_faculty_options(school, source_faculties)
        if options:
            display_index[school] = options

    all_schools = sorted(display_index.keys())

    target_display_index = {}
    candidate_target_schools = sorted(set(SCHOOL_FACULTY_INDEX.keys()) | set(TARGET_SCHOOL_FACULTY_INDEX.keys()) | set(OFFICIAL_SCHOOL_FACULTY_INDEX.keys()))
    for school in candidate_target_schools:
        if is_excluded_school(school):
            continue
        if is_short_college_school(school):
            continue
        if school in TARGET_SCHOOL_FACULTY_INDEX:
            # Official target list is the source of truth for target-setting dropdown.
            source_faculties = TARGET_SCHOOL_FACULTY_INDEX.get(school, [])
        elif school in OFFICIAL_SCHOOL_FACULTY_INDEX and school not in TARGET_SCHOOL_FACULTY_INDEX:
            # Officially tracked but not marked as internationally eligible: exclude from target setting.
            continue
        else:
            source_faculties = SCHOOL_FACULTY_INDEX.get(school, [])

        filtered_faculties = [
            faculty for faculty in source_faculties
            if is_target_faculty_candidate(school, faculty) and is_allowed_target_faculty_for_school(school, faculty)
        ]
        options = build_school_faculty_options(school, filtered_faculties)
        if options:
            target_display_index[school] = options

    target_schools = sorted(target_display_index.keys())

    return jsonify({
        'schools': all_schools,
        'index': display_index,
        'target_schools': target_schools,
        'target_index': target_display_index,
        'region_options': OFFICIAL_REGION_OPTIONS,
        'application_month_options': OFFICIAL_APPLICATION_MONTH_OPTIONS
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
    ielts = float(data.get('ielts', 0))
    new_toefl = float(data.get('new_toefl', 0))
    if ibt <= 0 and new_toefl > 0:
        ibt = new_toefl_to_ibt(new_toefl)
    if ibt <= 0 and ielts > 0:
        ibt = ielts_to_ibt(ielts)
    
    # Calculate user EJU total
    user_score = ja + math + sub
    
    # Find precise target if provided
    target_school = canonicalize_school_name(data.get('target_school'))
    target_faculty = data.get('target_faculty')
    mode = data.get('mode', 'prediction')
    region_filter = data.get('region_filter', 'all')
    school_type_filter = data.get('school_type_filter', 'all')
    major_direction_filter = data.get('major_direction_filter', 'all')
    application_month_filter = data.get('application_month_filter', 'all')

    target_res = None
    if target_school and target_faculty:
        if is_excluded_school(target_school):
            target_res = {
                'error': True,
                'message': '该学校已从系统中移除。'
            }
        else:
            is_compatible, incompatible_msg = validate_track_faculty_compatibility(target_school, target_faculty, track)
            if not is_compatible:
                target_res = {
                    'error': True,
                    'message': incompatible_msg
                }
            else:
                detail = build_target_school_detail(target_school, target_faculty, track)
                has_history = detail['stats'].get('passed_count', 0) + detail['stats'].get('failed_count', 0) > 0
                exact_or_faculty_history = detail.get('matched_scope') in {'exact', 'raw', 'contains', 'department'}
                has_passed_history = int(detail['stats'].get('passed_count', 0) or 0) > 0
                if has_history:
                    if not exact_or_faculty_history:
                        target_res = {
                            'error': True,
                            'message': '该学部当前没有可用的历史录取样本，系统不会用整校数据替代生成概率。建议改看相近学部，或先使用智能推荐功能。'
                        }
                    elif not has_passed_history:
                        target_res = {
                            'error': True,
                            'message': '该学部当前只有不合格样本、没有可参考的历史合格样本，系统不会输出误导性的合格概率。建议查看相近学部，或结合募集要项单独评估。'
                        }
                    elif mode == 'target_setting':
                        target_res = {
                            'school': target_school,
                            'faculty': target_faculty,
                            'faculty_display': display_official_faculty_name(target_school, target_faculty),
                            'school_rank': detail['school_rank'],
                            'school_type': detail['school_type'],
                            'official_meta': detail.get('official_meta', {}),
                            'mode': 'target_setting',
                            'stats': detail['stats'],
                            'note': f"基于 {detail['stats']['passed_count'] + detail['stats']['failed_count']} 条历史样本，给出该目标学校学部的总分与各科目标设定区间"
                        }
                    else:
                        stats = build_probability_stats_for_prediction(target_school, target_faculty, track, detail)

                        if stats is not None:
                            prob, msg = get_prediction_for_profile(user_score, ibt, toeic, stats, get_school_info(target_school), target_school, target_faculty, track)
                            tier, tier_color = classify_display_tier(prob, target_faculty)

                            target_res = {
                                'school': target_school,
                                'faculty': target_faculty,
                                'faculty_display': display_official_faculty_name(target_school, target_faculty),
                                'school_rank': detail['school_rank'],
                                'school_type': detail['school_type'],
                                'official_meta': detail.get('official_meta', {}),
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
            if is_excluded_school(school):
                continue
            if is_short_college_school(school):
                continue
            if is_blacklisted_school_faculty(school, faculty):
                continue
            if not is_track_faculty_compatible(track, faculty):
                continue
            if not passes_region_filter(school, region_filter):
                continue
            if not passes_school_type_filter(school, school_type_filter):
                continue
            if not passes_major_direction_filter(track, faculty, major_direction_filter):
                continue
            if not passes_application_month_filter(school, faculty, application_month_filter):
                continue

            # Keep recommendation probability on the same raw historical basis as precision analysis,
            # avoiding inconsistencies between list cards and school-detail view.
            stats_for_prob = {
                'median_score': stats.get('raw_median_score', stats.get('median_score')),
                'std_score': stats.get('raw_std_score', stats.get('std_score')),
                'avg_ibt': stats.get('avg_ibt'),
                'avg_toeic': stats.get('avg_toeic'),
                'target_range_q3': stats.get('target_range_q3'),
                'total_records': stats.get('total_records'),
            }

            prob, _ = get_prediction_for_profile(user_score, ibt, toeic, stats_for_prob, get_school_info(school), school, faculty, track)
            
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

            # Enforce domain-specific medical difficulty presentation rule:
            # 医学部 hardest, 獣医 next, nursing/health branches lower.
            med_level = medical_faculty_difficulty_level(faculty)
            if med_level == 3:
                tier = "冲刺"
                tier_sort = 0
            elif med_level == 2 and tier_sort == 2:
                tier = "适中"
                tier_sort = 1
            elif med_level == 1 and tier_sort == 0:
                tier = "适中"
                tier_sort = 1
                
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
                'median_score': float(stats.get('raw_median_score', stats['median_score'])),
                'total_records': int(stats['total_records']),
                'avg_kijutsu': float(stats['avg_kijutsu']) if stats['avg_kijutsu'] else None,
                'is_high_talent_bonus_school': school in HIGH_TALENT_BONUS_SCHOOLS,
                'application_window': get_official_program_meta(school, faculty).get('application_window', '')
            })
            
    # Sort recommendations by tier, then hensachi (descending), qs (ascending), then probability
    recommendations = sorted(recommendations, key=lambda x: (x['tier_sort'], x['hensachi'], -x['qs'], x['probability']), reverse=True)
    
    # Split into list
    safety = [r for r in recommendations if r['tier'] == "保底"]
    match = [r for r in recommendations if r['tier'] == "适中"]
    challenger = [r for r in recommendations if r['tier'] == "冲刺"]

    # Presentation-layer correction to avoid counter-intuitive difficulty inversions across buckets.
    safety, match, challenger = rebalance_tiers_by_hensachi(safety, match, challenger)
    safety, match, challenger = enforce_medical_hierarchy_buckets(safety, match, challenger)
    safety, match, challenger = enforce_probability_bucket_boundaries(safety, match, challenger)
    
    return jsonify({
        'user_score': user_score,
        'target_result': target_res,
        'applied_filters': {
            'region_filter': region_filter,
            'school_type_filter': school_type_filter,
            'major_direction_filter': major_direction_filter,
            'application_month_filter': application_month_filter
        },
        'recommendations': {
            # Keep a broader result window so mid-tier but suitable schools are not
            # pushed out entirely by a few ultra-competitive faculties at the same band.
            'safety': safety[:80],
            'match': match[:80],
            'challenger': challenger[:80],
            'total_found': len(recommendations)
        }
    })

@app.route('/api/school-detail', methods=['POST'])
def get_school_detail():
    data = request.json or {}
    school = canonicalize_school_name(data.get('school'))
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
        'official_meta': get_official_program_meta(resolved_school, resolved_faculty),
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
