import os
import pandas as pd
import numpy as np
import re


def normalize_text(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    text = re.sub(r'[\s\u3000\(\)\（\）\-\‐\–\—]+', '', text)
    return text.lower()


def load_official_name_maps():
    if not os.path.exists('2025進学調査（そうがく社）.xlsx'):
        print("Official 2025 name file not found; skipping name standardization.")
        return {}, {}, {}

    official = pd.read_excel('2025進学調査（そうがく社）.xlsx', header=3,
                             usecols=['受験校', '学部／研究科など', '学科／専攻／コース／系など'])

    school_map = {
        normalize_text(name): name.strip()
        for name in official['受験校'].dropna().astype(str).unique()
        if normalize_text(name)
    }
    faculty_map = {
        normalize_text(name): name.strip()
        for name in official['学部／研究科など'].dropna().astype(str).unique()
        if normalize_text(name)
    }
    department_map = {
        normalize_text(name): name.strip()
        for name in official['学科／専攻／コース／系など'].dropna().astype(str).unique()
        if normalize_text(name)
    }
    return school_map, faculty_map, department_map

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
    '芸術': '芸術学部',
    '教育': '教育学部',
    '外国語': '外国語学部',
    '人文': '人文学部',
    '情報理工': '情報理工学部',
    '理工': '理工学部',
    '基幹理工': '基幹理工学部',
    '先進理工': '先進理工学部',
    '創造理工': '創造理工学部',
    '環境社会理工': '環境社会理工学部',
    '芸術工': '芸術工学部',
    '建築': '建築学部',
    '薬': '薬学部',
    '歯': '歯学部',
    '医': '医学部',
    '看護': '看護学部',
    '保健福祉': '保健福祉学部',
    '国際教養': '国際教養学部',
    '国際関係': '国際関係学部',
    '総合政策': '総合政策学部',
    '地域政策': '地域政策学部',
    '観光': '観光学部',
    '生命科': '生命科学部',
    '生命環境': '生命環境学部',
    '心理': '心理学部',
    '教養': '教養学部',
    '文理': '文理学部',
    '人間科': '人間科学部',
    '人間社会': '人間社会学部',
    '人間文化': '人間文化学部',
    '人間環境': '人間環境学部',
    '文化構想': '文化構想学部',
    '生物資源科': '生物資源科学部',
    '国際経営': '国際経営学部',
    '経営情報': '経営情報学部',
    '情報': '情報学部',
    '情報通信': '情報通信学部',
    '情報工学': '情報工学部',
    '経営経済': '経営経済学部',
    '社会福祉': '社会福祉学部',
    '生活福祉': '生活福祉学部',
    '現代社会': '現代社会学部',
    '文化社会': '文化社会学部',
    '環境情報': '環境情報学部',
    '環境都市工': '環境都市工学部',
    'デザイン工': 'デザイン工学部',
    '地域創成': '地域創成学部',
}

def format_faculty_label(name):
    if pd.isna(name) or str(name).strip() == '':
        return str(name)
    label = str(name).strip()
    if label in FACULTY_DISPLAY_MAP:
        return FACULTY_DISPLAY_MAP[label]
    if label.endswith('科') and label[:-1] in FACULTY_DISPLAY_MAP:
        return FACULTY_DISPLAY_MAP[label[:-1]]
    if re.search(r'(学部|研究科|学科|院|コース|系)$', label):
        return label
    if label.endswith('学'):
        return label + '部'
    if label.endswith('科'):
        return label
    return label + '学部'


def clean_and_merge():
    # Load 2023
    df23 = pd.read_excel('2023進学調査（そうがく社）.xlsx', header=1)
    df23['年度'] = 2023
    
    # Load 2024
    df24 = pd.read_excel('2024進学調査（そうがく社）.xlsx', header=1)
    df24['年度'] = 2024
    
    # Load 2025
    df25 = pd.read_excel('2025進学調査（そうがく社）.xlsx', header=3)
    df25['年度'] = 2025
    
    # Rename 2025 columns to match 2023/2024
    rename_dict_25 = {
        '学部／研究科など': '学部／研究科',
        '学科／専攻／コース／系など': '学科／専攻',
        '①日語': '日本語_1',
        '①記述': '記述_1',
        '①物理': '物理_1',
        '①化学': '化学_1',
        '①生物': '生物_1',
        '①総合': '総合_1',
        '①数１': '数１_1',
        '①数２': '数２_1',
        '②日語': '日本語_2',
        '②記述': '記述_2',
        '②物理': '物理_2',
        '②化学': '化学_2',
        '②生物': '生物_2',
        '②総合': '総合_2',
        '②数１': '数１_2',
        '②数２': '数２_2',
    }
    df25 = df25.rename(columns=rename_dict_25)
    
    # Rename 2023/2024 columns to match standard
    rename_dict_23_24 = {
        '日本語': '日本語_1',
        '記 述': '記述_1',
        '物 理': '物理_1',
        '化 学': '化学_1',
        '生 物': '生物_1',
        '総合': '総合_1',
        'コース１': '数１_1',
        'コース２': '数２_1',
        '日本語.1': '日本語_2',
        '記 述.1': '記述_2',
        '物 理.1': '物理_2',
        '化 学.1': '化学_2',
        '生 物.1': '生物_2',
        '総合.1': '総合_2',
        'コース１.1': '数１_2',
        'コース２.1': '数２_2',
    }
    df23 = df23.rename(columns=rename_dict_23_24)
    df24 = df24.rename(columns=rename_dict_23_24)
    
    # --- Load New Years (2017, 2018, 2020, 2021) ---
    df17 = pd.read_excel('2017年度　そうがくしゃデータ印刷用(1).xlsx', header=1)
    df17['年度'] = 2017
    df17['大学コード'] = np.nan
    df17['分類１'] = np.nan
    df17['分類２'] = np.nan

    rename_17 = {
        '学部／研究科など': '学部／研究科',
        '学科／専攻／コース／系など': '学科／専攻',
        '日本語': '日本語_1',
        '記 述': '記述_1',
        '物 理': '物理_1',
        '化 学': '化学_1',
        '生 物': '生物_1',
        '総合\n科目': '総合_1',
        'コース１': '数１_1',
        'コース２': '数２_1',
        '日本語.1': '日本語_2',
        '記 述.1': '記述_2',
        '物 理.1': '物理_2',
        '化 学.1': '化学_2',
        '生 物.1': '生物_2',
        '総合\n科目.1': '総合_2',
        'コース１.1': '数１_2',
        'コース２.1': '数２_2',
    }
    df17 = df17.rename(columns=rename_17)

    rename_18_20_21 = {
        'コード': '大学コード',
        '受験大学名': '受験校',
        '学部名': '学部／研究科',
        '学科/専攻/コースなど': '学科／専攻',
        '区①': '分類１',
        '区②': '分類２',
        '①日語': '日本語_1',
        '①記述': '記述_1',
        '①物理': '物理_1',
        '①化学': '化学_1',
        '①生物': '生物_1',
        '①総合': '総合_1',
        '①数１': '数１_1',
        '①数２': '数２_1',
        '②日語': '日本語_2',
        '②記述': '記述_2',
        '②物理': '物理_2',
        '②化学': '化学_2',
        '②生物': '生物_2',
        '②総合': '総合_2',
        '②数１': '数１_2',
        '②数２': '数２_2',
    }

    df18 = pd.read_excel('2018そうがくしゃデータ(1).xlsx', header=0)
    df18['年度'] = 2018
    df18 = df18.rename(columns=rename_18_20_21)

    df20 = pd.read_excel('2020そうがくしゃデータ(1).xlsx', header=0)
    df20['年度'] = 2020
    df20 = df20.rename(columns=rename_18_20_21)

    df21 = pd.read_excel('2021年学部そうがくしゃ データ(2).xlsx', header=0)
    df21['年度'] = 2021
    df21 = df21.rename(columns=rename_18_20_21)

    # Select common columns
    common_cols = [
        '年度', '大学コード', '受験校', '学部／研究科', '学科／専攻', '分類１', '分類２', '合否',
        'N1', 'N2',
        '日本語_1', '記述_1', '物理_1', '化学_1', '生物_1', '総合_1', '数１_1', '数２_1',
        '日本語_2', '記述_2', '物理_2', '化学_2', '生物_2', '総合_2', '数１_2', '数２_2',
        'iBT', 'TOEIC'
    ]
    
    # Filter columns
    df17 = df17[common_cols]
    df18 = df18[common_cols]
    df20 = df20[common_cols]
    df21 = df21[common_cols]
    df23 = df23[common_cols]
    df24 = df24[common_cols]
    df25 = df25[common_cols]
    
    # Standardize '合否' column
    def std_pass_fail(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str in ['〇', '1', '合', '1.0', '○']:
            return 1
        elif val_str in ['×', '0', '否', '0.0']:
            return 0
        return np.nan
        
    df17['合否'] = df17['合否'].apply(std_pass_fail)
    df18['合否'] = df18['合否'].apply(std_pass_fail)
    df20['合否'] = df20['合否'].apply(std_pass_fail)
    df21['合否'] = df21['合否'].apply(std_pass_fail)
    df23['合否'] = df23['合否'].apply(std_pass_fail)
    df24['合否'] = df24['合否'].apply(std_pass_fail)
    df25['合否'] = df25['合否'].apply(std_pass_fail)
    
    # Merge
    merged_df = pd.concat([df17, df18, df20, df21, df23, df24, df25], ignore_index=True)
    
    # Drop rows where '合否' is NaN (since we can't use them for prediction training/matching)
    merged_df = merged_df.dropna(subset=['合否'])
    merged_df['合否'] = merged_df['合否'].astype(int)

    school_map, faculty_map, department_map = load_official_name_maps()
    def map_official_name(value, mapping):
        if pd.isna(value):
            return ''
        raw = str(value).strip()
        normalized = normalize_text(raw)
        if not normalized:
            return raw
        if normalized in mapping:
            return mapping[normalized]

        for suffix in ['学部', '学科', '研究科', '学群', '学域', '系', 'コース', '専攻']:
            candidate = normalize_text(raw + suffix)
            if candidate in mapping:
                return mapping[candidate]

        candidates = [mapped for key, mapped in mapping.items() if normalized in key or key in normalized]
        if len(set(candidates)) == 1:
            return candidates[0]

        return raw

    merged_df['受験校'] = merged_df['受験校'].apply(lambda v: map_official_name(v, school_map))
    merged_df['学部／研究科_raw'] = merged_df['学部／研究科'].astype(str).str.strip()
    merged_df['学科／専攻_raw'] = merged_df['学科／専攻'].astype(str).str.strip()
    merged_df['学部／研究科_official'] = merged_df['学部／研究科_raw'].apply(lambda v: map_official_name(v, faculty_map))
    merged_df['学科／専攻_official'] = merged_df['学科／専攻_raw'].apply(lambda v: map_official_name(v, department_map))
    merged_df['学部／研究科_display'] = merged_df['学部／研究科_official'].apply(format_faculty_label)
    merged_df['学科／専攻_display'] = merged_df['学科／専攻_official'].apply(format_faculty_label)
    merged_df['学部／研究科'] = merged_df['学部／研究科_display']
    merged_df['学科／専攻'] = merged_df['学科／専攻_display']

    # Convert all score columns to numeric (coerce string/spaces to NaN)
    score_cols = [
        'N1', 'N2',
        '日本語_1', '記述_1', '物理_1', '化学_1', '生物_1', '総合_1', '数１_1', '数２_1',
        '日本語_2', '記述_2', '物理_2', '化学_2', '生物_2', '総合_2', '数１_2', '数２_2',
        'iBT', 'TOEIC'
    ]
    for col in score_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        
    print("Merged shape after filtering invalid pass/fail:", merged_df.shape)
    print("Value counts for standardized '合否':")
    print(merged_df['合否'].value_counts(dropna=False))
    
    # Save to clean csv
    merged_df.to_csv('cleaned_eju_data.csv', index=False)
    print("Saved cleaned data to 'cleaned_eju_data.csv'")

if __name__ == '__main__':
    clean_and_merge()
