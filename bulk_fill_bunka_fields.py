import re
from pathlib import Path

import pandas as pd

FILES = [
    "2025年度 理科学部募集要項_私立大学募集要項.csv",
    "2025年度 理科学部募集要項_国公立大学募集要項.csv",
]

KEY_FIELDS = [
    "募集要项链接（募集要项一览界面的链接）",
    "具体学部PDF",
    "是否招收留学生",
    "EJU必要科目",
    "是否需要英语成绩",
    "英语成绩类型",
    "是否必须JLPT成绩",
    "是否有校内考",
    "出愿开始日期",
    "出愿结束日期",
    "第二次选考/校内考/面试 内容",
]

SAFE_FIELDS_ALL_TRACKS = [
    "募集要项链接（募集要项一览界面的链接）",
    "具体学部PDF",
    "是否招收留学生",
    "是否需要英语成绩",
    "英语成绩类型",
    "是否必须JLPT成绩",
    "是否有校内考",
]

BUNKA_WORDS = [
    "文",
    "法",
    "経済",
    "経営",
    "商",
    "社会",
    "国際",
    "人文",
    "教育",
    "心理",
    "政策",
    "観光",
    "教養",
    "コミュニケーション",
    "総合科目",
    "数学コース1",
    "文科",
]


def is_school_name(text: str) -> bool:
    value = str(text or "").strip()
    return bool(re.search(r"(大学院大学|短期大学|大学校|大学|学園大学|学院大学)$", value))


def clean_school_text(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"（[^）]*）", "", value)
    value = re.sub(r"\([^\)]*\)", "", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_school_and_faculty(text: str):
    value = clean_school_text(text)
    m = re.match(r"^(.*?(?:大学院大学|短期大学|大学校|大学|学園大学|学院大学))(?:[\-_ー／/]?)(.+)$", value)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


def is_missing(v: str) -> bool:
    t = str(v or "").strip()
    return t in {"", "0", "未", "未填写", "未更新", "None", "nan", "NaN"}


def is_bunka_like(name: str, eju: str) -> bool:
    joined = f"{name} {eju}".strip()
    return any(word in joined for word in BUNKA_WORDS)


def fill_one_file(file_path: Path):
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    name_col = df.columns[0]

    owner_col = "归属大学" if "归属大学" in df.columns else None
    update_col = "是否更新到最新" if "是否更新到最新" in df.columns else None

    for field in KEY_FIELDS:
        if field not in df.columns:
            df[field] = ""

    school_ctx = {}
    faculty_ctx = {}
    current_school = ""

    changed = 0

    # Pass 1: build context from rows already filled.
    for i, row in df.iterrows():
        raw_name = str(row[name_col]).strip()
        if not raw_name:
            continue

        owner = clean_school_text(row.get(owner_col, "")) if owner_col else ""
        clean_name = clean_school_text(raw_name)

        if owner and is_school_name(owner):
            current_school = owner
        if is_school_name(clean_name):
            current_school = clean_name

        school, faculty = split_school_and_faculty(clean_name)
        school_name = owner if (owner and is_school_name(owner)) else (school or current_school)

        if not school_name:
            continue

        non_empty = {f: str(row.get(f, "")).strip() for f in KEY_FIELDS if not is_missing(row.get(f, ""))}
        if not non_empty:
            continue

        school_ctx.setdefault(school_name, {}).update(non_empty)

        if school_name and faculty:
            key = (school_name, faculty)
            faculty_ctx.setdefault(key, {}).update(non_empty)

    # Pass 2: fill bunka-like rows.
    current_school = ""
    for i, row in df.iterrows():
        raw_name = str(row[name_col]).strip()
        if not raw_name:
            continue

        owner = clean_school_text(row.get(owner_col, "")) if owner_col else ""
        clean_name = clean_school_text(raw_name)

        if owner and is_school_name(owner):
            current_school = owner
        if is_school_name(clean_name):
            current_school = clean_name
            continue

        school, faculty = split_school_and_faculty(clean_name)
        school_name = owner if (owner and is_school_name(owner)) else (school or current_school)
        if not school_name:
            continue

        eju_text = str(row.get("EJU必要科目", "")).strip()
        bunka_like = is_bunka_like(clean_name, eju_text)

        sources = []
        if faculty:
            sources.append(faculty_ctx.get((school_name, faculty), {}))
        sources.append(school_ctx.get(school_name, {}))

        row_changed = False
        target_fields = KEY_FIELDS if bunka_like else SAFE_FIELDS_ALL_TRACKS
        for f in target_fields:
            if not is_missing(row.get(f, "")):
                continue
            for src in sources:
                val = str(src.get(f, "")).strip()
                if not is_missing(val):
                    df.at[i, f] = val
                    row_changed = True
                    break

        # Keep owner university consistent for child rows.
        if owner_col and not is_missing(school_name) and is_missing(row.get(owner_col, "")):
            df.at[i, owner_col] = school_name
            row_changed = True

        if update_col and row_changed and is_missing(row.get(update_col, "")):
            df.at[i, update_col] = "1"

        if row_changed:
            changed += 1

    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return changed, len(df)


def main():
    for file_name in FILES:
        path = Path(file_name)
        changed, total = fill_one_file(path)
        print(f"{file_name}: changed_rows={changed}, total_rows={total}")


if __name__ == "__main__":
    main()
