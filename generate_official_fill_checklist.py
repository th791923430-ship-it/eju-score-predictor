import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PRIVATE_CSV = "2025年度 理科学部募集要項_私立大学募集要項.csv"
PUBLIC_CSV = "2025年度 理科学部募集要項_国公立大学募集要項.csv"
OUTPUT_CSV = "official_fill_checklist_bunka_priority.csv"

BUNKA_KEYWORDS = [
    "文", "法", "経済", "経営", "商", "社会", "国際", "人文", "教育", "心理", "政策", "観光", "教養", "コミュニケーション"
]

KEY_FIELDS = [
    "募集要项链接（募集要项一览界面的链接）",
    "具体学部PDF",
    "是否招收留学生",
    "EJU必要科目",
    "是否需要英语成绩",
    "出愿开始日期",
    "出愿结束日期",
    "第二次选考/校内考/面试 内容",
]


def clean_school_text(text):
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"（[^）]*）", "", value)
    value = re.sub(r"\([^\)]*\)", "", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_school_name(text):
    value = clean_school_text(text)
    if not value:
        return False
    return bool(re.search(r"(大学院大学|短期大学|大学校|大学|学園大学|学院大学)$", value))


def split_school_and_faculty(text):
    value = clean_school_text(text)
    if not value:
        return "", ""
    match = re.match(r"^(.*?(?:大学院大学|短期大学|大学校|大学|学園大学|学院大学))(?:[\-_ー／/]?)(.+)$", value)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def pick_col(df, *names):
    for name in names:
        if name in df.columns:
            return name
    return None


def infer_track(name, eju_required):
    joined = f"{name} {eju_required}".strip()
    has_bunka = any(k in joined for k in ["総合", "数学コース１", "文科"])
    has_rika = any(k in joined for k in ["物理", "化学", "生物", "数学コース２", "理科"])
    if has_bunka and has_rika:
        return "both"
    if has_bunka:
        return "bunka"
    if has_rika:
        return "rika"
    if any(k in joined for k in BUNKA_KEYWORDS):
        return "bunka_likely"
    return "unknown"


def value_or_empty(row, col):
    if not col:
        return ""
    return str(row.get(col, "")).strip()


def is_missing(value):
    text = str(value or "").strip()
    return text in {"", "0", "未", "未填写", "未更新", "None", "nan", "NaN"}


def collect_rows(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    name_col = df.columns[0]
    owner_col = pick_col(df, "归属大学")
    updated_col = pick_col(df, "是否更新到最新")
    pdf_done_col = pick_col(df, "是否已下载pdf", "募集要项下载完毕")

    current_school = ""
    records = []

    for i, row in df.iterrows():
        raw_name = value_or_empty(row, name_col)
        if not raw_name:
            continue

        owner_raw = value_or_empty(row, owner_col)
        owner_school = clean_school_text(owner_raw)
        clean_name = clean_school_text(raw_name)

        if owner_school and is_school_name(owner_school):
            current_school = owner_school

        if is_school_name(clean_name):
            current_school = clean_name
            level = "school"
            school_name = clean_name
            faculty_name = ""
        else:
            parsed_school, parsed_faculty = split_school_and_faculty(clean_name)
            school_name = owner_school if (owner_school and is_school_name(owner_school)) else (parsed_school or current_school)
            faculty_name = parsed_faculty if parsed_faculty else clean_name
            level = "faculty"

        if not school_name:
            continue

        eju_required = value_or_empty(row, pick_col(df, "EJU必要科目"))
        track = infer_track(clean_name, eju_required)

        missing_fields = []
        for field in KEY_FIELDS:
            if field in df.columns and is_missing(value_or_empty(row, field)):
                missing_fields.append(field)

        note_flags = []
        if "（" in raw_name or "(" in raw_name:
            note_flags.append("name_has_note")
        if updated_col and is_missing(value_or_empty(row, updated_col)):
            note_flags.append("not_marked_latest")
        if pdf_done_col and is_missing(value_or_empty(row, pdf_done_col)):
            note_flags.append("pdf_not_downloaded")

        records.append(
            {
                "source_file": Path(path).name,
                "row_number": i + 2,
                "school": school_name,
                "entry_name": clean_name,
                "entry_level": level,
                "owner_school": owner_school,
                "track_hint": track,
                "missing_count": len(missing_fields),
                "missing_fields": " | ".join(missing_fields),
                "flags": " | ".join(note_flags),
                "guide_link": value_or_empty(row, pick_col(df, "募集要项链接（募集要项一览界面的链接）")),
                "faculty_pdf": value_or_empty(row, pick_col(df, "具体学部PDF")),
                "updated": value_or_empty(row, updated_col),
                "intl": value_or_empty(row, pick_col(df, "是否招收留学生")),
            }
        )

    return records


def aggregate_priority(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["school"]].append(rec)

    output = []
    for school, items in grouped.items():
        bunka_items = [x for x in items if x["track_hint"] in {"bunka", "both", "bunka_likely"}]
        focus_items = bunka_items if bunka_items else items

        missing_total = sum(x["missing_count"] for x in focus_items)
        max_missing = max((x["missing_count"] for x in focus_items), default=0)
        note_rows = sum(1 for x in focus_items if "name_has_note" in x["flags"])
        pending_latest = sum(1 for x in focus_items if "not_marked_latest" in x["flags"])

        score = missing_total * 3 + max_missing * 2 + note_rows * 2 + pending_latest

        sample = sorted(
            focus_items,
            key=lambda x: (-(x["missing_count"]), x["entry_level"], x["entry_name"])
        )[:3]

        sample_desc = " || ".join(
            f"{x['entry_name']} -> {x['missing_fields'] if x['missing_fields'] else 'no_key_missing'}"
            for x in sample
        )

        guide_link = ""
        for x in items:
            if x["guide_link"]:
                guide_link = x["guide_link"]
                break

        output.append(
            {
                "priority_score": score,
                "school": school,
                "focus_track": "bunka_first" if bunka_items else "mixed",
                "focus_rows": len(focus_items),
                "missing_total": missing_total,
                "max_missing_in_row": max_missing,
                "rows_with_note_in_name": note_rows,
                "rows_not_marked_latest": pending_latest,
                "sample_to_verify": sample_desc,
                "official_guide_link": guide_link,
                "suggested_action": "先核对外国人留学生学部清单与文科科目要求，再补齐关键日期字段",
            }
        )

    out_df = pd.DataFrame(output)
    if out_df.empty:
        return out_df

    out_df = out_df.sort_values(
        by=["priority_score", "missing_total", "rows_not_marked_latest", "school"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out_df.insert(0, "priority_rank", range(1, len(out_df) + 1))
    return out_df


def generate():
    all_records = []
    all_records.extend(collect_rows(PRIVATE_CSV))
    all_records.extend(collect_rows(PUBLIC_CSV))

    checklist = aggregate_priority(all_records)
    checklist.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return OUTPUT_CSV, len(checklist)


if __name__ == "__main__":
    output_path, school_count = generate()
    print(f"generated: {output_path} ({school_count} schools)")
