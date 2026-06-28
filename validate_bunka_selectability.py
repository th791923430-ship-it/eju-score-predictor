import argparse
import importlib
import json
import sys
from datetime import datetime

BUNKA_KEYWORDS = [
    "文", "法", "経済", "経営", "商", "社会", "国際", "人文", "教育", "心理", "政策", "観光", "教養", "コミュニケーション"
]

TOKYO_ALLOWED = {"文科一類", "文科二類", "文科三類", "理科一類", "理科二類"}


def has_bunka_hint(text):
    value = str(text or "").strip()
    if not value:
        return False
    return any(k in value for k in BUNKA_KEYWORDS)


def generate_report(include_tokyo=False):
    app_mod = importlib.reload(importlib.import_module("app"))
    app_mod.load_and_precompute()

    client = app_mod.app.test_client()
    payload = client.get("/api/schools").get_json()
    index = payload.get("index", {})

    base_schools = 0
    issues = []

    for school, historical_faculties in app_mod.SCHOOL_FACULTY_INDEX.items():
        bunka_like = [f for f in historical_faculties if has_bunka_hint(f)]
        if not bunka_like:
            continue

        base_schools += 1
        options = index.get(school, [])
        option_values = {str(x.get("value", "")).strip() for x in options}

        if school == "東京大学" and not include_tokyo:
            labels = {str(x.get("label", "")).strip() for x in options}
            if TOKYO_ALLOWED.issubset(labels):
                continue

        if not (set(bunka_like) & option_values):
            issues.append(
                {
                    "school": school,
                    "bunka_history_examples": bunka_like[:10],
                    "selectable_option_count": len(options),
                    "selectable_option_labels_preview": [str(x.get("label", "")).strip() for x in options[:15]],
                }
            )

    summary = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "schools_total_in_ui": len(payload.get("schools", [])),
        "schools_with_bunka_like_history": base_schools,
        "schools_failed_bunka_selectability": len(issues),
        "tokyo_special_rule_excluded": not include_tokyo,
    }

    return {"summary": summary, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description="Validate bunka faculty selectability for all schools in /api/schools")
    parser.add_argument("--output", default="", help="Optional path to save JSON report")
    parser.add_argument("--include-tokyo", action="store_true", help="Do not exclude Tokyo University special category rule")
    parser.add_argument("--fail-on-issues", action="store_true", help="Return non-zero exit code if issues are found")
    args = parser.parse_args()

    report = generate_report(include_tokyo=args.include_tokyo)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if report["issues"]:
        print("\nTop issue samples:")
        for item in report["issues"][:20]:
            print(f"- {item['school']} | options={item['selectable_option_count']} | bunka_examples={item['bunka_history_examples'][:3]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {args.output}")

    if args.fail_on_issues and report["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
