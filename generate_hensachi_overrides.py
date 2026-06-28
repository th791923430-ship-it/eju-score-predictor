import csv
import importlib
from pathlib import Path

OUTPUT = Path('university_hensachi_overrides.csv')


def main():
    app_mod = importlib.reload(importlib.import_module('app'))
    app_mod.load_and_precompute()

    schools = sorted(app_mod.SCHOOL_FACULTY_INDEX.keys())
    rows = []
    for school in schools:
        info = app_mod.get_school_info(school)
        rows.append(
            {
                'school': school,
                'hensachi': int(info.get('hensachi', 52)),
                'qs': int(info.get('qs', 9999)) if str(info.get('qs', '')).strip() else 9999,
                'tier': str(info.get('tier', 'C级')),
                'source': str(info.get('source', 'static_or_fallback')),
                'note': 'auto-generated; review and adjust if needed',
            }
        )

    with OUTPUT.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['school', 'hensachi', 'qs', 'tier', 'source', 'note'],
        )
        writer.writeheader()
        writer.writerows(rows)

    count_50 = sum(1 for r in rows if int(r['hensachi']) == 50)
    print(f'generated: {OUTPUT} rows={len(rows)} hensachi_50={count_50}')


if __name__ == '__main__':
    main()
