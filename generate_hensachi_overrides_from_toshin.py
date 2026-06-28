import csv
import importlib
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

OUTPUT = Path('university_hensachi_overrides.csv')
BASE_URL = 'https://www.toshin-hensachi.com/'
KANJI_VARIANTS = {
    '澤': '沢',
    '學': '学',
    '國': '国',
    '﨑': '崎',
    '塚': '塚',
    '髙': '高',
    '濵': '浜',
    'ヶ': 'ケ',
}


def canonicalize_name(text):
    s = str(text or '')
    for src, dst in KANJI_VARIANTS.items():
        s = s.replace(src, dst)
    return s


def fetch_html(url, timeout=30):
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode('utf-8', 'ignore')


def build_url(line, university_type, page):
    params = {
        'line': str(line),
        'university_type': str(university_type),
        'page': str(page),
        'sort': 'deviation',
        'direction': 'DESC',
    }
    return BASE_URL + '?' + urllib.parse.urlencode(params)


def parse_cards(html):
    rows = []
    # Eastin page is server-rendered and repeated card fragments are stable:
    # school-name-university ... school-score(偏差値:x)
    pattern = re.compile(
        r'<span\s+class="school-name-university">\s*([^<]+?)\s*</span>.*?'
        r'<span\s+class="school-score">\s*偏差値：\s*(\d+(?:\.\d+)?)\s*</span>',
        re.S,
    )

    for school, score in pattern.findall(html):
        school = re.sub(r'\s+', '', school)
        rows.append((school, float(score)))
    return rows


def crawl_toshin_scores(max_pages=80, sleep_sec=0.05):
    school_scores = defaultdict(list)
    seen_pages = set()

    # Observed category switches from page links.
    for line in (1, 2):
        for univ_type in (11, 12, 13, 21, 22, 23):
            empty_streak = 0
            for page in range(1, max_pages + 1):
                url = build_url(line, univ_type, page)
                if url in seen_pages:
                    continue
                seen_pages.add(url)

                try:
                    html = fetch_html(url)
                except Exception:
                    empty_streak += 1
                    if empty_streak >= 3:
                        break
                    continue

                rows = parse_cards(html)
                if not rows:
                    empty_streak += 1
                    if empty_streak >= 3:
                        break
                    continue

                empty_streak = 0
                for school, score in rows:
                    school_scores[school].append(score)

                time.sleep(sleep_sec)

    return school_scores


def main():
    app_mod = importlib.reload(importlib.import_module('app'))
    app_mod.load_and_precompute()

    toshin_scores = crawl_toshin_scores()
    schools = sorted(app_mod.SCHOOL_FACULTY_INDEX.keys())

    norm_map = defaultdict(list)
    for name in toshin_scores.keys():
        norm_map[app_mod.normalize_text(canonicalize_name(name))].append(name)

    rows = []
    matched = 0
    for school in schools:
        info = app_mod.get_school_info(school)
        norm = app_mod.normalize_text(canonicalize_name(school))
        candidates = norm_map.get(norm, [])

        values = []
        picked_names = []
        for cand in candidates:
            vals = toshin_scores.get(cand, [])
            if vals:
                values.extend(vals)
                picked_names.append(cand)

        if values:
            h = int(round(statistics.median(values)))
            matched += 1
            source = 'toshin_hensachi_2026'
            note = f'matched={"|".join(sorted(set(picked_names)))[:120]}'
        else:
            h = int(info.get('hensachi', 52))
            source = str(info.get('source', 'fallback_non_toshin'))
            note = 'not_found_in_toshin_keep_existing'

        qs = int(info.get('qs', 9999)) if str(info.get('qs', '')).strip() else 9999
        tier = app_mod.tier_from_hensachi(h)
        rows.append(
            {
                'school': school,
                'hensachi': h,
                'qs': qs,
                'tier': tier,
                'source': source,
                'note': note,
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
    toshin_count = sum(1 for r in rows if r['source'] == 'toshin_hensachi_2026')
    print(
        f'generated: {OUTPUT} rows={len(rows)} toshin_matched={toshin_count} '
        f'fallback={len(rows)-toshin_count} hensachi_50={count_50}'
    )


if __name__ == '__main__':
    main()