import unittest
from pathlib import Path

import pandas as pd
import app as app_module


class LatestSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_module.app.config['AUTH_ENABLED'] = False

    def test_latest_survey_is_loaded_without_losing_scores(self):
        path = next(Path(__file__).parent.glob('2026進学調査*.xlsx'))
        raw = pd.read_excel(path, header=4)
        # The runtime app appends supplemental copies; compare original survey rows.
        csv = pd.read_csv(Path(__file__).parent / 'cleaned_eju_data.csv')
        original = app_module.EJU_DF.iloc[:len(csv)]
        latest = original[original['年度'] == 2026]
        self.assertEqual(len(latest), len(raw))
        self.assertEqual(int(latest['合否'].sum()), int((raw['合否'] == '合').sum()))
        for source, target in [('①日語', '日本語_1'), ('②総合', '総合_2'),
                               ('①数２', '数２_1'), ('iBT', 'iBT')]:
            expected = pd.to_numeric(raw[source], errors='coerce').fillna(0)
            actual = latest[target].fillna(0).reset_index(drop=True)
            pd.testing.assert_series_equal(actual, expected, check_names=False,
                                           check_dtype=False)

    def test_latest_records_contribute_to_recommendation_statistics(self):
        df = app_module.EJU_DF
        latest = df[(df['年度'] == 2026) & df['has_full_bunka'] & (df['合否'] == 1)]
        school, faculty = latest.groupby(['受験校', '学部／研究科']).size().idxmax()
        group = df[(df['受験校'] == school) & (df['学部／研究科'] == faculty) & df['is_bunka']]
        stats = app_module.STATS_CACHE[(school, faculty, 'bunka')]
        self.assertEqual(stats['total_records'], len(group))
        self.assertGreater(stats['total_records'], len(group[group['年度'] < 2026]))
        accepted = group[group['has_full_bunka'] & (group['合否'] == 1)]
        self.assertEqual(stats['raw_median_score'], accepted['文科EJU总分'].median())


if __name__ == '__main__':
    unittest.main()
