import unittest

import app as app_module


class SupplementalRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_module.load_and_precompute()
        app_module.app.testing = True
        cls.client = app_module.app.test_client()

    def test_supplemental_schools_show_in_indexes(self):
        response = self.client.get('/api/schools')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn('開志専門職大学', data['schools'])
        self.assertIn('新潟医療福祉大学', data['schools'])
        self.assertIn('リハビリテーションカレッジ島根', data['schools'])

        target_index = data['target_index']
        self.assertIn('開志専門職大学', target_index)
        self.assertTrue(any(item['label'] == 'アニメ・漫画学部' for item in target_index['開志専門職大学']))
        self.assertTrue(any(item['label'] == '健康スポーツ学科' for item in target_index['新潟医療福祉大学']))
        self.assertTrue(any(item['label'] == '観光学科' for item in target_index['おたる国際福祉観光専修学院']))

    def test_low_japanese_score_puts_supplemental_schools_into_safety(self):
        response = self.client.post(
            '/api/predict',
            json={
                'mode': 'prediction',
                'track': 'bunka',
                'ja': 220,
                'kijutsu': 25,
                'sub': 120,
                'math': 90,
                'ibt': 0,
                'toeic': 0,
                'ielts': 0,
                'new_toefl': 0,
                'region_filter': ['all'],
                'school_type_filter': 'all',
                'major_direction_filter': 'all',
            }
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        safety_pairs = {(item['school'], item['faculty_display']) for item in data['recommendations']['safety']}
        self.assertIn(('開志専門職大学', 'アニメ・漫画学部'), safety_pairs)
        self.assertIn(('新潟医療福祉大学', '健康スポーツ学科'), safety_pairs)
        self.assertIn(('おたる国際福祉観光専修学院', '観光学科'), safety_pairs)

    def test_target_setting_works_for_supplemental_synthetic_faculty(self):
        response = self.client.post(
            '/api/predict',
            json={
                'mode': 'target_setting',
                'track': 'rika',
                'target_school': '新潟医療福祉大学',
                'target_faculty': '健康データサイエンス学科',
            }
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        target = data['target_result']
        self.assertFalse(target.get('error'), target)
        self.assertEqual(target['school'], '新潟医療福祉大学')
        self.assertEqual(target['faculty_display'], '健康データサイエンス学科')
        self.assertIsNotNone(target['stats']['target_range_q1'])
        self.assertGreater(len(target['stats']['subject_scores']), 0)

    def test_region_filter_returns_expected_supplemental_school(self):
        response = self.client.post(
            '/api/predict',
            json={
                'mode': 'prediction',
                'track': 'bunka',
                'ja': 220,
                'kijutsu': 25,
                'sub': 120,
                'math': 90,
                'region_filter': ['kyushu'],
                'school_type_filter': 'all',
                'major_direction_filter': 'all',
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        buckets = data['recommendations']['safety'] + data['recommendations']['match'] + data['recommendations']['challenger']
        schools = {item['school'] for item in buckets}
        self.assertIn('長崎国際大学', schools)
        self.assertTrue(all(item['school'] in schools for item in buckets))

    def test_major_direction_filter_returns_expected_supplemental_school(self):
        response = self.client.post(
            '/api/predict',
            json={
                'mode': 'prediction',
                'track': 'rika',
                'ja': 225,
                'kijutsu': 30,
                'sub': 110,
                'math': 100,
                'region_filter': ['all'],
                'school_type_filter': 'all',
                'major_direction_filter': 'rika_info',
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        buckets = data['recommendations']['safety'] + data['recommendations']['match'] + data['recommendations']['challenger']
        pairs = {(item['school'], item['faculty_display']) for item in buckets}
        self.assertIn(('開志専門職大学', '情報学部'), pairs)
        self.assertIn(('新潟医療福祉大学', '健康データサイエンス学科'), pairs)

    def test_supplemental_detail_exposes_official_guideline_url(self):
        response = self.client.post(
            '/api/predict',
            json={
                'mode': 'target_setting',
                'track': 'bunka',
                'target_school': '開志専門職大学',
                'target_faculty': 'アニメ・漫画学部',
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        target = data['target_result']
        self.assertFalse(target.get('error'), target)
        self.assertEqual(
            target['official_meta'].get('guideline_url'),
            'https://kaishi-pu.ac.jp/admissions/examination/'
        )


if __name__ == '__main__':
    unittest.main()