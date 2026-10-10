import unittest

import app as app_module


class OEJUSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_module.app.config['AUTH_ENABLED'] = False

    def predict(self, **changes):
        payload = dict(track='bunka', ja=200, math=60, sub=80)
        payload.update(changes)
        response = app_module.app.test_client().post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        return response.get_json()['recommendations']

    def test_low_scores_include_catalog_with_conditions_without_fake_statistics(self):
        data = self.predict()
        cards = [i for i in data['safety'] if i.get('recommendation_basis') == 'oeju_n2_rule']
        schools = {i['school'] for i in cards}
        self.assertIn('帝京大学', schools)
        self.assertIn('九州共立大学', schools)
        self.assertIn('平安女学院大学', schools)
        self.assertIn('リハビリテーションカレッジ島根', schools)
        self.assertNotIn('事業創造大学院大学', schools)
        self.assertNotIn('新潟医療福祉大学', schools)
        self.assertEqual(len(schools), len(cards))
        for card in cards:
            self.assertIsNone(card['probability'])
            self.assertIsNone(card['median_score'])
            self.assertTrue(card['admission_requirements'])
            self.assertTrue(card['condition_note'])

    def test_catalog_respects_filters_and_low_score_scope(self):
        for changes in [dict(ja=320, math=160, sub=160),
                        dict(school_type_filter='public'),
                        dict(application_month_filter='9')]:
            data = self.predict(**changes)
            self.assertFalse(any(i.get('recommendation_basis') == 'oeju_n2_rule'
                                 for i in data['safety']))
        data = self.predict(region_filter=['kyushu'])
        cards = [i for i in data['safety'] if i.get('recommendation_basis') == 'oeju_n2_rule']
        self.assertIn('長崎国際大学', {i['school'] for i in cards})
        self.assertNotIn('帝京大学', {i['school'] for i in cards})
        data = self.predict(track='rika', major_direction_filter='rika_info')
        cards = [i for i in data['safety'] if i.get('recommendation_basis') == 'oeju_n2_rule']
        self.assertIn('開志専門職大学', {i['school'] for i in cards})
        self.assertNotIn('平成音楽大学', {i['school'] for i in cards})


if __name__ == '__main__':
    unittest.main()
