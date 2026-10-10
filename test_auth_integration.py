import re
import tempfile
import unittest

import app as app_module
from auth import AccountStore


class PlatformLoginIntegrationTests(unittest.TestCase):
    def test_login_protects_real_recommendations_and_page_supplies_csrf(self):
        app = app_module.app
        previous_store = app.extensions['account_store']
        previous_enabled = app.config['AUTH_ENABLED']
        with tempfile.TemporaryDirectory() as directory:
            account_store = AccountStore('sqlite:///' + directory + '/accounts.sqlite3')
            app.extensions['account_store'] = account_store
            app.config['AUTH_ENABLED'] = True
            try:
                account_store.create('test_student', '测试学生', 'integration-test-password', must_change=False)
                client = app.test_client()
                self.assertEqual(client.get('/api/schools').status_code, 401)
                self.assertEqual(client.post('/api/predict', json={}).status_code, 401)
                token = client.get('/api/auth/me').get_json()['csrf_token']
                login = client.post('/login', data=dict(username='test_student', password='integration-test-password', csrf_token=token))
                self.assertEqual(login.location, '/')
                page = client.get('/')
                self.assertEqual(page.status_code, 200)
                html = page.get_data(as_text=True)
                self.assertNotIn('__AUTH_', html)
                self.assertIn('测试学生', html)
                self.assertNotIn('<a href="/admin/accounts">账号管理</a>', html)
                token = re.search(r'<meta name="csrf-token" content="([^"]+)"', html).group(1)
                self.assertEqual(client.get('/api/schools').status_code, 200)
                response = client.post('/api/predict', json=dict(track='bunka', ja=200, math=60, sub=80), headers={'X-CSRF-Token': token})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(any(item.get('recommendation_basis') == 'oeju_n2_rule'
                                    for item in response.get_json()['recommendations']['safety']))
                self.assertEqual(client.get('/admin/accounts').status_code, 403)
            finally:
                app.extensions['account_store'] = previous_store
                app.config['AUTH_ENABLED'] = previous_enabled
                account_store.engine.dispose()


if __name__ == '__main__':
    unittest.main()
