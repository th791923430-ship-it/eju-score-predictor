import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from flask import Flask, jsonify
from werkzeug.security import check_password_hash

from auth import AccountStore, init_auth, users


class AccountAccessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))
        self.app.config.update(TESTING=True, SECRET_KEY='test-only-signing-key',
                               AUTH_DATABASE_URL='sqlite:///' + self.directory.name + '/accounts.sqlite3')
        init_auth(self.app)
        self.store = self.app.extensions['account_store']
        self.store.create('admin01', '老师', 'admin-test-password', role='admin', must_change=False)
        self.store.create('student01', '学生', 'student-test-password')

        @self.app.route('/')
        def home():
            return 'protected homepage'

        @self.app.route('/api/private', methods=['GET', 'POST'])
        def private():
            return jsonify(ok=True)

        self.client = self.app.test_client()

    def tearDown(self):
        self.store.engine.dispose()
        self.directory.cleanup()

    def token(self, client=None):
        return (client or self.client).get('/api/auth/me').get_json()['csrf_token']

    def post(self, path, data, client=None):
        client = client or self.client
        return client.post(path, data={**data, 'csrf_token': self.token(client)})

    def login(self, username='admin01', password='admin-test-password', client=None):
        return self.post('/login', dict(username=username, password=password), client)

    def finish_student_password(self):
        self.login('student01', 'student-test-password')
        self.post('/account/password', dict(current_password='student-test-password',
                  password='changed-student-password', confirm_password='changed-student-password'))
        return self.login('student01', 'changed-student-password')

    def test_anonymous_home_and_apis_are_protected(self):
        self.assertEqual(self.client.get('/').location, '/login')
        self.assertEqual(self.client.get('/api/private').status_code, 401)
        self.assertEqual(self.client.post('/api/private', json={}).status_code, 401)
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.assertEqual(self.client.get('/login').status_code, 200)

    def test_csrf_login_api_and_logout(self):
        self.assertEqual(self.client.post('/login', data=dict(username='admin01', password='admin-test-password')).status_code, 403)
        self.login()
        self.assertEqual(self.client.get('/api/private').status_code, 200)
        self.assertEqual(self.client.post('/api/private', json={}).status_code, 403)
        self.assertEqual(self.client.post('/api/private', json={}, headers={'X-CSRF-Token': self.token()}).status_code, 200)
        self.post('/logout', {})
        self.assertEqual(self.client.get('/api/private').status_code, 401)

    def test_first_login_and_student_role(self):
        response = self.login('student01', 'student-test-password')
        self.assertEqual(response.location, '/account/password')
        self.assertEqual(self.client.get('/api/private').get_json()['error'], 'password_change_required')
        self.finish_student_password()
        self.assertEqual(self.client.get('/api/private').status_code, 200)
        self.assertEqual(self.client.get('/admin/accounts').status_code, 403)
        self.assertEqual(self.post('/admin/accounts', {'action': 'create', 'username': 'intruder', 'display_name': 'x', 'password': 'long-enough-password'}).status_code, 403)

    def test_admin_create_duplicate_and_password_hash_persistence(self):
        self.login()
        payload = dict(action='create', username='Student02', display_name='测试学生', password='new-student-password', role='admin')
        self.assertEqual(self.post('/admin/accounts', payload).status_code, 302)
        account = self.store.find(username='student02')
        self.assertEqual(account['role'], 'student')
        self.assertNotEqual(account['password_hash'], payload['password'])
        self.assertTrue(check_password_hash(account['password_hash'], payload['password']))
        self.assertTrue(account['must_change_password'])
        response = self.post('/admin/accounts', payload)
        self.assertIn('账号已存在', response.get_data(as_text=True))
        reopened = AccountStore(self.app.config['AUTH_DATABASE_URL'])
        self.assertEqual(reopened.find(username='student02')['id'], account['id'])
        reopened.engine.dispose()
        self.assertNotIn(payload['password'], self.client.get('/admin/accounts').get_data(as_text=True))

    def test_disable_and_reset_revoke_sessions(self):
        self.finish_student_password()
        admin = self.app.test_client()
        self.login(client=admin)
        account = self.store.find(username='student01')
        self.post('/admin/accounts', dict(action='toggle', user_id=account['id']), admin)
        self.assertEqual(self.client.get('/api/private').status_code, 401)
        self.post('/admin/accounts', dict(action='toggle', user_id=account['id']), admin)
        self.login('student01', 'changed-student-password')
        self.assertEqual(self.client.get('/api/private').status_code, 200)
        self.post('/admin/accounts', dict(action='reset', user_id=account['id'], password='new-reset-password'), admin)
        self.assertEqual(self.client.get('/api/private').status_code, 401)
        self.assertEqual(self.login('student01', 'new-reset-password').location, '/account/password')
        self.assertEqual(self.post('/admin/accounts', dict(action='toggle', user_id=self.store.find(username='admin01')['id']), admin).status_code, 403)

    def test_failed_logins_are_locked_and_bad_password_rejected(self):
        for _ in range(5):
            self.assertEqual(self.login('student01', 'wrong-password').status_code, 401)
        self.assertGreater(self.store.find(username='student01')['locked_until'], 0)
        self.assertEqual(self.login('student01', 'student-test-password').status_code, 401)
        self.login()
        response = self.post('/admin/accounts', dict(action='create', username='student02', display_name='x', password='short'))
        self.assertIn('密码长度', response.get_data(as_text=True))
        self.assertIsNone(self.store.find(username='student02'))

    def test_display_names_are_escaped(self):
        self.store.create('student02', '<script>alert(1)</script>', 'student-test-password')
        self.login()
        html = self.client.get('/admin/accounts').get_data(as_text=True)
        self.assertNotIn('<script>alert(1)</script>', html)
        self.assertIn('&lt;script&gt;', html)

    def test_production_requires_secret_and_persistent_database_configuration(self):
        with patch.dict(os.environ, {'RENDER': 'true', 'AUTH_ENABLED': '1'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'SECRET_KEY'):
                init_auth(Flask('missing-secret'))
            app = Flask('missing-database')
            app.config['SECRET_KEY'] = 'test-signing-key-that-is-long-enough'
            with self.assertRaisesRegex(RuntimeError, 'DATABASE_URL'):
                init_auth(app)

    def test_render_configuration_uses_secure_cookies_and_forwarded_address(self):
        app = Flask('render-test', template_folder=str(Path(__file__).parent / 'templates'))
        app.config.update(SECRET_KEY='test-signing-key-that-is-long-enough',
                          AUTH_DATABASE_URL=self.app.config['AUTH_DATABASE_URL'])
        with patch.dict(os.environ, {'RENDER': 'true', 'AUTH_ENABLED': '1'}, clear=True):
            init_auth(app)
        client = app.test_client()
        response = client.get('/api/auth/me', base_url='https://platform.example')
        cookie = response.headers['Set-Cookie']
        self.assertIn('Secure', cookie)
        self.assertIn('HttpOnly', cookie)
        token = response.get_json()['csrf_token']
        client.post('/login', base_url='https://platform.example',
                    headers={'X-Forwarded-For': '192.0.2.1', 'X-Forwarded-Proto': 'https'},
                    data=dict(username='nobody', password='wrong-password', csrf_token=token))
        self.assertIn('192.0.2.1', app.extensions['login_attempts'])
        app.extensions['account_store'].engine.dispose()

    def test_public_render_release_needs_no_credentials_and_exposes_no_accounts(self):
        app = Flask('public-render')
        with patch.dict(os.environ, {'RENDER': 'true'}, clear=True):
            init_auth(app)
        self.assertFalse(app.config['AUTH_ENABLED'])
        self.assertNotIn('account_store', app.extensions)
        client = app.test_client()
        self.assertEqual(client.get('/healthz').status_code, 200)
        for path in ['/login', '/admin/accounts', '/account/password', '/api/auth/me']:
            self.assertEqual(client.get(path).status_code, 404)


if __name__ == '__main__':
    unittest.main()
