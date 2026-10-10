"""Administrator-issued accounts, kept separate from admissions data."""
import os
import re
import secrets
import threading
import time
from datetime import timedelta
from pathlib import Path

import click
from flask import (Blueprint, abort, current_app, flash, g, jsonify,
                   redirect, render_template, request, session, url_for)
from sqlalchemy import (Boolean, Column, Integer, MetaData, String, Table,
                        create_engine, insert, select, update)
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix


metadata = MetaData()
users = Table(
    'platform_users', metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String(64), nullable=False, unique=True),
    Column('display_name', String(80), nullable=False),
    Column('password_hash', String(256), nullable=False),
    Column('role', String(16), nullable=False),
    Column('active', Boolean, nullable=False, default=True),
    Column('must_change_password', Boolean, nullable=False, default=True),
    Column('session_version', Integer, nullable=False, default=1),
    Column('failed_logins', Integer, nullable=False, default=0),
    Column('locked_until', Integer, nullable=False, default=0),
)
auth = Blueprint('auth', __name__)


def normalize_username(value):
    username = str(value or '').strip().lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.@-]{2,63}', username):
        raise ValueError('账号需为3–64位字母、数字或 _.@-，以字母或数字开头。')
    return username


def password_error(password):
    if not isinstance(password, str) or not 12 <= len(password) <= 128:
        return '密码长度需为12–128个字符。'
    return None


class AccountStore:
    def __init__(self, database_url):
        if database_url.startswith(('postgres://', 'postgresql://')):
            database_url = 'postgresql+psycopg://' + database_url.split('://', 1)[1]
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.ready = False
        self.schema_lock = threading.Lock()

    def ensure_schema(self):
        with self.schema_lock:
            if not self.ready:
                metadata.create_all(self.engine)
                self.ready = True

    def find(self, **filters):
        self.ensure_schema()
        with self.engine.connect() as conn:
            return conn.execute(select(users).filter_by(**filters)).mappings().first()

    def create(self, username, display_name, password, role='student', must_change=True):
        username = normalize_username(username)
        error = password_error(password)
        if error:
            raise ValueError(error)
        name = str(display_name or '').strip()
        if not name or len(name) > 80:
            raise ValueError('姓名需为1–80个字符。')
        if role not in {'student', 'admin'}:
            raise ValueError('无效的账号角色。')
        self.ensure_schema()
        with self.engine.begin() as conn:
            conn.execute(insert(users).values(
                username=username, display_name=name,
                password_hash=generate_password_hash(password, method='pbkdf2:sha256:1000000'), role=role,
                active=True, must_change_password=must_change,
                session_version=1, failed_logins=0, locked_until=0))

    def revise(self, user_id, **values):
        with self.engine.begin() as conn:
            conn.execute(update(users).where(users.c.id == user_id).values(**values))

    def revoke(self, user_id, **values):
        # SQL increment prevents concurrent admin updates from losing revocations.
        with self.engine.begin() as conn:
            conn.execute(update(users).where(users.c.id == user_id).values(
                session_version=users.c.session_version + 1, **values))


def store():
    return current_app.extensions['account_store']


def csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return session['csrf_token']


def init_auth(app):
    production = os.environ.get('RENDER', '').lower() == 'true'
    if production:
        # Render terminates HTTPS and forwards through its edge proxy.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    # Public Render release is intentional until persistent account storage is ready.
    # Local development retains login; production enables it explicitly.
    app.config.setdefault('AUTH_ENABLED', os.environ.get(
        'AUTH_ENABLED', '0' if production else '1') == '1')
    if not app.config['AUTH_ENABLED']:
        @app.before_request
        def public_release():
            g.user = None

        app.add_url_rule('/healthz', endpoint='public_health', view_func=health)
        return
    secret = app.config.get('SECRET_KEY') or os.environ.get('SECRET_KEY')
    if not secret:
        if production:
            raise RuntimeError('Set SECRET_KEY in Render before deploying account login.')
        directory = Path(app.instance_path)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'session-secret'
        try:
            with path.open('x') as output:
                output.write(secrets.token_urlsafe(48))
            path.chmod(0o600)
        except FileExistsError:
            pass
        secret = path.read_text().strip()
    if production and len(secret) < 32:
        raise RuntimeError('SECRET_KEY must contain at least 32 random characters in Render.')
    database_url = app.config.get('AUTH_DATABASE_URL') or os.environ.get('DATABASE_URL')
    if not database_url:
        db_path = os.environ.get('AUTH_DB_PATH')
        if production and not db_path:
            raise RuntimeError('Set DATABASE_URL, or AUTH_DB_PATH on a persistent disk, in Render.')
        db_path = Path(db_path or Path(app.instance_path) / 'accounts.sqlite3').resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = 'sqlite:///' + str(db_path)
    app.config.update(
        SECRET_KEY=secret, SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_NAME='eju_platform_session',
        SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=production,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8), MAX_CONTENT_LENGTH=65536,
    )
    app.extensions['account_store'] = AccountStore(database_url)
    app.extensions['login_attempts'] = {}
    app.extensions['login_attempts_lock'] = threading.Lock()
    app.extensions['dummy_password_hash'] = generate_password_hash(secrets.token_urlsafe(24), method='pbkdf2:sha256:1000000')
    app.jinja_env.globals['csrf_token'] = csrf_token
    app.register_blueprint(auth)

    @app.before_request
    def protect_access():
        g.user = None
        if not current_app.config['AUTH_ENABLED']:
            return None
        if session.get('user_id'):
            account = store().find(id=session['user_id'])
            if account and account['active'] and account['session_version'] == session.get('session_version'):
                g.user = account
            else:
                session.clear()
        public = request.endpoint in {'auth.login', 'auth.health', 'auth.me', 'static'}
        if not public and not g.user:
            if request.path.startswith('/api/'):
                return jsonify(error='login_required', message='请先登录。'), 401
            return redirect(url_for('auth.login'))
        if g.user and g.user['must_change_password'] and request.endpoint not in {
                'auth.change_password', 'auth.logout', 'auth.me', 'auth.health', 'static'}:
            if request.path.startswith('/api/'):
                return jsonify(error='password_change_required', message='请先修改初始密码。'), 403
            return redirect(url_for('auth.change_password'))
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            supplied = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token', '')
            expected = session.get('csrf_token', '')
            if not supplied or not expected or not secrets.compare_digest(supplied, expected):
                if request.path.startswith('/api/'):
                    return jsonify(error='csrf_failed', message='页面已过期，请刷新后重试。'), 403
                abort(403, '页面已过期，请刷新后重试。')

    @app.after_request
    def private_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        if current_app.config['AUTH_ENABLED']:
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.cli.command('create-admin')
    @click.argument('username')
    @click.option('--name', default='管理员')
    def create_admin(username, name):
        """Create an administrator; password is entered privately, never in arguments."""
        password = click.prompt('管理员密码（至少12位）', hide_input=True, confirmation_prompt=True)
        try:
            store().create(username, name, password, role='admin', must_change=False)
        except (ValueError, IntegrityError) as exc:
            raise click.ClickException('账号已存在。' if isinstance(exc, IntegrityError) else str(exc))
        click.echo('管理员账号已创建。')


@auth.route('/healthz')
def health():
    return jsonify(status='ok')


@auth.route('/api/auth/me')
def me():
    user = g.user
    return jsonify(authenticated=bool(user), csrf_token=csrf_token(),
                   user=({'username': user['username'], 'display_name': user['display_name'],
                          'role': user['role']} if user else None))


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect('/account/password' if g.user['must_change_password'] else '/')
    error = None
    if request.method == 'POST':
        # Bound password work per source address. Account lockout is persisted too.
        now = int(time.time())
        key = request.remote_addr or 'unknown'
        attempts = current_app.extensions['login_attempts']
        with current_app.extensions['login_attempts_lock']:
            expired = [ip for ip, values in attempts.items() if now - values[0] >= 900]
            for ip in expired:
                del attempts[ip]
            first, count = attempts.get(key, (now, 0))
            attempts[key] = (first, count + 1)
        if count >= 100:
            return render_template('login.html', error='尝试过于频繁，请15分钟后重试。'), 429
        try:
            username = normalize_username(request.form.get('username'))
        except ValueError:
            username = ''
        account = store().find(username=username)
        password = request.form.get('password', '')
        candidate_hash = account['password_hash'] if account else current_app.extensions['dummy_password_hash']
        correct = len(password) <= 128 and check_password_hash(candidate_hash, password)
        if account and account['active'] and account['locked_until'] <= now and correct:
            store().revise(account['id'], failed_logins=0, locked_until=0)
            session.clear()
            session.update(user_id=account['id'], session_version=account['session_version'])
            session.permanent = True
            csrf_token()
            return redirect('/account/password' if account['must_change_password'] else '/')
        if account and account['locked_until'] <= now:
            with store().engine.begin() as conn:
                current = conn.execute(select(users.c.failed_logins).where(users.c.id == account['id'])).scalar_one()
                failures = current + 1
                conn.execute(update(users).where(users.c.id == account['id']).values(
                    failed_logins=failures, locked_until=now + 900 if failures >= 5 else 0))
        error = '账号或密码错误，或账号已停用／暂时锁定。请联系管理员或稍后重试。'
    return render_template('login.html', error=error), 401 if error else 200


@auth.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('auth.login'))


@auth.route('/account/password', methods=['GET', 'POST'])
def change_password():
    error = None
    if request.method == 'POST':
        password = request.form.get('password', '')
        error = password_error(password)
        current_password = request.form.get('current_password', '')
        if len(current_password) > 128 or not check_password_hash(g.user['password_hash'], current_password):
            error = '当前密码不正确。'
        elif password != request.form.get('confirm_password'):
            error = '两次输入的新密码不一致。'
        elif check_password_hash(g.user['password_hash'], password):
            error = '新密码需与当前密码不同。'
        if not error:
            store().revoke(g.user['id'], password_hash=generate_password_hash(password, method='pbkdf2:sha256:1000000'),
                           must_change_password=False, failed_logins=0, locked_until=0)
            session.clear()
            flash('密码已更新，请使用新密码登录。')
            return redirect(url_for('auth.login'))
    return render_template('password.html', error=error)


@auth.route('/admin/accounts', methods=['GET', 'POST'])
def accounts():
    if g.user['role'] != 'admin':
        abort(403)
    error = None
    if request.method == 'POST':
        action = request.form.get('action')
        try:
            if action == 'create':
                store().create(request.form.get('username'), request.form.get('display_name'),
                               request.form.get('password'))
                flash('学生账号已创建，请将账号和初始密码交给学生。')
            else:
                account = store().find(id=int(request.form.get('user_id', '0')))
                if not account or account['role'] != 'student':
                    abort(403)
                if action == 'toggle':
                    store().revoke(account['id'], active=not account['active'])
                    flash('账号状态已更新，之前的登录会话已失效。')
                elif action == 'reset':
                    password = request.form.get('password', '')
                    if password_error(password):
                        raise ValueError(password_error(password))
                    store().revoke(account['id'], password_hash=generate_password_hash(password, method='pbkdf2:sha256:1000000'),
                                   must_change_password=True, failed_logins=0, locked_until=0)
                    flash('初始密码已重置，学生下次登录需修改密码。')
                else:
                    abort(400)
            return redirect(url_for('auth.accounts'))
        except IntegrityError:
            error = '账号已存在，请使用其他账号名。'
        except ValueError as exc:
            error = str(exc)
    with store().engine.connect() as conn:
        students = conn.execute(select(users).where(users.c.role == 'student').order_by(users.c.id.desc())).mappings().all()
    return render_template('accounts.html', students=students, error=error)
