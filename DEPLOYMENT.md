# 上线与管理员账号

目标地址：`https://www.platform.xly-edu.com`。
原 Render 服务：`https://eju-score-predictor.onrender.com`。
原官网 `https://www.xly-edu.com` 保留。

## 先确认现有配置

登录 https://dashboard.render.com ，打开 `eju-score-predictor` 服务，查看：

- Settings：实例类型、GitHub 仓库与分支、构建和启动命令、Health Check。
- Environment：是否已有 `SECRET_KEY`、`DATABASE_URL`（仅确认名称，不要把值发送到聊天）。
- 工作区：是否已有 PostgreSQL 数据库，是否属于同一个服务／区域，是否有到期时间。
- 如果使用 SQLite：服务是否已挂载持久磁盘，以及实际 Mount Path。

仓库当前 `render.yaml` 的实例类型为 `free`，不代表控制台一定仍是免费方案。未确认现有资源前，不创建数据库、不升级付费实例。

## 本地使用

安装依赖后，首次创建管理员：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m flask --app app create-admin admin01 --name 管理员
```

命令会私下询问密码和确认密码；至少12个字符。不要把密码放在命令参数或代码中。

```bash
.venv/bin/python app.py
```

打开 http://localhost:5001/login 。管理员登录后可在 `/admin/accounts` 创建学生账号、停用／启用账号及重置密码。学生首次登录或密码重置后需先修改初始密码。未登录不能访问页面和推荐接口；学生不能管理账号。

本地账号默认保存在 `instance/accounts.sqlite3`，签名密钥保存在 `instance/session-secret`；均已加入忽略规则，不提交 Git。没有预设管理员密码，也没有开放注册入口。

## Render 上线前配置

Render 环境需配置下列环境变量，必须先配置再部署登录版本：

| 名称 | 内容 |
| --- | --- |
| `SECRET_KEY` | 随机、长期固定的签名密钥，建议至少32个字符；使用控制台的 Generate 功能或安全生成器。 |
| `DATABASE_URL` | 现有持久 PostgreSQL 数据库连接串，优先使用同区域的 Internal Database URL。 |

如果已经使用付费服务及持久磁盘，也可用 `AUTH_DB_PATH=/var/data/accounts.sqlite3` 替代 `DATABASE_URL`，其中 `/var/data` 必须是真实的持久磁盘挂载目录。两者都配置时优先使用 `DATABASE_URL`。不能把账号库放在普通 Render 工作目录；服务重启或重新部署会清除该目录的写入。

首次数据库连接时会自动创建 `platform_users` 表，不改动录取历史数据。上线前需通过实际登录流程验证数据库可写；`/healthz` 是应用存活检查，不能替代账号数据库验证。

运行配置：

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT app:app
Health Check Path: /healthz
```

Render Shell 中首次创建管理员：

```bash
python -m flask --app app create-admin admin01 --name 管理员
```

若当前实例不支持 Shell，需要根据已有数据库和 Render 配置选择一次性管理命令的执行方式；不要添加公开的管理员初始化接口。没有数据库和管理员初始化条件前，不推送触发上线的更新。

## 域名绑定

1. 在该 Render 服务的 Settings → Custom Domains 中添加 `www.platform.xly-edu.com`。
2. 在 `xly-edu.com` 的 DNS 管理中新增：

| 类型 | 主机名（当区域为 xly-edu.com 时） | 目标 |
| --- | --- | --- |
| CNAME | `www.platform` | `eju-score-predictor.onrender.com` |

以 Render 控制台给出的实际目标和 DNS 提供商字段规则为准。某些控制台需填写完整主机名 `www.platform.xly-edu.com`。不要修改主站的 `@`、`www` 或邮件记录。

3. 在 Render 验证域名，等待 HTTPS 证书成功签发。
4. 测试域名上的登录、学生首次修改密码、账号管理和文／理科推荐。直接通过旧 `.onrender.com` 网址访问同样需要登录。

如需同时支持 `platform.xly-edu.com`，必须另外添加该域名和对应 DNS 记录；它与带 `www` 的地址不会自动视作同一个域名。

DNS 检查显示当前名称服务器是 `ns1.dns.ne.jp` 和 `ns2.dns.ne.jp`，可请官网管理员据此查找 Sakura 的管理入口。需要 DNS 编辑权限；无需把账户密码发送给开发者。

参考：[Render 自定义域名](https://render.com/docs/custom-domains)、[持久磁盘](https://render.com/docs/disks)、[免费方案限制](https://render.com/docs/free)。
