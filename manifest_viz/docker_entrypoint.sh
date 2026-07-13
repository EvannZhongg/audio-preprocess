#!/usr/bin/env bash
# 容器入口:初始化 Superset 元数据库(MySQL)、建 admin、用 gunicorn 起服务。
#
# 元数据库走 MySQL(持久化),所以 SUPERSET_METADATA_URI 必须传入,例如:
#   docker run -e SUPERSET_METADATA_URI='mysql+pymysql://user:pass@host:3306/superset?charset=utf8mb4' ...
#
# 其它可选环境变量:
#   SUPERSET_SECRET_KEY        会话/加密密钥(生产务必传入,勿用镜像内默认值）
#   SUPERSET_ADMIN_USER/PASSWORD/EMAIL   admin 账号(默认 admin/admin）
#   PORT                       监听端口(默认 8088）
#   GUNICORN_WORKERS           worker 数(默认 4）
set -euo pipefail

: "${PORT:=8088}"
: "${GUNICORN_WORKERS:=4}"
: "${SUPERSET_ADMIN_USER:=admin}"
: "${SUPERSET_ADMIN_PASSWORD:=admin}"
: "${SUPERSET_ADMIN_EMAIL:=admin@example.com}"

# 元数据库必须是外部 MySQL——容器无持久卷,SQLite 会随容器销毁而丢。
if [[ -z "${SUPERSET_METADATA_URI:-}" ]]; then
  echo "ERROR: 必须设置 SUPERSET_METADATA_URI 指向 MySQL,例如:" >&2
  echo "  -e SUPERSET_METADATA_URI='mysql+pymysql://user:pass@host:3306/superset?charset=utf8mb4'" >&2
  exit 1
fi


curl -sSL https://d.juicefs.com/install | sh -
juicefs mount -d redis://:flowtts@123@9.223.107.179:6379/1 ~/jfs


echo "[entrypoint] 升级/初始化元数据库 ..."
superset db upgrade

echo "[entrypoint] 创建 admin(已存在则跳过)..."
superset fab create-admin \
  --username "$SUPERSET_ADMIN_USER" \
  --firstname Admin --lastname User \
  --email "$SUPERSET_ADMIN_EMAIL" \
  --password "$SUPERSET_ADMIN_PASSWORD" || true

echo "[entrypoint] 初始化角色/权限 ..."
superset init

echo "[entrypoint] 启动 gunicorn (port=$PORT, workers=$GUNICORN_WORKERS) ..."
# gunicorn 是生产级 WSGI 服务器(比 `superset run` 的 Flask dev server 稳)。
# gthread + threads 适配 Superset 的 IO 密集型请求。
exec gunicorn \
  -w "$GUNICORN_WORKERS" \
  -k gthread \
  --threads 20 \
  --timeout 120 \
  -b "0.0.0.0:${PORT}" \
  "superset.app:create_app()"
