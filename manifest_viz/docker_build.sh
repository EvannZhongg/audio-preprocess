#!/usr/bin/env bash
# 构建并推送 manifest_viz Superset 服务镜像。
# 构建上下文就是 manifest_viz/ 目录(Dockerfile、代码、requirements 都在这)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REGISTRY="csighub.tencentyun.com/avchat"
IMAGE="manifest-viz"
version=$(date "+%Y%m%d%H%M%S")
version="manifest_viz_${version}"
TAG="${REGISTRY}/${IMAGE}:${version}"
echo "构建镜像: ${TAG}"

# 需要登录私有仓库(账号密码按需填,或事先 docker login 好)。
# docker login --username=<user> https://csighub.tencentyun.com --password=<pass>

docker build --network=host -t "${TAG}" ./
docker push "${TAG}"

echo "完成: ${TAG}"
echo ""
echo "运行示例(元数据库走 MySQL,数据源 parquet 运行时挂载):"
echo "  docker run -d --name manifest-viz -p 8088:8088 \\"
echo "    -e SUPERSET_SECRET_KEY='<随机密钥>' \\"
echo "    -e SUPERSET_METADATA_URI='mysql+pymysql://root:<密码>@30.170.139.156:3306/superset?charset=utf8mb4' \\"
echo "    -v /juicefs/manifest:/data/manifest:ro \\"
echo "    ${TAG}"
