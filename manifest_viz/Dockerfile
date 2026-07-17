# manifest_viz Superset 服务镜像。
# 轻量 python-slim 基础镜像:Superset 是纯 Python,不需要 CUDA / conda。
FROM python:3.11-slim

LABEL maintainer="binaryzhang" \
      description="Superset + DuckDB 的 manifest parquet 可视化服务"

# 少量系统依赖:build-essential 供极少数需现场编译的 pip 包;curl 供健康检查。
# 装完清理 apt 缓存,保持镜像精简。
RUN apt-get update && \
    apt-get install -y fuse && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/manifest_viz

# 先只拷依赖清单,利用 Docker 层缓存:requirements 不变时不重装。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷服务代码(配置、启动脚本、建看板脚本等)。
# 数据文件(manifest.duckdb / superset.db / 初始化标记)由 .dockerignore 排除,
# 见下方说明:元数据库走 MySQL,数据源在运行时挂载。
COPY . .

# Superset 读取配置的入口。
ENV SUPERSET_CONFIG_PATH=/app/manifest_viz/superset_config.py \
    FLASK_APP=superset \
    PORT=8088 \
    PYTHONUNBUFFERED=1

EXPOSE 8088

# 容器启动:初始化元数据库(MySQL)+ 建 admin + 起 gunicorn 服务。
ENTRYPOINT ["/app/manifest_viz/docker_entrypoint.sh"]
# ENTRYPOINT ["/bin/bash"]
