"""Superset 配置 —— 团队共享的 manifest parquet 可视化(BI / dashboard）。

数据源:DuckDB 库 manifest_viz/manifest.duckdb,内含指向 ft_local/*.parquet
的视图(manifest / paths / top)。Superset 通过 duckdb-engine(SQLAlchemy 方言)
连接,查询下推到 DuckDB,parquet 明细不进浏览器。

元数据库(Superset 自身的 dashboard / 用户 / 权限)默认用同目录下的 SQLite
文件;团队正式共享可改成 Postgres(见 README)。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# 允许用环境变量覆盖(部署到别处时无需改代码)。
SECRET_KEY = os.environ.get(
    "SUPERSET_SECRET_KEY",
    "vjGm45Uv_FSzsPnFZcacdPCbhfssV5zwt9lbh5q9ZMtFcAHpc1emtl7l",
)

# Superset 自身的元数据库(不是被分析的数据)。默认 SQLite 文件,零外部依赖。
# 正式团队共享请设 SUPERSET_METADATA_URI 指向 Postgres。
SQLALCHEMY_DATABASE_URI = os.environ.get(
    "SUPERSET_METADATA_URI",
    f"sqlite:///{os.path.join(_HERE, 'superset.db')}",
)

# 允许在 SQL Lab 里跑 DDL / CTAS(方便临时建视图);团队只读可关掉。
FEATURE_FLAGS = {
    "DASHBOARD_RBAC": True,
    "EMBEDDED_SUPERSET": False,
}

# 单机多人访问够用;并发大时可调。
SUPERSET_WEBSERVER_TIMEOUT = 120
ROW_LIMIT = 5000            # 图表默认取数上限
SQL_MAX_ROW = 100000        # SQL Lab 单次返回上限

# 关掉遥测。
SQLLAB_CTAS_NO_LIMIT = True
