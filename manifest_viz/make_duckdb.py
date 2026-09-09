#!/usr/bin/env python
"""在 DuckDB 库里建一个视图,指向 parquet(本地或 JuiceFS 挂载路径)。

Superset 通过 duckdb-engine 连接这个库文件查询。幂等,可反复跑;每来一类
数据集,带参数跑一次即可。

    python make_duckdb.py --db zh.duckdb --view manifest --parquet '/data/zh/*.parquet'

跑完后在 Superset UI 里接入(只需一次):
  1. Settings → Database Connections → + Database → 选 "Other"
  2. SQLAlchemy URI 填(路径是 Superset 进程能访问到的,容器里用挂载路径;
     加 read_only 避免和本脚本的写连接抢锁):
       duckdb:////data/zh.duckdb?access_mode=read_only
  3. Test Connection → Connect
  4. Datasets → + Dataset → 选该 Database / schema=main / 上面建的视图名 → 保存
  (往老库里新增视图时,连接已在,只需重复第 4 步加 dataset。)
"""
import argparse
import duckdb

p = argparse.ArgumentParser()
p.add_argument("--db", required=True, help="DuckDB 库文件路径") # 打开/创建这个 .duckdb 文件 和database同级
p.add_argument("--view", required=True, help="视图名") # 创建视图 和表同级
p.add_argument("--parquet", required=True, help="parquet glob,如 '/data/zh/*.parquet'")
args = p.parse_args()

con = duckdb.connect(args.db)
con.execute(
    f'CREATE OR REPLACE VIEW "{args.view}" AS '
    f"SELECT * FROM read_parquet('{args.parquet}', union_by_name=true)"
)
n = con.execute(f'SELECT count(*) FROM "{args.view}"').fetchone()[0]
con.close()
print(f"{args.db}: 视图 {args.view} -> {args.parquet}  ({n} 行)")
