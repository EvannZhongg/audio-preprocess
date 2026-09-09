# `status/` — pipeline_v3 运行进度监控

一个**只读**的常驻监控工具。每 5 分钟扫描一次 `pipeline_v3` 的输出目录，把进度报告写到
`status/status.log`，每小时向企业微信群推送一次精简版。

不写入 `--output` 目录，不修改任何现有代码；只在本目录下产生
`status.log` / `history.jsonl` / `scan_cache.json` / `state.json`。

```bash
# 常驻运行（推荐）
cd /path/to/audio-preprocess
nohup python status/monitor.py \
    --output   /path/to/pipeline_v3/output \
    --manifest /path/to/manifest \
    > /dev/null 2>&1 &

# 看一眼当前数字就退出
python status/monitor.py --output ... --manifest ... --once

# 测试企微 webhook 是否通
python status/monitor.py --output ... --manifest ... --once --push-now
```

日志本身已经落盘并打到 stdout，所以 `nohup` 的输出可以直接丢掉。
停止用 `kill <pid>`（SIGTERM），会在当前周期结束后干净退出。

---

## 输出格式

报告和企微消息共用同一个摘要块（`reporter.render_summary`），所以群里和日志里
不可能出现同一次扫描却对不上的数字。头部略有不同：企微多一行带标题的时间戳，
便于在聊天列表里一眼认出是什么消息。

企微推送：

```
[TTS 数据处理进度] 2026-08-24 15:04:21
文件: huyuan_ko/out/5

进度: 59.84%  (49,775.11h / 83,182.30h)
stage2 有效音频: 12,004.54h ， 有效率: 24.12%

已运行: 1d 19h 36m
预计剩余: 21h 18m
预计最终有效音频: 20,061.54h
```

`status.log`（摘要块之后还有完整的分段明细）：

```
========================================================================
huyuan_ko/out/5    @ 2026-08-24 15:04:21
========================================================================

进度: 59.84%  (49,775.11h / 83,182.30h)
...
```

路径取 `--output` 的**末尾 3 级**：`/root/jfs/itachi/huyuan_ko/out/5/` →
`huyuan_ko/out/5`。前面的挂载点每个任务都一样，只占宽度不带信息，末 3 级才是区分
并行任务的部分。日志的 `-- PROGRESS` 段里仍保留完整路径备查。

摘要块之后（仅日志）是完整的分段明细：`PROGRESS` / `STAGE 1` / `STAGE 2` /
`RATE & ETA` / `SCAN`。摘要之前会插入告警行（写入停滞、manifest 缺失或对不上、
有 parquet 读不了），因为只看第一屏的人也必须知道 pipeline 已经死了。

`有效率` 即 `end-to-end yield` = stage2 有效时长 / stage1 已处理的原始时长。
任何算不出来的量都渲染成 `N/A` 而不是 `0` —— 这里的 `0` 会被当成真实测量值。

---

## 报告里每个数字的口径

所有"有效性"判定都直接调用 `qc/loaders.py`，不重新实现。该文件的 docstring 明确
它是各阶段 valid 语义的 single source of truth；监控如果自己定义一套，数字就会和
QC 报告、和 pipeline 实际行为悄悄分叉，这比没有监控更糟。

| 报告字段 | 含义 | 代码依据 |
|---|---|---|
| `raw audio total` | 本次任务要处理的原始音频总时长 = manifest 全量 `duration` 之和 | `source_scan/manifest.py:31-36` |
| `stage1 processed` | **已处理的原始音频时长**，见下方专门说明 | `pipeline_v2_ray/segments.py:23-41` |
| `progress` | `stage1 processed / raw audio total` | |
| `valid segments` | stage1 切出的真实段数（`error IS NULL`） | `qc.loaders.stage1_valid_mask` |
| `VALID OUTPUT AUDIO` | **stage2 有效音频时长**：无 error 且五个 `dropped_by_*` 均未置位 | `qc.loaders.stage2_kept_mask` |
| `keep rate (by dur)` | stage2 有效时长 / stage2 见到的总时长 | |
| `end-to-end yield` | stage2 有效时长 / stage1 已处理的原始时长 | |
| `dropped by ...` | 五类丢弃原因各自的段数与时长，**不互斥**（一段可同时触发多条） | `qc.loaders.stage2_drop_mask` |
| `retriable ASR failures` | `asr_access_failed`，下轮会自动重跑，故单列 | `pipeline_v2_ray/stage2_segments.py:89-98` |

### 为什么"已处理小时数"要回查 manifest

stage1 的 parquet 里**没有原始文件时长**这一列：

- `chunk_duration` 是导出的 chunk wav 时长
- `seg_duration` 是切出来的段时长

两者都已经被降噪、VAD、质量过滤层层削减，远小于原始音频。所以正确算法是：

1. 读所有 `segments_part-*.parquet` 的 `source` 列并去重 —— `source` 就是 manifest 的
   `relative_path`（`pipeline_v2_ray/segments.py:26`）
2. 拿这个集合去 manifest 求 `duration` 之和

这与 driver 自己的统计口径完全一致（`pipeline_v3/driver.py:378` 累加的 `item.duration`
正是来自 manifest），所以监控数字和 pipeline 日志能对上。

一个文件的 chunk 可能横跨两个 part（driver 每 10 万行 flush 一次，不管 chunk 边界），
所以去重是**全局**的，不是按 part 的。

### 失败文件算作"已处理"

stage1 失败的文件会写**恰好一行**占位行，只有 `source`/`shard`/`pipeline_version`/`error`
有值（`pipeline_v2_ray/segments.py:44-53`），且 `resume_state()` 把这些文件视为 done ——
重跑不会再试。它们确实消耗了 GPU 时间、也确实不会再被处理，所以计入已处理；
否则进度会永久性偏低、ETA 永远收敛不了。

报告里 `files done` 后面的 `(failed N)` 就是这部分。

---

## 耗时与剩余时间是怎么估的

`pipeline_v3/driver.py:_log_progress` 已经在算吞吐和 ETA，但那是**进程内**的：
pipeline 一重启就归零，而且按 `(stage, shard)` 分开打，回答不了"整个任务跑了多久"。
本工具补这个洞，用两类独立证据。

### 两个"已运行时长"，永远分开展示

| 字段 | 来源 | 偏差 |
|---|---|---|
| `pipeline running for` | 输出树里**最早**的 parquet mtime | 略偏小（首次 flush 晚于启动，driver 满 10 万行或每 300s 才 flush，`driver.py:61`）；**续跑场景会严重偏大**，因为最老的 part 是上一轮留下的 |
| `monitor observing for` | 本工具第一次采样至今 | 只反映它自己观测到的窗口，说不了它启动之前的事 |

两个都给，读日志的人自己判断哪个可信。

### 两个速率

每轮往 `history.jsonl` 追加一个采样点，速率就是"已处理秒数"对墙钟时间的斜率：

- **`rate (last Xm)`** —— 滑动窗口（默认 60 分钟），**这是主口径**。集群规模在跑的过程中
  会变（`ActorPool.reconcile`、stage1 输入耗尽后 `retire_all`），全程均速反应太慢，
  没有参考价值。
- **`rate (whole run)`** —— 全程均速，作为慢变参照。两个并排，能直接看出在加速还是在衰减。

速率用**首尾端点差分**而非最小二乘拟合：这个序列是单调递增的累计量，端点斜率就等于
窗口内的平均速率，且不会被单个异常采样点带偏。

**首轮没有历史**时退化为 `已处理 / (now - 最早mtime)`，此时报告里会标成
`rate (mtime est.)` 并注明是粗估 —— 它继承了上面 mtime 的偏差。

### 停滞检测

超过 `--stale-minutes`（默认 15 分钟，远大于 driver 的 300s flush 间隔）没有任何
parquet 写入，就判定 pipeline 可能已停或卡死。此时：

- 报告和企微消息开头给出显著告警
- **ETA 不再输出** —— 用一个已经归零的速率去外推是主动误导
- 速率照旧显示（它是历史事实）

### 最终产出预估

`projected final valid = raw audio total × (stage2 有效时长 / stage1 已处理时长)`

注意这个值**在早期会偏低**：stage2 的交接队列没有背压（见 `pipeline_v3/driver.py` 头部
注释），允许下游任意落后，所以刚开始 stage2 还没追上 stage1 时，这个比率是从 0 往上爬的。

---

## CLI

| 参数 | 默认 | 说明 |
|---|---|---|
| `--output` | *必填* | `main_v3_ray.py` 的 `--output` |
| `--manifest` | – | `build_manifest.py` 的产出（目录或单个 parquet）。**不给就没有进度和 ETA**，因为原始音频时长只存在于 manifest |
| `--interval` | `300` | 扫描间隔（秒） |
| `--push-interval` | `3600` | 企微推送最小间隔（秒）；`0` 关闭推送 |
| `--window-minutes` | `60` | 主速率的滑动窗口；调小反应更快，调大更平滑 |
| `--stale-minutes` | `15` | 多久没有写入判定为停滞（必须大于 driver 的 300s flush 间隔） |
| `--workers` | `16` | 读 parquet 的线程数 |
| `--once` | off | 扫一次就退出 |
| `--push-now` | off | 首轮立刻推送，不等 `--push-interval`；配合 `--once` 可测 webhook |
| `--log-file` | `status/status.log` | |
| `--log-max-bytes` / `--log-backups` | `32MB` / `5` | 日志滚动 |
| `--state-dir` | `status/` | 状态文件位置 |

## 企业微信

复用 `utils/msg_bot.py` 里已有的群 webhook。该函数没有超时、没有异常捕获、
不检查返回码，所以本工具外面包了一层：导入是**懒加载**的（统计路径在没装
`requests` 的机器上也能跑），推送失败只记 `ERROR`，绝不影响主循环。

推送时机按 `now - last_push_ts >= push_interval` 判断，`last_push_ts` 持久化在
`state.json` 里 —— 用计数会漂移，不持久化则每次重启都会立刻重复推一条。

消息按 **字节** 截断到 2048（企微的限制是字节数，而内容以中文为主），且不会把
一个字符劈成两半。

## 性能

part 一旦以 `*.parquet` 出现就是不可变的（写盘是 temp-then-rename，
`segments.py:61-63`），所以按 `路径|mtime|大小` 做缓存是安全的。稳定态下每轮只需读
新增的那 1-2 个 part，其余全部命中缓存，报告里的
`parquet parts: N (read A, cached B)` 就是这个命中情况。

另外：

- 只读必要列（stage1 读 3 列、stage2 读 7 列），不读全表
- stage1 的 `source` 去重用 `pc.unique` 在 Arrow 内核里做，只把去重后的结果转成
  Python，一个 10 万行的 part 通常只产生几千个字符串
- 因此用**线程池**而非进程池：耗时部分会释放 GIL，且不需要把大 set 来回 pickle
- manifest 常驻为两列 Arrow Table，用 `pc.is_in` 求交。千万级文件如果用 Python dict
  会占好几个 GB，列式存储则紧凑得多

`*.parquet.tmp`（正在 rename 的）和 `_` 开头的目录（如 `_qc_reports`）都会被
`qc.layout.discover_shards` 自动排除。

## 健壮性

- 单个 part 读失败只计数并跳过（`qc.loaders.ParquetReadError`），报告里列出前几个路径
- 整轮扫描抛异常 → `logger.exception` 后照常 sleep 进入下一轮，进程不退出
- `history.jsonl` 只追加，崩溃最多损坏最后一行，读取时跳过
- `scan_cache.json` / `state.json` 全部 temp-then-rename 原子落盘；损坏则降级为空重算
- 若 stage1 的 `source` 和 manifest 的 `relative_path` 完全对不上（传错 manifest），
  报告会显式告警，而不是一直显示 0%

## 与 `qc/`、`tmp/` 的关系

- `qc/` 是**一次性全量深度质检**（含 GPU 重检）；本工具是**高频轻量进度看板**，
  只读 parquet、不碰模型。两者的有效性判定共用 `qc/loaders.py`，数字可以互相核对。
- `tmp/stat_stage2.py` 是 stage2 保留率的独立参照实现，口径与本工具一致
  （都源自同一个 keep 谓词），可用于人工验算。
