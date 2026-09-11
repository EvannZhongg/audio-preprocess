# PipelineV2 local_adapter_v2 优化模式

PipelineV2 现在保留两种可直接对比的处理模式：

- **原始模式**：继续使用现有 `configs/config*.json`。默认说话人模型仍为
  `pyannote/speaker-diarization-3.1`，分段防线默认关闭，行为保持不变。
- **优化模式**：传入 PipelineV2 原生配置
  `configs/config_pipeline_v2_diarizen_tts_clean_v2.json`。该模式使用
  `BUT-FIT/diarizen-wavlm-large-s80-md-v2`，并启用 local_adapter_v2 的
  人声分离、异说话人防线、grace 后最短长度复检及质量阈值。分离器使用
  SMRU（与原始配置一致）；DiariZen 以常驻子进程运行且
  `segmentation_step` 放大到 0.25，详见下方「性能开关」。

两种模式都经过同一个 `pipeline_v2.steps.export.Exporter`，所以最终
WAV、JSON sidecar、`SegmentRecord` 和 Ray parquet 的数据结构不变。

## DiariZen 运行环境

DiariZen 依赖定制版 pyannote，需使用独立 Python 环境。可通过以下命令安装：

```bash
chmod +x scripts/install_pipeline_v2_diarizen.sh
scripts/install_pipeline_v2_diarizen.sh
```

安装脚本默认使用 `python3.11`（DiariZen 要求 Python 3.10+）；若路径不同，
可通过 `PYTHON=/path/to/python3.11` 指定。
为保证与本次验证版本一致，脚本默认固定 DiariZen revision
`844f5555b0a98acd0931511fc641a8c5b8ba92c7`；需要升级时可通过
`DIARIZEN_REV=<commit>` 显式覆盖。

优化配置默认寻找独立运行环境：

```text
.venv-diarizen/bin/python
```

若部署路径不同，可在
`configs/config_pipeline_v2_diarizen_tts_clean_v2.json` 的 `diarizen`
字段中覆盖：

```json
{
  "python_executable": "/path/to/diarizen/python",
  "model_dir_cache": "/path/to/diarizen-wavlm-large-s80-md-v2",
  "embedding_model_path": "/path/to/wespeaker-voxceleb-resnet34-LM.bin",
  "cache_dir": "/path/to/huggingface/cache"
}
```

默认配置不绑定 `local_adapter_v2/models`。未指定有效本地模型路径时，
runner 会按以下两个模型 ID 从 Hugging Face 缓存或 Hub 加载：

```text
BUT-FIT/diarizen-wavlm-large-s80-md-v2
pyannote/wespeaker-voxceleb-resnet34-LM
```

DiariZen 子进程继承 PipelineV2 的 `device_name`。普通 PipelineV2 和 Ray
actor 默认都使用 `cuda:0`；Ray 场景下该编号对应 actor 通过
`CUDA_VISIBLE_DEVICES` 获得的当前 GPU，因此 DiariZen 与原流程共享同一张
卡及同一个 GPU 临界区。

## 性能开关

优化配置默认开启两项提速措施，均可在 `diarizen` 字段中调整：

```json
{
  "segmentation_step": 0.25,
  "resident": true
}
```

### `segmentation_step`（默认 0.25）

滑窗推进步长，以 `seg_duration`（本模型 16 秒）的比例表示。模型自带
`config.toml` 的默认值是 `0.1`，即 16 秒窗每 1.6 秒推进一次——**每帧音频被
WavLM-Large 前向 10 次**。这部分冗余与短促串音的时间分辨率（约 0.02 秒）无关，
可以安全放大。

CPU 实测（180 秒音频）：

| `segmentation_step` | 有效步长 | 推理耗时 | 段数 | 短段(≤1.5s) | 短应答召回 |
|---|---|---|---|---|---|
| 0.1（模型默认） | 1.6s | 190.5s | 91 | 56 | 基准 |
| **0.25（当前默认）** | **4.0s** | **76.3s** | 93 | 58 | 54/56 |
| 0.5 | 8.0s | 51.6s | 97 | 63 | 56/56 |

提速约 2.5 倍，短应答检测能力不降。**若该字段缺省则完全沿用模型自带的
`config.toml`**，所以既有配置行为不变。

另有 `batch_size` 与 `apply_median_filtering` 两个同类开关，同样缺省即继承
模型配置。注意 `apply_median_filtering` 是 11 帧（约 0.22 秒）中值滤波，关闭后
会多出若干亚秒级碎片，但它们本就会被 `min_segment_length` 丢弃。

> ⚠️ `segmentation_step` 放大会减少检出的异说话人重叠，进而让下游
> `drop_segments_with_foreign_speech` / `split_around_foreign` 防线的丢弃变少。
> 换素材时请同时观察 `chunk_done` 的 `retain_seg_percent` 与说话人数，
> **不要只看单个短片段的结果就调整默认值**。

### `resident`（默认 true）

DiariZen 作为常驻子进程运行，模型只加载一次，通过 stdin/stdout 的行分隔 JSON
协议接收请求。这与 pyannote 后端的资源模型一致——后者本来就在
`Diarizer.__init__` 里加载一次并常驻整个进程生命周期。

实测（20 秒 chunk，CPU）：首次请求 9.4 秒（含 7.4 秒启动），后续请求 1.7 秒。
在 GPU 上推理被压缩后，这笔固定开销（含 CUDA context 初始化）占比更高。

常驻还消除两个结构性问题：每个 chunk 重新解析 HuggingFace 路径（与曾导致
brouhaha 挂起 27 分钟同类的风险），以及进程 spawn 期间霸占 GPU 锁阻塞同一
actor 的其他文件。

子进程在 **stdin 关闭时自行退出**。这是唯一可靠的防孤儿机制：生产环境的
`ray.kill(no_restart=True)` 和 `mp.Pool.terminate()` 都不执行父进程的任何用户
代码，而内核保证父进程无论以何种方式死亡（含 SIGKILL）都会关闭管道写端。
子进程的 stderr 写入 `logs/diarizen_worker.<pid>.log`。

设 `"resident": false` 可退回"每个 chunk 一个新进程"的原行为，用于 A/B 或回滚。

> ⚠️ 常驻模式下每个 `Diarizer` 持有一个常驻子进程。`main_v2.py --num-workers N`
> 会产生 N 个 `Diarizer`（且都指向同一张卡），显存占用随之乘 N。在显存紧张的
> 卡上（V100 16GB 还需与 SMRU / eres2net / brouhaha / DNSMOS 共存）部署前
> 请先实测单进程常驻显存。

### 日志字段

`dia_time_cost` 增加了几个字段用于归因：

- `infer_ms`——**仅子进程内的推理耗时**。此前该字段错误地包含了进程启动、
  `import torch`、HF 解析和模型加载，导致真实推理耗时无法测量。
- `spawn_ms`——仅首次请求非零，直接体现常驻带来的收益。
- `write_wav_ms` / `ipc_ms`——临时 WAV 写入与传输开销。


## 单独运行

原始 PipelineV2：

```bash
python main_v2.py \
  --config configs/config.json \
  --input /path/to/audio \
  --output /path/to/output/baseline \
  --num-workers 1
```

优化后的 PipelineV2：

```bash
python main_v2.py \
  --config configs/config_pipeline_v2_diarizen_tts_clean_v2.json \
  --input /path/to/audio \
  --output /path/to/output/optimized \
  --num-workers 1
```

## 一次运行三组对比

三个配置刻意做了变量隔离，每一步只改一件事：

| 模式 | 配置 | 分离模型 | 分段防线 |
|---|---|---|---|
| `baseline` | `configs/config_pipeline_v2_baseline_ab.json` | pyannote 3.1 | 关 |
| `modelswap` | `configs/config_pipeline_v2_diarizen_swap_ab.json` | **DiariZen** | 关 |
| `optimized` | `configs/config_pipeline_v2_diarizen_tts_clean_v2.json` | DiariZen | **开** |

于是 `baseline → modelswap` 的差异**只归因于分离模型**，
`modelswap → optimized` 的差异**只归因于分段流程**。

```bash
bash scripts/run_ab_compare.sh audios_test ab_out
# 只跑其中两组：
bash scripts/run_ab_compare.sh audios_test ab_out "baseline modelswap"
```

三个配置在 chunking、分离器（均为 SMRU）、brouhaha 路径、`min_audio_seconds`
上完全一致——**不要拿 `configs/config.json` 直接当基线**，它的
`min_audio_seconds: 600` 会让短于 10 分钟的文件被整个跳过
（`standardization.py:77`），基线一段都不输出，看起来像优化版完胜。

### 读结果的关键一点

对比脚本报两个污染指标，**只有第二个能跨模型比较**：

- `contaminated segments`——导出段之间的跨说话人时间重叠。它**只能看到该模式
  自己的分离模型标注出来的重叠**。一个压根没检出应答的模型在这里得 0 分，
  看起来完美，实际上把应答整个吞了。
- `swallowed backchannels`——以**所有模式检出结果的并集**为参照，统计有多少短
  应答落在本模式某个归属他人的段内部。这才是诚实的跨模型数字。

这正是本次引入 DiariZen 的原因：它是唯一能识别这类短促串音的模型，而
"看不见的污染"在旧指标下是隐形的。
