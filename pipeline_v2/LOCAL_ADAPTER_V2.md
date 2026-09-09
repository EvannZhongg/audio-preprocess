# PipelineV2 local_adapter_v2 优化模式

PipelineV2 现在保留两种可直接对比的处理模式：

- **原始模式**：继续使用现有 `configs/config*.json`。默认说话人模型仍为
  `pyannote/speaker-diarization-3.1`，分段防线默认关闭，行为保持不变。
- **优化模式**：传入 PipelineV2 原生配置
  `configs/config_pipeline_v2_diarizen_tts_clean_v2.json`。该模式使用
  `BUT-FIT/diarizen-wavlm-large-s80-md-v2`，并启用 local_adapter_v2 的
  人声分离、异说话人防线、grace 后最短长度复检及质量阈值。

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

## 一次运行两组对比

```bash
python scripts/compare_pipeline_v2_modes.py \
  --input /path/to/audio \
  --output-root /path/to/ab-output \
  --baseline-config configs/config.json \
  --optimized-config configs/config_pipeline_v2_diarizen_tts_clean_v2.json \
  --overwrite
```

输出分别位于 `ab-output/baseline` 和 `ab-output/optimized`。原始模式可能
仍需要现有配置声明的模型缓存；优化模式还需要上述 DiariZen 独立环境。
脚本还会写入 `ab-output/comparison_summary.json`，汇总段数、保留时长、
最短/最长段长、DNSMOS/C50/SNR 中位数，并校验两组 JSON 字段结构一致。
