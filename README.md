# TTS 数据处理管线

这是一个端到端的音频数据处理管线，专门用于将"野生"音频数据（podcast、有声书、Web 音频等）转换为高质量、可直接用于训练的 TTS 数据集。管线集成了降噪、说话人分离、VAD、ASR、强制对齐、文本/音频质量评分、领域标注等多个步骤，最终输出按说话人切分的干净音频片段及结构化 JSON 元数据。

- **千万小时级处理能力**：基于 Ray 分布式框架，支持跨机器、多 GPU 水平扩展，可处理千万小时级音频数据集
- **多语种支持**：中文、英文、日文、韩文、法文、德文、俄文，每种语言独立校准阈值配置
- **灵活的运行模式**：单机（`main.py`，单/多进程、多 GPU 自动分配）、Ray 集群（`run_ray_task.py`）两种模式按需选择

集群部署方案详见 [Ray 分布式处理](#ray-分布式处理-)。

## 主要功能

### 音频处理
- **音频标准化**：自动调整采样率、位深度、声道数和音量
- **人声背景声分离/降噪**：支持 `UVR`（人声/背景声分离）和 `SMRU`（高性能降噪）。**推荐使用 SMRU**
- **说话人分离 (Diarization)**：基于 pyannote 的说话人日志，配合 ERes2Net 嵌入精修
- **VAD**：silero-vad 二次精切 + 同说话人合并 + 长段在 silero 内部停顿处再切

### 转录与对齐
- **ASR**：支持 5 种 ASR 引擎，可热切换：
  - `whisper` (faster-distil-whisper-large-v3)
  - `funasr` (SenseVoiceSmall，中/英)
  - `funasr_nano` (Fun-ASR-MLT-Nano-2512，多语种含俄语)
  - `paraformer` (中文专用)
  - `qwen3_asr` (Qwen3-ASR-1.7B 内部服务)
  - `gemini` (Google Gemini 2.5)
- **交叉验证 ASR**：可启用第二个 ASR 引擎做交叉验证，按 WER 阈值过滤
- **强制对齐 (Alignment)**：基于 WhisperX wav2vec2-CTC，输出 word-level CTC 置信度

### 质量评分（每个片段独立打分）

| 维度 | 字段 | 说明 |
|---|---|---|
| 音频质量 | `dnsmos` / `c50` / `snr` | DNSMOS 信号质量 + Brouhaha 信噪比/混响 |
| 语速 | `speaking_rate` | chars / voiced_duration（用 silero-vad 排除静音） |
| 文本质量 | `ppl` / `spell_score` / `llm_quality` / `semantic_completeness` / `tts_suitability` | Qwen2.5-0.5B PPL + 多语言拼写 + Qwen3-Omni LLM 三维评分 |
| 强制对齐 | `alignment_score` | wav2vec2 mean per-word CTC confidence |
| 异常静音 | `abnormal_silence_count` | 段内 ≥300ms 无标点解释的内部停顿数 |
| 领域标注 | `text_domain` / `acoustic_domain` / `speaker_domain` | Qwen3-Omni LLM file-level + speaker-level 多领域分类 |

每个评分模块都可以独立 `enable`、各自有阈值，失败软降级（填 -1 不丢段）。

### 多语言支持
中文、英文、法文、日文、韩文、德文、俄文。每种语言有独立校准的阈值配置。

### 输出格式
LibriTTS、自定义 JSON 元数据。

### 分布式
- 单机：`main.py`（单进程 / 多进程 / 多 GPU）
- 集群：`run_ray_task.py`（Ray 分布式，需配 `ray_task/config.py`）

## 安装依赖

```bash
conda create -y -n AudioPipeline python=3.10  # Docker 部署用 3.9.21
conda activate AudioPipeline

# 1. 安装 PyTorch（torch 不在 requirements.txt 中，需单独安装）
# CUDA 12.x 驱动（如 L20/L40/A100）使用 cu124：
pip install torch==2.5.0 torchaudio==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 安装其余依赖
pip install -r requirements.txt
```

> **注意**：torch 在 `requirements.txt` 中被注释掉（Dockerfile 里通过 conda 安装），本地机器需手动执行第 1 步。

主要依赖：`pyannote.audio`、`whisperx`、`funasr`、`faster-whisper`、`silero-vad`、`librosa`、`language-tool-python`、`transformers`、`ray`。

### 常见问题：cuDNN 缺失

**报错**：`libcudnn_ops_infer.so.8: cannot open shared object file`

**原因**：系统缺少 cuDNN 8 库。

**修复**：

```bash
# 安装 cuDNN
pip install nvidia-cudnn-cu12==8.9.7.29

# 临时生效（当前 session）
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
```

**永久生效**（每次激活 conda env 自动设置）：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH' \
    > $CONDA_PREFIX/etc/conda/activate.d/cudnn.sh
```

## 模型文件准备

模型文件放在 `audio-preprocess/ckpts/` 目录下

### 核心模型
```
ckpts/
├── pretrained_eres2netv2.ckpt          # ERes2Net 说话人嵌入
├── sig_bak_ovr.onnx                    # DNSMOS 质量评分
└── pyannote_config_v2.yaml             # pyannote diarization 配置
```

### 分离/降噪模型
**SMRU（推荐）**
```
ckpts/
├── a_merge_from_a06_labotf_v2.pt
├── denoise_derev_48k_SFI_E128.yaml
├── 2task_48k_lessmusic__addrir_5merged.pt
└── denoise_derev_48k_SFI.yaml
```

**UVR（备选）**
```
ckpts/
└── UVR-MDX-NET-Inst_HQ_3.onnx
```

### ASR / 文本质量 / 对齐模型（按需）
- `Qwen/Qwen2.5-0.5B`：PPL 评分用，缓存到 `~/.cache/huggingface/hub`
- `Systran/faster-distil-whisper-large-v3`：whisper ASR
- `iic/SenseVoiceSmall` / `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`：FunASR
- `FunAudioLLM/Fun-ASR-MLT-Nano-2512`：俄语等小语种 ASR
- WhisperX 各语种 wav2vec2 align 模型：自动从 HF 下载，可手动放 `~/.cache/huggingface/hub/whisperx-align`

## 配置文件

按场景/语言/GPU显存大小/性能选择：

| 配置 | 用途 |
|---|---|
| `configs/config_for_v100_for_{zh,en,fr,ja,ko,de,russian}.json` | V100 各语种独立校准 |
| `configs/config_for_a10_for_{en,ja}.json` | A10 英/日语优化版 |
| `configs/config_for_p40_for_zh.json` | P40 中文版 |
| `configs/config_for_l20_for_zh.json` | L20 中文版 |

### 关键配置项

#### 段切策略 `strategy_parameters`
| 字段 | 推荐值 | 说明 |
|---|---|---|
| `merge_gap` | 0.5 | 同说话人段间隔 < 0.5s 才合并 |
| `min_segment_length` | 2.0 | 段不足 2s 丢弃 |
| `max_segment_length` | 30.0 | 段超 30s 走 silero 切分；切不开则丢 |
| `inter_similarity_threshold` | 0.5 | 段内说话人嵌入一致性阈值 |
| `intra_similarity_threshold` | 0.68 | 跨段嵌入相似度合并阈值 |

#### 质量打分 + 过滤
所有评分模块的开关和阈值：
```json
{
    "speaking_rate": { "enable": true, "thresholds": { "speaking_rate_min": 4.0, "speaking_rate_max": 16.0 } },
    "alignment":     { "enable": true, "thresholds": { "alignment_score_min": 0.55 } },
    "silence_filter":{ "enable": true, "min_silence_ms": 300, "thresholds": { "abnormal_silence_count_max": 1 } },
    "text_quality":  { "enable": true, "ppl": {...}, "spell": {...}, "llm": {...}, "thresholds": {...} },
    "domain_annotation": { "enable": true, "text_domain": {...}, "acoustic_domain": {...}, "speaker_domain": {...} }
}
```

阈值设为 `null` 表示**只打分不过滤**（写入 metadata，但不丢段）。

#### ASR 与交叉验证
```json
"asr_provider": "qwen3_asr",
"validation_asr_provider": "funasr_nano",
"asr_validation": { "enable": true, "wer_threshold": 0.6, "language": "ru" }
```

#### 通用
- `huggingface_token`：HF 访问令牌（pyannote 必需）
- `language.supported`：支持的 ISO 语言代码列表
- `language.multilingual`：是否多语言模式

## 基本使用

### 快速开始

1. 准备音频文件：例如放在 `ORIGINAL_DATA/` 等文件夹中
2. 按音频语种选择对应配置，运行处理管线：
   ```bash
   # 中文音频
   python main.py --input_folder_path ORIGINAL_DATA/ximalaya/ --config_path configs/config_for_v100_for_zh.json --output_folder ./PROCESSED_DATA/ximalaya
   # 英文音频
   python main.py --input_folder_path ORIGINAL_DATA/spotify/ --config_path configs/config_for_v100_for_en.json --output_folder ./PROCESSED_DATA/spotify
   # 其他语种类推，配置文件列表见「配置文件」章节
   ```
3. 查看结果：输出目录由 `--output_folder` 指定

### 命令行参数

```bash
python main.py [OPTIONS]
```

主要参数：
- `--input_folder_path`：输入文件夹（默认 `examples/`）
- `--config_path`：配置文件（默认 `config.json`）
- `--num_workers`：进程数（V100 建议 2，devcloud 建议 1）

## 多 GPU 并行处理

`main.py` 通过 `--num_workers` 启动多进程池，`init_pipeline_global` 会按 worker 编号自动把每个 worker 绑到一张 GPU 上（`worker_id % 可用GPU数`），所以无需额外命令即可跑满多卡。

```bash
# 4 卡机器，每卡跑 2 个 worker（共 8 进程）
python main.py \
    --input_folder_path /path/to/audio \
    --config_path configs/config_for_v100_for_zh.json \
    --output_folder /path/to/out \
    --num_workers 8

# 只用 GPU 1、2（屏蔽 0、3），每卡 2 个 worker
CUDA_VISIBLE_DEVICES=1,2 python main.py \
    --input_folder_path /path/to/audio \
    --config_path configs/config_for_v100_for_zh.json \
    --output_folder /path/to/out \
    --num_workers 4
```

要点：
- **进程数 = GPU 数 × 每卡 worker 数**：例如 4 卡 × 每卡 2 worker → `--num_workers 8`
- **屏蔽指定 GPU**：用环境变量 `CUDA_VISIBLE_DEVICES` 控制可见 GPU（`main.py` 本身没有 `--disabled_gpu_ids` 参数）
- **每卡 worker 数**：V100 上 batch_size 8 时建议每卡 1~2 个 worker，显存不够就降到 1
- **CPU 线程**：`--threads` 控制每个 worker 的 torch 线程数，默认 2；总线程数 ≈ `num_workers × threads`
- **断点续传**：`--report_path processing_report.csv`，重跑时会自动跳过已处理文件

## 输入输出

### 输入
```
input_folder_path/
├── dir1/
│   ├── sub_dir1/
│   │   ├── 1-00001.wav
│   │   └── 1-00002.wav
│   └── sub_dir2/
│       └── 2-00002.wav
└── dir2/
    └── 1-00001.wav
```

### 输出
保留输入目录结构。每个音频文件得到一个同名子目录，里面有 `.wav`（合并后的干净音频）和 `.json`（元数据）：
```
output_folder/
├── dir1/
│   ├── sub_dir1/
│   │   ├── 1-00001/
│   │   │   ├── 1-00001.wav
│   │   │   └── 1-00001.json
│   │   └── 1-00002/
│   │       ├── 1-00002.wav
│   │       └── 1-00002.json
│   └── sub_dir2/
│       └── 2-00002/
│           ├── 2-00002.wav
│           └── 2-00002.json
└── dir2/
    └── 1-00001/
        ├── 1-00001.wav
        └── 1-00001.json
```

### 输出 JSON Schema

每个 `.json` 文件包含原音频的所有有效片段：
```json
{
  "pipeline_version": "1.0",
  "origin": {
    "raw_audio_path": "/path/to/source.mp3",
    "sample_rate": 24000,
    "duration": 1234.56
  },
  "sentences": [
    {
      "utt_id": "abc123",
      "spk_id": "SPK_a1b2c3d4_00",
      "speaker_min_similarity": "0.6543",
      "time_range": { "duration": 5.32, "start": 12.41, "end": 17.73 },

      "transcription_info": {
        "text": "Привет, как дела?",
        "val_text": "...",            // 交叉验证 ASR 输出
        "norm_text": "...",            // 文本归一化结果
        "wer": "0.0521",
        "avg_char_duration": "0.1923",
        "speaking_rate": "5.3214",     // chars per voiced second
        "alignment_score": "0.8721",   // wav2vec2 CTC confidence [0,1]
        "abnormal_silence_count": 0    // 段内无解释长停顿数
      },

      "audio_quality_info": {
        "dnsmos": "3.2156",
        "c50": "42.3120",
        "snr": "28.7654"
      },

      "text_quality_info": {
        "ppl": "215.32",
        "spell_score": "0.9321",
        "llm_quality": "7.5",
        "semantic_completeness": "8.0",
        "tts_suitability": "7.0"
      },

      "domain_info": {
        "text_domain":     { "domain": "podcast", "scenario": "interview", "style": "conversational" },
        "acoustic_domain": { "environment": "studio", "background": "clean", "quality": "high" },
        "speaker_domain":  { "gender": "male", "age_group": "adult", "accent": "standard" }
      }
    }
  ]
}
```

> 任何字段值为 `-1` / `-1.0` / `"unknown"` 表示该评分未启用或失败软降级。**这种段不会被自动过滤**，留给下游消费方决定。

## Pipeline 处理步骤

```
Step 0:    Standardization (采样率 / 声道 / 音量)
Step 1:    Source Separation (SMRU / UVR)
Step 2:    Speaker Diarization (pyannote)
Step 3:    VAD (silero-vad)
Step 3.5:  VAD Refinement (ERes2Net 嵌入精修)
Step 4:    Post-process VAD (按 merge_gap / min/max_segment_length 合并切分)
Step 5:    ASR + 交叉验证
Step 5.5:  Domain Annotation (Qwen3-Omni LLM)
Step 5.7:  Speaking Rate Scoring & Filter
Step 5.75: Abnormal Silence Detection & Filter
Step 5.8:  Audio-Text Alignment (WhisperX) Scoring & Filter
Step 6:    Audio Metrics Prediction & Filter (DNSMOS / C50 / SNR / 时长 / 字符)
Step 6.5:  Text Quality Scoring & Filter (PPL / Spell / LLM)
Step 7:    Export (合并干净音频 + 写 JSON)
```

每个 Filter 步骤都会更新 `processing_stats`，最终在日志里打印保留/丢弃的段数和时长统计。

## 支持的音频格式

`.mp3` / `.wav` / `.flac` / `.m4a` / `.aac` / `.mp4` / `.ogg` / `.webm`

## Ray 分布式处理 🚀

面向**千万小时级**音频数据的集群化处理方案。通过 Ray 将任务分发到多台机器、多张 GPU，实现水平扩展。

### 适用场景

- 单机 `main.py` 处理速度不够时（通常 >10 万小时）
- 需要跨机器并行处理多个数据集
- 数据存放在共享 CFS 上，多节点可同时读写

### 架构

```
Head Node（调度）
    └── run_ray_task.py  ←  任务分发 + 进度追踪
Worker Node × N（计算）
    └── 每节点若干 GPU，每 GPU 跑 MAX_WORKERS 个 pipeline 进程
```

### 第一步：配置

编辑 `ray_task/config.py`：

```python
DATASET_NAME = "ximalaya_all"               # 数据集名称
CONFIG_PATH  = "./configs/config_for_v100_for_zh.json"  # 处理配置

PODCAST_PATH    = "/cfs/.../DATA"           # 输入音频根目录（CFS）
OUTPUT_ROOT_DIR = "/cfs/.../PROCESSED_DATA" # 输出根目录（CFS）

MAX_WORKERS   = 2     # 每张 GPU 并发 pipeline 数
GPU_PER_TASK  = 0.5   # 每个 worker 占用 GPU 份额（0.5 = 2个worker共享1张卡）
CPU_PER_TASK_GPU = 5  # 每个 worker 占用 CPU 核数
```

### 第二步：启动集群

在**每台机器**上执行（自动判断是头节点还是 worker 节点）：

```bash
bash scripts/start_ray.sh auto
```

也可以手动分步：
```bash
# 头节点
bash scripts/start_ray.sh start-head

# 每台 worker 节点
bash scripts/start_ray.sh start-node
```

### 第三步：提交任务

在**头节点**上运行：

```bash
nohup python run_ray_task.py > run.log 2>&1 &
tail -f run.log
```

任务进度和结果保存在 `ray_task/config.py` 中配置的 `TASK_RESULT_FILE` 和 `REPORT_PATH`。

## 数据策略原则

1. **软降级而非丢弃**：任何评分模块失败（模型加载失败、API 超时、处理异常）一律填 `-1` 写入 metadata，**不影响其他评分，不丢段**。
2. **打分与过滤分离**：所有阈值都是可选的；设 `null` 即只打分不过滤，下游可灵活决定。
3. **宁可丢段也不强切**：超过 `max_segment_length` 的长段，如果 silero-vad 找不到内部停顿，丢弃整段而非在词中间硬切。
4. **每语种独立校准**：阈值（PPL、speaking_rate、alignment_score 等）按语种 ASR 输出特性单独配置。
5. **失败可观测**：每步的丢弃数、丢弃时长写入 `processing_report.csv` 便于事后分析。

## PipelineV2：local_adapter_v2 优化模式

PipelineV2 通过原生配置
`configs/config_pipeline_v2_diarizen_tts_clean_v2.json` 使用
`BUT-FIT/diarizen-wavlm-large-s80-md-v2` 和对应分段优化，同时保持现有
导出数据结构不变。原始配置行为默认不变，且不读取
`local_adapter_v2/configs/tts_clean_v2.json`。

安装、GPU 运行和原始/优化 A/B 测试方式见：

```text
pipeline_v2/LOCAL_ADAPTER_V2.md
```
