# TTS 数据处理管线

这是一个端到端的音频数据处理管线，专门用于将"野生"音频数据转换为高质量的 TTS（文本转语音）训练数据集。该管线集成了降噪、说话人分离、语音活动检测（VAD）、自动语音识别（ASR）等多个步骤，最终输出按说话人分类的干净音频-文本对。

## 主要功能

- **音频标准化**：自动调整音频采样率、位深度、声道数和音量
- **人声背景声分离/降噪**：支持 `UVR` (人声/背景声分离) 和 `SMRU` (高性能降噪) 两种模型。**推荐使用 SMRU** 以获得更好的效果。
- **说话人分离**：基于 pyannote 的说话人日志技术
- **语音活动检测**：精细化的语音片段检测和优化
- **自动语音识别**：支持多种 ASR 引擎（Whisper、FunASR、Paraformer）
- **质量筛选**：基于 DNSMOS 的音频质量评分和过滤
- **多语言支持**：支持中文、英文、法文、日文、韩文、德文等
- **灵活输出格式**：支持 LibriTTS 格式或自定义格式

## 安装依赖

```bash
conda create -y -n AudioPipeline python=3.9 
conda activate AudioPipeline

bash env.sh
```

## 模型文件准备

在运行之前，请确保以下模型文件已下载并放置在 `audio-preprocess/ckpts/` 目录下（目前先暂存cfs/cfs-du3y2r4h/share/ckpts中，若无法获取私聊bobbsun）：

### 核心模型
```
ckpts/
├── pretrained_eres2netv2.ckpt          # ERes2Net 说话人嵌入模型
└── sig_bak_ovr.onnx                    # DNSMOS 质量评分模型
```

### 分离/降噪模型 (根据配置选择)

#### SMRU (推荐)
这是推荐的高性能降噪模型，能有效去除背景噪音同时保留人声的自然度。
```
ckpts/
├── a_merge_from_a06_labotf_v2.pt      # SMRU 模型文件 (推荐)
├── denoise_derev_48k_SFI_E128.yaml    # 对应的 SMRU 配置文件
├── 2task_48k_lessmusic__addrir_5merged.pt # 另一个 SMRU 模型文件
└── denoise_derev_48k_SFI.yaml         # 对应的 SMRU 配置文件
```

#### UVR
这是传统的人声/背景声分离模型。
```
ckpts/
└── UVR-MDX-NET-Inst_HQ_3.onnx         # UVR 模型文件
```

## 配置文件

修改 `config.json` 以适应您的需求：

- `separate.provider`：选择分离/降噪模型。`"smru"` (推荐) 或 `"uvr"`。
- `separate.smru.batch_size`: (当使用smru时) 用于推理的批次大小，可根据显存调整以优化速度。
- `asr_provider`：选择 ASR 引擎（"whisper"、"funasr"、"paraformer"）
- `language.supported`：支持的语言列表
- `mos_filter.strategy`：质量过滤策略（"average" 或 "fixed"）
- `huggingface_token`：Hugging Face 访问令牌（用于说话人分离模型）

## 基本使用

### 快速开始

1. **准备音频文件**：将待处理的音频文件放在某个文件夹中（如 `examples/`）
2. **运行处理管线**：

   ```bash
   python main.py --input_folder_path examples/
   ```
3. **查看结果**：处理完成后，结果将保存在 `examples_processed/` 目录中

### 命令行参数

```bash
python main.py [OPTIONS]
```

#### 主要参数：

- `--input_folder_path`：输入音频文件夹路径（默认：`examples/`）
- `--config_path`：配置文件路径（默认：`config.json`）
- `--num_workers`：进程数（默认：`2`，目前v100建议为2，devcloud建议为1）

### 使用示例

```bash
# 基本使用
python main.py --input_folder_path /path/to/audio/files
```

## 多 GPU 并行处理

对于拥有多张 GPU 的用户，可以使用 `main_multi.py` 脚本来显著加速处理流程。该脚本会自动将待处理的音频文件平均分配给所有可用的 GPU，并在每张卡上并行运行多个工作进程。

### 使用示例

```bash
# 使用所有可用的 GPU，在每张卡上运行 2 个工作进程
python main_multi.py \
    --input_folder_path /path/to/your/audio/files \
    --output_folder /path/to/your/processed_data \
    --num_workers_per_gpu 2

# 假设有4张GPU (0, 1, 2, 3)，禁用 0 号和 3 号卡，只在 1 号和 2 号卡上运行
python main_multi.py \
    --input_folder_path /path/to/your/audio/files \
    --output_folder /path/to/your/processed_data \
    --num_workers_per_gpu 2 \
    --disabled_gpu_ids "0,3"
```

### 主要参数

- `--input_folder_path`: 输入音频文件夹路径。
- `--manifest_path`: (可选) 指定一个 CSV 清单文件，优先于 `--input_folder_path`。
- `--output_folder`: 处理结果的输出根目录。
- `--num_workers_per_gpu`: 指定在**每张** GPU 上启动的工作进程数量（默认：`2`）。
- `--disabled_gpu_ids`: (可选) 需要禁用的 GPU ID 列表，以逗号分隔（例如, `"0,3"`）。

## 输入输出格式

通常来说，我们需要传入一个包含复合格式要求音频文件的文件夹路径，如：
```
input_folder_path/
├── dir1/
│   ├── sub_dir1/
│   │   ├── 1-00001.wav
│   │   ├── 1-00002.wav
│   │   └── ...
│   └── sub_dir1/
│       └── 2-00002.wav
└── dir2/
    └── 1-00001.wav
```
对于每个解析的合法音频文件, `pipiline`处理完成后会生成一个同名的meta.json已经降噪处理后的音频文件，输出结果保存在`--output_folder`中，目录结构和输入目录结构保持一致，如：
```
output_folder/
├── dir1/
│   ├── sub_dir1/
│   │   ├── 1-00001
│   │   |   ├── 1-00001.wav
│   │   |   ├── 1-00001.json
│   │   ├── 1-00002
│   │   |   ├── 1-00002.wav
│   │   |   ├── 1-00002.json
│   │   └── ...
│   └── sub_dir1/
│       └── 2-00002
│   │   |   ├── 2-00002.wav
│   │   |   ├── 2-00002.json
└── dir2/
    └── 1-00001
    │   │   ├── 1-00001.wav
│   │   |   ├── 1-00001.json
```

## 支持的音频格式

- MP3 (`.mp3`)
- WAV (`.wav`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- AAC (`.aac`)
- MP4 (`.mp4`)

## ray分布式处理 🚀
- 使用scripts/start_ray.sh可以启动头节点和worker节点
- run_ray_task.py在头节点运行，输入和输出路径在ray_task/config.py
- 方案设计：https://iwiki.woa.com/p/4015720564
