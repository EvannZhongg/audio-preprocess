# TTS 数据处理管线

这是一个端到端的音频数据处理管线，专门用于将"野生"音频数据转换为高质量的 TTS（文本转语音）训练数据集。该管线集成了降噪、说话人分离、语音活动检测（VAD）、自动语音识别（ASR）等多个步骤，最终输出按说话人分类的干净音频-文本对。

## 主要功能

- **音频标准化**：自动调整音频采样率、位深度、声道数和音量
- **人声背景声分离**：使用深度学习模型分离人声和背景音
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

在运行之前，请确保以下模型文件已下载并放置在 `ckpts/` 目录下（目前先暂存cfs/cfs-du3y2r4h/share/ckpts中，若无法获取私聊bobbsun）：

```
ckpts/
├── pretrained_eres2netv2.ckpt          # ERes2Net 说话人嵌入模型
├── UVR-MDX-NET-Inst_HQ_3.onnx         # 人声分离模型
└── sig_bak_ovr.onnx                    # DNSMOS 质量评分模型
```

## 配置文件

修改 `config.json` 以适应您的需求：

- `asr_provider`：选择 ASR 引擎（"whisper"、"funasr"、"paraformer"）
- `language.supported`：支持的语言列表
- `mos_filter.strategy`：质量过滤策略（"average" 或 "fixed"）
- `output_format`：输出格式（"default" 或 "libritts"）
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

## 输出格式

### 默认格式

### LibriTTS 格式

当 `config.json` 中设置 `"output_format": "libritts"` 时：

```
input_folder_processed/
├── audio1/
│   ├── SPK_abc123_00/
│   │   ├── SPK_abc123_00-00001.wav
│   │   ├── SPK_abc123_00-00001.normalized.txt
│   │   ├── SPK_abc123_00-00002.wav
│   │   ├── SPK_abc123_00-00002.normalized.txt
│   │   └── ...
│   └── SPK_abc123_01/
│       └── ...
└── audio2/
    └── ...
```

## 支持的音频格式

- MP3 (`.mp3`)
- WAV (`.wav`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- AAC (`.aac`)
- MP4 (`.mp4`)

## Coming Soon 🚀

- **Web 界面**：基于 `main_init.py` 的网页版体验界面，支持文件上传和实时进度显示
- **多 GPU 支持**：基于 `main_multi.py` 的多卡并行处理，大幅提升处理速度
