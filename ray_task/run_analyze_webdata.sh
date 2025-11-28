#!/bin/bash

# 配置
# audiobooks
dataset_name="有声小说2"
INPUT="/apdcephfs/tts_common/DATA/webdata/audiobooks/${dataset_name}"
OUTPUT="${INPUT}/data_list.json"


MAPPING="/apdcephfs/tts_common=/cfs/cfs-czb184s7"

SCRIPT_PATH="./analyze_webdata.py"

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT" 2>/dev/null || echo .)"

# 运行分析（使用 pydub 版本）
python3 $SCRIPT_PATH \
  --path "$INPUT" \
  --output "$OUTPUT" \
  --path-mapping "$MAPPING" \
  --log-level INFO
