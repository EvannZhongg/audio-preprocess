#!/bin/bash

# 配置
# audiobooks
dataset_name="audiobooks/有声小说7"
# dataset_name="podcasts/xiaoyuzhou"
ROOT_DIR="/apdcephfs/tts_common/DATA/webdata"
INPUT="${ROOT_DIR}/${dataset_name}"
OUTPUT="${INPUT}/data_list.json"


MAPPING="/apdcephfs/tts_common/DATA/webdata=/cfs/cfs-czb184s7/DATA/webdata"

SCRIPT_PATH="./analyze_webdata.py"

mkdir -p "$(dirname "$OUTPUT" 2>/dev/null || echo .)"

python3 $SCRIPT_PATH \
  --path "$INPUT" \
  --base_dir "${ROOT_DIR}" \
  --output "$OUTPUT" \
  --path-mapping "$MAPPING" \
  --log-level INFO
