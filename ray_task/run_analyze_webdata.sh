#!/bin/bash

# 配置
# audiobooks
# dataset_name="audiobooks/有声小说1"
dataset_name="podcasts/xiaoyuzhou"
ROOT_DIR="/apdcephfs/tts_common/DATA/webdata"
INPUT="${ROOT_DIR}/${dataset_name}"
OUTPUT="${INPUT}/data_list.json"


MAPPING="/apdcephfs/tts_common/DATA/webdata=/cfs/cfs-czb184s7/DATA/webdata"

# 要跳过的二级目录（用逗号分隔）
# 默认跳过: _processed,.temp,temp,.git,__pycache__
# 可以追加自定义目录，如: SKIP_DIRS="_processed,.temp,temp,.git,__pycache__,已处理,测试数据"
SKIP_DIRS="_processed,.temp,temp,.git,__pycache__"

SCRIPT_PATH="./analyze_webdata.py"

mkdir -p "$(dirname "$OUTPUT" 2>/dev/null || echo .)"

python3 $SCRIPT_PATH \
  --path "$INPUT" \
  --base_dir "${ROOT_DIR}" \
  --output "$OUTPUT" \
  --path-mapping "$MAPPING" \
  --skip-dirs "$SKIP_DIRS" \
  --log-level INFO
