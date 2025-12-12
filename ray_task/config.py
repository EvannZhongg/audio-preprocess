import os

# ==============================================================================
# 硬件环境说明 (基于你的 V100 配置)
# ------------------------------------------------------------------------------
# 显卡: V100 (32GB 显存, 11 CPU) 
# 显卡: P40 (24GB 显存, 12 CPU) 
# 显卡: A10 (24GB 显存) 
# ==============================================================================

# 数据集配置
DATASET_NAME = "podcasts/xiaoyuzhou"
CONFIG_PATH = "./configs/config_for_v100.json" 


# DATASET_NAME = "audiobooks/有声小说1"
# CONFIG_PATH = "./configs/config_for_p40.json" 
# ------------------------------------------------------------------------------
# 核心调度参数优化
# ------------------------------------------------------------------------------

# [显存安全锁]
# 限制 Worker 内部同时处理 3 个音频
MAX_WORKERS = 3
# MAX_WORKERS = 2

# [Ray 资源预留]
# 每个 Ray Worker 占用 1 整张 GPU
GPU_PER_TASK = 1

# [CPU 资源预留]
# Ray 调度时为该 Worker 预留的 CPU 总数(节点请申请10核及以上)
CPU_PER_TASK_GPU = 11

# ------------------------------------------------------------------------------
# 内部线程分配 (总额 10 核的分配方案)
# 逻辑: MAX_WORKERS * (FFmpeg + Torch) <= CPU_PER_TASK_GPU
# ------------------------------------------------------------------------------

# [FFmpeg]
# 音频解码核心数
CPU_PER_TASK_CPU = 2
# MAX_WORKERS = 4

# [ONNX/Torch] 
# PyTorch/ONNX 分配核心数
TORCH_THREAD_NUM = 1

# ------------------------------------------------------------------------------
# 批处理与超时
# ------------------------------------------------------------------------------

# [超时安全线]
# 音频多线程解码超时(支持解码约3小时音频)
FFMPEG_TIME_OUT = 1200 
TIME_OUT = 6000

# [调度缓冲]
# 配合 MAX_WORKERS=3，单个worker每次处理 32 个音频
BATCH_SIZE = 32
LONG_AUDIO_BATCH_SIZE = 9 # 长音频包含的音频数量

# [任务队列上限]
MAX_POOL_SIZE = 60


# [OOM 熔断]
# 超过 3 小时的音频直接过滤
MAX_AUDIO_DURATION_SECONDS = 3 * 3600

# 路径配置
PODCAST_PATH = "/cfs/cfs-czb184s7/DATA/webdata"
OUTPUT_ROOT_DIR = "/cfs/cfs-czb184s7/PROCESSED_DATA/webdata"


OUTPUT_PATH = os.path.join(OUTPUT_ROOT_DIR, DATASET_NAME)
PODCAST_DATA_FILE = os.path.join(PODCAST_PATH, DATASET_NAME, "data_list.json")

TASK_RESULT_FILE = os.path.join(OUTPUT_PATH, "ray_task_result.json")
TASK_RESULT_BACKUP_FILE = os.path.join(OUTPUT_PATH, "ray_task_result_backup.json")
REPORT_PATH = os.path.join(OUTPUT_PATH, "processing_report.csv")
