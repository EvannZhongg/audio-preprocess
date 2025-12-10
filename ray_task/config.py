# T4: 16G, 4cpu, P40: 24G, 12cpu, A10: 24G, 12cpu, V100: 32, 11cpu

# audiobooks/有声小说1
# DATASET_NAME = "audiobooks/有声小说1"
# CONFIG_PATH = "./configs/config_for_p40.json" 
# MAX_WORKERS = 2
# GPU_PER_TASK = 1
# TORCH_THREAD_NUM = 3  #  ONNX/Torch资源调度
# CPU_PER_TASK_CPU = 2  #  音频解码核心数
# CPU_PER_TASK_GPU = 12
# TIME_OUT = 300
# MAX_POOL_SIZE = 50
# BATCH_SIZE = 20

DATASET_NAME = "podcasts/xiaoyuzhou"
CONFIG_PATH = "./configs/config_for_v100.json"
MAX_WORKERS = 2
GPU_PER_TASK = 1
TORCH_THREAD_NUM = 3  #  ONNX/Torch资源调度
CPU_PER_TASK_CPU = 2  #  音频解码核心数
CPU_PER_TASK_GPU = 11
TIME_OUT = 300
MAX_POOL_SIZE = 50
BATCH_SIZE = 20

MAX_AUDIO_DURATION_SECONDS = 3 * 3600  # 超过3小时音频直接过滤







PODCAST_PATH = "/cfs/cfs-czb184s7/DATA/webdata"
OUTPUT_ROOT_DIR="/cfs/cfs-czb184s7/PROCESSED_DATA/webdata"
OUTPUT_PATH = f"{OUTPUT_ROOT_DIR}/{DATASET_NAME}"
PODCAST_DATA_FILE = f"{PODCAST_PATH}/{DATASET_NAME}/data_list.json"


TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
TASK_RESULT_BACKUP_FILE = f"{OUTPUT_PATH}/ray_task_result_backup.json"
REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"
