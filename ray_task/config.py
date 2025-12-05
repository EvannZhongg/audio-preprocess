# PODCAST_PATH = "/cfs-du3y2r4h/binaryzhang/example"
# OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data" # 注意OUTPUT一定不要填错，代码中存在删除脏数据逻辑

# 小宇宙数据配置
# PODCAST_PATH = "/cfs-du3y2r4h/podcast-data/download"
# OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data"

# PODCAST_DATA_FILE = f"{OUTPUT_PATH}/podcast_data.json"
# TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
# TASK_RESULT_BACKUP_FILE = f"{OUTPUT_PATH}/ray_task_result_backup.json"
# REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"


# audiobooks/有声小说1
# DATASET_NAME = "audiobooks/有声小说1"

DATASET_NAME = "podcasts/xiaoyuzhou"
CONFIG_PATH = "./configs/config_for_v100.json"
# CPU_PER_TASK_GPU = 5  # 每个 GPU 任务占用 5 个 CPU (11 CPU / 2 任务 ≈ 5)
# GPU_PER_TASK = 0.5    # 每个 GPU 任务占用 0.5 个 GPU
# CPU_PER_TASK_CPU = 4  # 如果只跑 CPU 任务，保持原有的 4 个 CPU 占用








PODCAST_PATH = "/cfs/cfs-czb184s7/DATA/webdata"
OUTPUT_ROOT_DIR="/cfs/cfs-czb184s7/PROCESSED_DATA/webdata"
OUTPUT_PATH = f"{OUTPUT_ROOT_DIR}/{DATASET_NAME}"
PODCAST_DATA_FILE = f"{PODCAST_PATH}/{DATASET_NAME}/data_list.json"


TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
TASK_RESULT_BACKUP_FILE = f"{OUTPUT_PATH}/ray_task_result_backup.json"
REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"
