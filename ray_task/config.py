# PODCAST_PATH = "/cfs-du3y2r4h/binaryzhang/example"
# OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data" # 注意OUTPUT一定不要填错，代码中存在删除脏数据逻辑

# 小宇宙数据配置
# PODCAST_PATH = "/cfs-du3y2r4h/podcast-data/download"
# OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data"

# PODCAST_DATA_FILE = f"{OUTPUT_PATH}/podcast_data.json"
# TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
# TASK_RESULT_BACKUP_FILE = f"{OUTPUT_PATH}/ray_task_result_backup.json"
# REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"

# audiobooks/有声小说2
DATASET_NAME = "audiobooks/有声小说2"
CONFIG_PATH = "./configs/config_for_ximalaya_audiobooks.json"
PODCAST_PATH = f"/cfs/cfs-czb184s7/DATA/webdata/{DATASET_NAME}"
OUTPUT_PATH = f"/cfs/cfs-czb184s7/PROCESSED_DATA/webdata/{DATASET_NAME}"





PODCAST_DATA_FILE = f"{PODCAST_PATH}/data_list.json"
TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
TASK_RESULT_BACKUP_FILE = f"{OUTPUT_PATH}/ray_task_result_backup.json"
REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"
