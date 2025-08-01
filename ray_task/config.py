PODCAST_PATH = "/cfs-du3y2r4h/binaryzhang/example"
OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data" # 注意OUTPUT一定不要填错，代码中存在删除脏数据逻辑

# PODCAST_PATH = "/cfs-du3y2r4h/podcast-data/download"
# OUTPUT_PATH = "/cfs-du3y2r4h/binaryzhang/podcast-processed-data"

PODCAST_DATA_FILE = f"{OUTPUT_PATH}/podcast_data.json"
TASK_RESULT_FILE = f"{OUTPUT_PATH}/ray_task_result.json"
REPORT_PATH = f"{OUTPUT_PATH}/processing_report.csv"
