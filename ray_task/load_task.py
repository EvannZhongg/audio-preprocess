import json
import shutil
import os

from ray_task.config import PODCAST_DATA_FILE, OUTPUT_PATH

from utils.logger import Logger
logger = Logger.get_logger()

def load_tasks(file_path):
    tasks = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

    if not tasks:
        with open(PODCAST_DATA_FILE, 'r', encoding='utf-8') as f:
            podcast_data = json.load(f)
            tasks['todo'] = [
                {
                    "podcast_name": podcast['podcast_name'],
                    "episode_name": episode['episode_name'],
                    "audio_duration_second": episode['audio_duration_second']
                }
                for podcast in podcast_data['podcast_data']
                for episode in podcast['episode_data']
            ]
            tasks['total_num'] = len(tasks['todo'])
            tasks['total_hour'] = sum(ep['audio_duration_second'] for ep in tasks['todo']) / 3600
            tasks['processing'] = {}

            tasks['complete'] = []
            tasks['complete_num'] = 0
            tasks['complete_total_hour'] = 0
            tasks['failed'] = []
            tasks['failed_num'] = 0
            tasks['failed_total_hour'] = 0

    if len(tasks["processing"]) > 0:
        for k in tasks["processing"]:
            podcast_name = tasks["processing"][k]["podcast_name"]
            episode_name = os.path.splitext(os.path.basename(tasks["processing"][k]["episode_name"]))[0]
            processing_dir = f"{OUTPUT_PATH}/{podcast_name}/{episode_name}"
            if os.path.exists(processing_dir):
                logger.debug(f"please rm -rf {processing_dir}")
                shutil.rmtree(processing_dir)
        tasks["processing"] = {}

    return tasks