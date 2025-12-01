import json
import os
import re
import shutil
from pathlib import Path

from ray_task.config import OUTPUT_ROOT_DIR, PODCAST_DATA_FILE
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
                    "relative_path": podcast['relative_path'],
                    "audio_path": podcast['audio_path'],
                    "audio_duration_second": podcast['audio_duration_second']
                }
                for podcast in podcast_data['podcast_data']
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
            relative_path = tasks["processing"][k]["relative_path"]
            audio_path = tasks["processing"][k]["audio_path"]
            fid = re.sub(r"['\"\s]", "", Path(audio_path).stem)
            processing_dir = f"{OUTPUT_ROOT_DIR}/{relative_path}/{fid}"
            if os.path.exists(processing_dir):
                logger.debug(f"please rm -rf {processing_dir}")
                shutil.rmtree(processing_dir)
        tasks["processing"] = {}

    return tasks
