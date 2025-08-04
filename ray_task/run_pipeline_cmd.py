import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from ray_task.config import REPORT_PATH
from utils.tool import load_cfg
from pipeline import global_var

from utils.logger import Logger
logger = Logger.get_logger(f"ray-task")


def get_audio_manifest(audio_path):
    path_parts = Path(audio_path).parts
    podcast_name = path_parts[-2] if len(path_parts) > 1 else "UnknownPodcast"
    episode_name = os.path.splitext(os.path.basename(audio_path))[0]
    return {
        "PodcastName": podcast_name,
        "EpisodeName": episode_name,
        "FilePath": audio_path
    }

@dataclass
class TaskCmdArgs:
    batch_size: int = 8
    compute_type: str = 'float16'
    whisper_arch: str = 'medium'
    threads: int = 2


def run_audio_preprocess_pipeline(input_audio_path, output_dir, task_key):
    main_cfg = load_cfg("config.json")
    cli_args = TaskCmdArgs(batch_size=8, compute_type='float16', whisper_arch='medium', threads=2)
    logger.info(f"pipeline config={main_cfg} cli_args={cli_args}")
    global_var.init_pipeline_global(main_cfg, cli_args)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processed data will be saved in: {output_dir}")

    from pipeline.main_process import main_process
    manifest_entry = get_audio_manifest(input_audio_path)
    main_process(manifest_entry, output_dir, "processing_report.csv")
    
    logger.info("--- All files have been processed. ---")
    return "SUCCESS"