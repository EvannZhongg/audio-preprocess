import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from ray_task.config import REPORT_PATH
from utils.tool import load_cfg
from pipeline import global_var

from utils.logger import Logger
logger = Logger.get_logger(f"ray-task")


def get_audio_manifest(audio_path, base_dir):   
    relative_path = os.path.relpath(os.path.dirname(audio_path), base_dir)
    return {
        "RelativePath": relative_path,
        "FilePath": audio_path
    }

@dataclass
class TaskCmdArgs:
    batch_size: int = 8
    compute_type: str = 'float16'
    whisper_arch: str = 'medium'
    threads: int = 4


def run_audio_preprocess_pipeline(config_path, input_audio_path, prefix_path, output_dir, task_key):
    main_cfg = load_cfg(config_path)
    cli_args = TaskCmdArgs(batch_size=8, compute_type='float16', threads=4)
    logger.info(f"pipeline config={main_cfg} cli_args={cli_args}")
    global_var.init_pipeline_global(main_cfg, cli_args)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processed data will be saved in: {output_dir}")

    from pipeline.main_process import main_process
    manifest_entry = get_audio_manifest(input_audio_path, prefix_path)
    main_process(manifest_entry, output_dir, "processing_report.csv")
    
    logger.info("--- All files have been processed. ---")
    return "SUCCESS"
