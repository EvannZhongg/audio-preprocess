

import os
import traceback
from dataclasses import dataclass
from pathlib import Path

from pipeline import global_var
from pipeline.main_process import main_process
from utils.logger import Logger
from utils.tool import load_cfg

logger = Logger.get_logger(f"ray-task")


def get_task_key(task):
    return task["relative_path"]

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


def run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_dir):
    try:
        main_cfg = load_cfg(config_path)
        cli_args = TaskCmdArgs(batch_size=8, compute_type='float16', threads=4)
        logger.info(f"Pipeline config={main_cfg} cli_args={cli_args}")
        global_var.init_pipeline_global(main_cfg, cli_args)
    
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processed data will be saved in: {output_dir}")
        
    except Exception:
        logger.error(f"Global pipeline initialization failed: {traceback.format_exc()}")
        return "FAILURE_INIT"


    for task in task_batch:
        input_audio_path = task["audio_path"]
        task_key = get_task_key(task)

        try:
            manifest_entry = get_audio_manifest(input_audio_path, prefix_path)
            main_process(manifest_entry, output_dir, "processing_report.csv")
            
            logger.info(f"Sub-task {task_key} processed successfully.")
            task["pipeline_status"] = "SUCCESS" 

        except Exception as e:
            logger.error(f"Error processing sub-task {task_key} in Batch: {traceback.format_exc()}")
            task["pipeline_status"] = "FAILURE" 
            
    logger.info("--- All sub-tasks in Batch have been processed. ---")
    return "SUCCESS"
