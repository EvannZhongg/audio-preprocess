

import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import ray

from pipeline import global_var
from pipeline.main_process import main_process
from ray_task.config import CPU_PER_TASK_GPU, GPU_PER_TASK
from utils.logger import Logger
from utils.tool import load_cfg

_pipeline_initialized = False


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
    
    
def _ensure_pipeline_initialized(config_path):
    global _pipeline_initialized
    if not _pipeline_initialized:
        try:
            main_cfg = load_cfg(config_path)
            cli_args = TaskCmdArgs(...)
            global_var.init_pipeline_global(main_cfg, cli_args)
            _pipeline_initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {e}") from e


@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, max_retries=0)
def run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_dir):
    try:
        _ensure_pipeline_initialized(config_path)
        logger = global_var.PipelineParam.logger
        logger.info(f"Processing batch on device: {global_var.PipelineParam.device}")
        
        os.makedirs(output_dir, exist_ok=True)
        
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
