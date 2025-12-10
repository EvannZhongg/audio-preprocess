import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pipeline import global_var
from pipeline.main_process import main_process
from ray_task.config import CPU_PER_TASK_CPU, MAX_WORKERS
from utils.tool import load_cfg

_pipeline_initialized = False
_init_lock = threading.Lock()


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
    threads: int = CPU_PER_TASK_CPU
    
    
def _ensure_pipeline_initialized(config_path):
    global _pipeline_initialized

    if not _pipeline_initialized:
        with _init_lock:
            if not _pipeline_initialized:
                try:
                    main_cfg = load_cfg(config_path)
                    cli_args = TaskCmdArgs() 
                    global_var.init_pipeline_global(main_cfg, cli_args)
                    _pipeline_initialized = True
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize pipeline: {e}") from e


def run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_dir):
    try:
        _ensure_pipeline_initialized(config_path)

        logger = getattr(global_var.PipelineParam, 'logger', None)
        if logger is None:
            import logging
            logger = logging.getLogger("FallbackLogger")
            
        logger.info(f"Processing batch on device: {getattr(global_var.PipelineParam, 'device', 'unknown')}")
        os.makedirs(output_dir, exist_ok=True)
        
    except Exception:
        print(f"CRITICAL: Global pipeline initialization failed:\n{traceback.format_exc()}")
        return "FAILURE_INIT"

    def process_single_task(task):
        input_audio_path = task.get("audio_path")
        if not input_audio_path:
            logger.warning(f"Skipping task with missing audio_path: {task}")
            return False

        task_key = get_task_key(task)

        try:
            manifest_entry = get_audio_manifest(input_audio_path, prefix_path)
            
            safe_filename = os.path.basename(input_audio_path).replace(' ', '_')
            report_file = f"report_{safe_filename}.csv"

            main_process(manifest_entry, output_dir, report_file)
            
            logger.info(f"Sub-task {task_key} processed successfully.")
            task["pipeline_status"] = "SUCCESS"
            return True

        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error processing sub-task {task_key} in Batch: {error_msg}")
            
            task["pipeline_status"] = "FAILURE"
            task["error_msg"] = str(e) 
            return False

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_single_task, task): task 
            for task in task_batch
        }
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result() 
            except Exception as e:
                logger.error(f"Critical exception in thread for task {get_task_key(task)}: {e}")
                task["pipeline_status"] = "FAILURE"
                task["error_msg"] = str(e)

    logger.info("--- All sub-tasks in Batch have been processed. ---")
    return "SUCCESS"
