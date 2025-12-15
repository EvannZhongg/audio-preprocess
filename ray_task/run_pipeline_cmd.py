import os
import shutil
import sys

LARGE_TEMP_PATH = "/home/oicq/audio_preprocess/TEMP" 

try:
    os.makedirs(LARGE_TEMP_PATH, exist_ok=True)
    os.environ["LARGE_TEMP_DIR"] = LARGE_TEMP_PATH
    # 覆盖系统默认临时目录，防止 soundfile/ffmpeg 写爆 /tmp
    os.environ["TMPDIR"] = LARGE_TEMP_PATH
    os.environ["TEMP"] = LARGE_TEMP_PATH
    os.environ["TMP"] = LARGE_TEMP_PATH
except Exception as e:
    print(f"Failed to set large temp dir: {e}")

import gc
import logging
import random
import threading
import time
import traceback
from dataclasses import dataclass

import torch

from pipeline import global_var
from pipeline.main_process import main_process
from ray_task.config import TORCH_THREAD_NUM
from utils.tool import load_cfg

logger = logging.getLogger(__name__)

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
    threads: int = TORCH_THREAD_NUM
    
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


def calculate_smart_jitter(file_path):
    """
    根据文件大小计算合理的抖动时间。
    避免对短音频等待过久，同时确保长音频能有效错峰。
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > 200:
            return random.uniform(15, 40) 
        
        elif size_mb > 50:
            return random.uniform(5, 15)
            
        else:
            return random.uniform(1, 4)
            
    except Exception:
        return random.uniform(1, 3)

def run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_dir):
    try:
        _ensure_pipeline_initialized(config_path)
        global_logger = getattr(global_var.PipelineParam, 'logger', logger)
        global_logger.info(f"Processing batch on device: {getattr(global_var.PipelineParam, 'device', 'unknown')}")
        os.makedirs(output_dir, exist_ok=True)
        
    except Exception:
        print(f"CRITICAL: Global pipeline initialization failed:\n{traceback.format_exc()}")
        return "FAILURE_INIT"

    #  启动时的随机抖动 (防止并发冲击 CFS)
    if task_batch:
        first_audio = task_batch[0].get("audio_path")
        if first_audio and os.path.exists(first_audio):
            jitter_sec = calculate_smart_jitter(first_audio)
            global_logger.info(f"⏳ Smart Jitter: Detected file {os.path.basename(first_audio)}, sleeping {jitter_sec:.2f}s...")
            time.sleep(jitter_sec)
    
    success_count = 0
    total_count = len(task_batch)
    global_logger.info(f"🚀 Starting Batch Processing: {total_count} files (Sequential Mode)")

    for i, task in enumerate(task_batch):
        input_audio_path = task.get("audio_path")
        task_key = get_task_key(task)
        
        if not input_audio_path:
            global_logger.warning(f"Skipping task with missing audio_path: {task}")
            continue

        global_logger.info(f"[{i+1}/{total_count}] Processing: {task_key}")

        try:
            manifest_entry = get_audio_manifest(input_audio_path, prefix_path)
            safe_filename = os.path.basename(input_audio_path).replace(' ', '_')
            report_file = f"report_{safe_filename}.csv"

            # 核心处理流程
            main_process(manifest_entry, output_dir, report_file)
            
            global_logger.info(f"✅ Sub-task {task_key} SUCCESS.")
            task["pipeline_status"] = "SUCCESS"
            success_count += 1

        except Exception as e:
            error_msg = traceback.format_exc()
            global_logger.error(f"❌ Error processing sub-task {task_key}: {error_msg}")
            task["pipeline_status"] = "FAILURE"
            task["error_msg"] = str(e)
            
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            time.sleep(random.uniform(0.1, 0.5))

    global_logger.info(f"--- Batch Completed. Success: {success_count}/{total_count} ---")
    return "SUCCESS"
