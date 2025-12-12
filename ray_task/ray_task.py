import json
import os
import random
import shutil
import time
import traceback
from collections import deque

import ray

from utils import msg_bot
from utils.logger import Logger

logger = Logger.get_logger(f"ray-task")

from ray_task.config import (BATCH_SIZE, CONFIG_PATH, CPU_PER_TASK_GPU,
                             DATASET_NAME, GPU_PER_TASK, LONG_AUDIO_BATCH_SIZE,
                             LONG_AUDIO_THRESHOLD, MAX_AUDIO_DURATION_SECONDS,
                             MAX_POOL_SIZE, OUTPUT_PATH, OUTPUT_ROOT_DIR,
                             PODCAST_PATH, TASK_RESULT_BACKUP_FILE,
                             TASK_RESULT_FILE)
from ray_task.load_task import load_tasks
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline

# ------------------------------------------------------------------
# --- Global Settings ---
# ------------------------------------------------------------------

SAVE_INTERVAL_SECONDS = 3600

# ------------------------------------------------------------------
# --- Progress Monitor ---
# ------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, dataset_name, report_interval=3*3600):
        self.dataset_name = dataset_name
        self.report_interval = report_interval
        self.last_send_time = 0
        
    def report(self, tasks, force_send=False):
        total_h = tasks.get("total_hour", 0)
        success_h = tasks.get("complete_total_hour", 0)
        failed_h = tasks.get("failed_total_hour", 0)
        processed_h = success_h + failed_h
        
        percent = (processed_h / total_h * 100) if total_h > 0 else 0
        
        is_finished = percent >= 99.99
        time_due = (time.time() - self.last_send_time) > self.report_interval
        
        if time_due or is_finished or force_send:
            self._send_bot(percent, success_h, failed_h, total_h)
            self.last_send_time = time.time()
        
        logger.debug(f"[{percent:5.2f}%] Done:{success_h:.1f}h | Fail:{failed_h:.1f}h")

    def _send_bot(self, percent, success_h, failed_h, total_h):
        msg = (f"{self.dataset_name}: Progress {percent:.2f}% | "
               f"Done: {int(success_h)}h | Failed: {int(failed_h)}h | Total: {total_h:.1f}h")
        try:
            msg_bot.send_msg(msg)
        except:
            pass

# ------------------------------------------------------------------
# --- Utilities ---
# ------------------------------------------------------------------

def save_tasks(tasks, main_file_path, backup_file_path=None):
    if isinstance(tasks.get('todo'), deque):
        tasks_dict = {k: v for k, v in tasks.items() if k != 'todo'}
        tasks_dict['todo'] = list(tasks['todo'])
    else:
        tasks_dict = tasks

    temp_file = main_file_path + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(tasks_dict, f, indent=2, ensure_ascii=False) 
        
    shutil.move(temp_file, main_file_path)
    if backup_file_path:
        shutil.copy(main_file_path, backup_file_path)

def get_task_key(task):
    return task["relative_path"]

# ------------------------------------------------------------------
# --- Worker Logic ---
# ------------------------------------------------------------------

def handle_task(config_path, task_batch, prefix_path, output_path):
    successful_tasks = []
    failed_tasks = []
    
    # [Jitter] 作用于 Worker 进程启动时
    time.sleep(random.uniform(2, 8))

    if not task_batch:
        return [], []

    try:
        # [Pipeline 调用]
        batch_status = run_audio_preprocess_pipeline(
            config_path, 
            task_batch, 
            prefix_path, 
            output_path
        )
        
        if batch_status != "SUCCESS":
            logger.error(f"Batch failed with status: {batch_status}")
            return [], task_batch 

    except Exception as e:
        logger.error(f"Handle Batch Exception: {traceback.format_exc()}")
        return [], task_batch

    for task in task_batch:
        task_status = task.get("pipeline_status", "UNKNOWN")
        if task_status == "SUCCESS":
            successful_tasks.append(task)
        else:
            logger.warning(f"Task {get_task_key(task)} failed: {task_status}")
            failed_tasks.append(task)
            
    return successful_tasks, failed_tasks

@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, scheduling_strategy="SPREAD", max_retries=0)
def handle_task_ray_gpu(config_path, task_batch, audio_prefix_path, output_path):
    return handle_task(config_path, task_batch, audio_prefix_path, output_path)

# ------------------------------------------------------------------
# --- Main Loop ---
# ------------------------------------------------------------------

def run():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tasks = load_tasks(TASK_RESULT_FILE)
    
    # [排序] 短任务优先。
    if isinstance(tasks['todo'], list):
        tasks['todo'].sort(key=lambda x: x['audio_duration_second'])
    
    task_queue = deque(tasks['todo'])
    tasks['todo'] = task_queue

    monitor = ProgressMonitor(DATASET_NAME)
    
    result_refs = []
    result_ref_map = {}
    last_save_time = time.time()

    logger.info(f"🚀 Cluster Job Started. Nodes: Batch Strategy: Mixed.")

    short_batch_buffer = [] 
    long_batch_buffer = []

    while task_queue or result_refs or short_batch_buffer or long_batch_buffer:
        
        # --- Submission ---
        while len(result_refs) < MAX_POOL_SIZE and task_queue:
            task = task_queue.popleft()
            task_key = get_task_key(task)
            duration = task["audio_duration_second"]

            if task_key in tasks["processing"]: continue
            if duration < 10: continue
            if duration > MAX_AUDIO_DURATION_SECONDS: 
                tasks['failed'].append(task)
                continue

            # [逻辑核心]
            is_long_task = duration > LONG_AUDIO_THRESHOLD

            if is_long_task:
                long_batch_buffer.append(task)
                if len(long_batch_buffer) >= LONG_AUDIO_BATCH_SIZE:
                    batch_to_send = long_batch_buffer[:LONG_AUDIO_BATCH_SIZE]
                    long_batch_buffer = long_batch_buffer[LONG_AUDIO_BATCH_SIZE:]
                    
                    for t in batch_to_send: tasks["processing"][get_task_key(t)] = t
                    ref = handle_task_ray_gpu.remote(CONFIG_PATH, batch_to_send, PODCAST_PATH, OUTPUT_ROOT_DIR)
                    result_refs.append(ref)
                    result_ref_map[ref] = batch_to_send

            else:
                short_batch_buffer.append(task)
                if len(short_batch_buffer) >= BATCH_SIZE:
                    batch_to_send = short_batch_buffer[:BATCH_SIZE]
                    short_batch_buffer = short_batch_buffer[BATCH_SIZE:]
                    
                    for t in batch_to_send: tasks["processing"][get_task_key(t)] = t
                    ref = handle_task_ray_gpu.remote(CONFIG_PATH, batch_to_send, PODCAST_PATH, OUTPUT_ROOT_DIR)
                    result_refs.append(ref)
                    result_ref_map[ref] = batch_to_send
            
            if len(result_refs) >= MAX_POOL_SIZE: break
        
        # --- Tail Flushing ---
        if not task_queue and len(result_refs) < MAX_POOL_SIZE:
            # 队列彻底为空, 发送残余长音频
            if long_batch_buffer:
                for t in long_batch_buffer: tasks["processing"][get_task_key(t)] = t
                ref = handle_task_ray_gpu.remote(CONFIG_PATH, long_batch_buffer, PODCAST_PATH, OUTPUT_ROOT_DIR)
                result_refs.append(ref)
                result_ref_map[ref] = long_batch_buffer
                long_batch_buffer = []
            
            if short_batch_buffer:
                for t in short_batch_buffer: tasks["processing"][get_task_key(t)] = t
                ref = handle_task_ray_gpu.remote(CONFIG_PATH, short_batch_buffer, PODCAST_PATH, OUTPUT_ROOT_DIR)
                result_refs.append(ref)
                result_ref_map[ref] = short_batch_buffer
                short_batch_buffer = []

        # --- Retrieval ---
        if not result_refs and not task_queue and not short_batch_buffer and not long_batch_buffer: break
        
        wait_timeout = 5 if len(result_refs) >= MAX_POOL_SIZE else 0.1
        ready_refs, result_refs = ray.wait(result_refs, num_returns=1, timeout=wait_timeout)
        
        if not ready_refs:
            monitor.report(tasks)
            continue

        for ready_ref in ready_refs:
            task_batch = result_ref_map.pop(ready_ref, None)
            try:
                successful_tasks, failed_tasks = ray.get(ready_ref)
            except Exception as e:
                logger.error(f"Ray Crash: {e}")
                successful_tasks = []
                failed_tasks = task_batch if task_batch else []

            for task in successful_tasks:
                task_key = get_task_key(task)
                tasks["processing"].pop(task_key, None)
                tasks['complete'].append(task)
                tasks['complete_num'] += 1
                tasks['complete_total_hour'] += task['audio_duration_second'] / 3600

            for task in failed_tasks:
                task_key = get_task_key(task)
                tasks["processing"].pop(task_key, None)
                tasks['failed'].append(task)
                tasks['failed_num'] += 1
                tasks['failed_total_hour'] += task['audio_duration_second'] / 3600

        # --- Save ---
        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
            logger.info("Auto-saving...")
            save_tasks(tasks, TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE)
            last_save_time = current_time
            monitor.report(tasks, force_send=True)

    logger.info("Done.")
    save_tasks(tasks, TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE)
    monitor.report(tasks, force_send=True)

def main():
    ray.init(ignore_reinit_error=True) 
    try:
        run()
    except Exception:
        logger.error(f"Main Crashed: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
    main()
