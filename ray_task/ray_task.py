import json
import os
import shutil
import sys
import time
import traceback

import ray

from utils import msg_bot
from utils.logger import Logger

logger = Logger.get_logger(f"ray-task")


from ray_task.config import (BATCH_SIZE, CONFIG_PATH, CPU_PER_TASK_CPU,
                             CPU_PER_TASK_GPU, DATASET_NAME, GPU_PER_TASK,
                             MAX_POOL_SIZE, OUTPUT_PATH, OUTPUT_ROOT_DIR,
                             PODCAST_PATH, TASK_RESULT_BACKUP_FILE,
                             TASK_RESULT_FILE)
from ray_task.load_task import load_tasks
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline

assert GPU_PER_TASK > 0, "GPU_PER_TASK must be > 0 for GPU-only mode"


# ------------------------------------------------------------------
# --- Global Settings ---
# ------------------------------------------------------------------

SAVE_INTERVAL_SECONDS = 3 * 60 * 60  # 3小时

# ------------------------------------------------------------------
# --- Utilities ---
# ------------------------------------------------------------------

def get_ray_available_resources():
    available_resources = ray.available_resources()
    available_cpus = available_resources.get('CPU', 0)
    available_gpus = available_resources.get('GPU', 0)
    return available_cpus, available_gpus

    
def save_tasks(tasks, main_file_path, backup_file_path=None):
    """
    Save task state to main file, and optionally to a backup file.
    """
    # 清理 todo 列表中的非 dict 项（防御性编程）
    if 'todo' in tasks and isinstance(tasks['todo'], list):
        tasks['todo'] = [item for item in tasks['todo'] if isinstance(item, dict)]

    # 保存主文件
    with open(main_file_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)

    # 如果指定了备份路径，复制一份
    if backup_file_path:
        shutil.copy(main_file_path, backup_file_path)
        

def get_task_key(task):
    return task["relative_path"]

# ------------------------------------------------------------------
# --- Task Handling Logic (GPU-only) ---
# ------------------------------------------------------------------

def handle_task(config_path, task_batch, prefix_path, output_path):
    successful_tasks = []
    failed_tasks = []
    
    padding_tasks = [task for task in task_batch if task["relative_path"].startswith("padding_")]
    if padding_tasks:
        logger.debug(f"Processing padding tasks: {len(padding_tasks)}")
        # 直接返回成功（填充任务不需要实际处理）
        for task in padding_tasks:
            task["pipeline_status"] = "SUCCESS"
        # 从任务列表中移除填充任务
        task_batch = [task for task in task_batch if not task["relative_path"].startswith("padding_")]
    
    # 如果没有实际任务，直接返回
    if not task_batch:
        return [], []

    try:
        batch_status = run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_path)
        if batch_status != "SUCCESS":
            logger.error(f"Batch task failed with status: {batch_status}")
            failed_tasks = task_batch  # 所有任务失败
            return successful_tasks, failed_tasks

    except Exception as e:
        logger.error(f"handle Batch task error {traceback.format_exc()}")
        failed_tasks = task_batch  # 所有任务失败
        return successful_tasks, failed_tasks

    for task in task_batch:
        task_status = task.get("pipeline_status", "UNKNOWN")
        if task_status == "SUCCESS":
            successful_tasks.append(task)
        else:
            logger.warning(f"Sub-task {get_task_key(task)} failed: {task_status}")
            failed_tasks.append(task)
            
    if padding_tasks:
        successful_tasks.extend(padding_tasks)

    return successful_tasks, failed_tasks

@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, scheduling_strategy="STRICT_SPREAD", max_retries=0)
def handle_task_ray_gpu(config_path, task_batch, audio_prefix_path, output_path):
    return handle_task(config_path, task_batch, audio_prefix_path, output_path)

# ------------------------------------------------------------------
# --- Scheduling Policy (GPU-only) ---
# ------------------------------------------------------------------

def get_optimal_task_strategy():
    available_cpus, available_gpus = get_ray_available_resources()

    if available_gpus < GPU_PER_TASK:
        logger.debug(f"Insufficient GPU resources (need {GPU_PER_TASK}, available: {available_gpus}). Waiting...")
        return handle_task_ray_gpu, 0

    max_gpu_tasks = available_gpus // GPU_PER_TASK
    max_cpu_tasks = available_cpus // CPU_PER_TASK_GPU
    MAX_NUM_PENDING_TASKS = min(max_gpu_tasks, max_cpu_tasks)
    
    if MAX_NUM_PENDING_TASKS == 0:
        logger.debug("Insufficient CPU resources for GPU tasks. Waiting...")
        return handle_task_ray_gpu, 0

    return handle_task_ray_gpu, MAX_NUM_PENDING_TASKS

last_send_bot_msg = time.time()
def print_progress(tasks):
    global last_send_bot_msg
    total_hour = tasks["total_hour"]
    handled_hour = tasks["complete_total_hour"]
    log_str = f"{DATASET_NAME}: audio-pipeline handled_hour/total_hour={round(handled_hour, 2)}/{round(total_hour, 2)}"
    logger.debug(log_str)
    
    if time.time() - last_send_bot_msg > 3 * 3600:  # 3小时
        last_send_bot_msg = time.time()
        msg_bot.send_msg(log_str)

# ------------------------------------------------------------------
# --- Main Run Loop (GPU-only) ---
# ------------------------------------------------------------------

def run():
    if os.path.exists(OUTPUT_PATH) and os.listdir(OUTPUT_PATH):
        logger.warning(f"OUTPUT_PATH is not empty: {OUTPUT_PATH}. Clearing...")
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tasks = load_tasks(TASK_RESULT_FILE)
    
    # MAX_POOL_SIZE = 60
    
    
    def ensure_task_pool_size():
        if len(tasks['todo']) < MAX_POOL_SIZE:
            padding_tasks = []
            num_padding = MAX_POOL_SIZE - len(tasks['todo'])
            for i in range(num_padding):
                # 创建填充任务（相对路径以padding_开头，避免被跳过）
                padding_task = {
                    "relative_path": f"padding_{i}",
                    "audio_duration_second": 15,  # 15秒（避免被跳过条件）
                    # 其他字段可以留空，因为填充任务会特殊处理
                }
                padding_tasks.append(padding_task)
            
            tasks['todo'].extend(padding_tasks)
            logger.info(f"Added {num_padding} padding tasks to ensure GPU utilization (now: {len(tasks['todo'])})")

    result_refs = []
    result_ref_map = {}
    last_save_time = time.time()

    while tasks['todo'] or result_refs:
        
        ensure_task_pool_size()
        
        optimal_task_func, MAX_NUM_PENDING_TASKS = get_optimal_task_strategy()

        logger.debug(f"GPU-only: using {optimal_task_func._function.__name__}, MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")

        while tasks['todo'] and len(result_refs) < MAX_POOL_SIZE:
            task_batch = []
            i = 0
            while i < len(tasks['todo']):
                task = tasks['todo'][i]
                task_key = get_task_key(task)

                if task_key in tasks["processing"]:
                    logger.warning(f"Task {task_key} is already processing. Removing from todo.")
                    tasks['todo'].pop(i)
                    continue  

                if task["audio_duration_second"] < 10:
                    logger.info(f"Skipping short task {task_key}. Removing from todo.")
                    tasks['todo'].pop(i)
                    continue 

                task_batch.append(task)
                i += 1 

                if len(task_batch) >= BATCH_SIZE:
                    break

            if not task_batch:
                break

            # 标记为处理中
            for task in task_batch:
                tasks["processing"][get_task_key(task)] = task

            # 从todo移除已提交任务
            tasks['todo'] = tasks['todo'][len(task_batch):]

            logger.debug(f"Submitting GPU Batch (size={len(task_batch)}) starting with {get_task_key(task_batch[0])}...")

            result_ref = optimal_task_func.remote(CONFIG_PATH, task_batch, PODCAST_PATH, OUTPUT_ROOT_DIR)
            result_refs.append(result_ref)
            result_ref_map[result_ref] = task_batch

        if result_refs:
            num_to_wait = min(max(10, int(len(result_refs) * 0.2)), 20, len(result_refs))
            
            ready_refs, result_refs = ray.wait(result_refs, num_returns=num_to_wait, timeout=5)
            if not ready_refs:
                print_progress(tasks)
                time.sleep(0.1)
                continue

            logger.debug(f"Completed {len(ready_refs)} Batch(es). Remaining: {len(result_refs)}")

            for ready_ref in ready_refs:
                task_batch = result_ref_map.pop(ready_ref, None)
                if task_batch is None:
                    logger.error("Task batch not found in map. Skipping.")
                    continue

                try:
                    successful_tasks, failed_tasks = ray.get(ready_ref)
                except Exception as e:
                    logger.error(f"Failed to get Batch result: {traceback.format_exc()}")
                    for task in task_batch:
                        task_key = get_task_key(task)
                        tasks["processing"].pop(task_key, None)
                        tasks['failed'].append(task)
                        tasks['failed_num'] += 1
                        tasks['failed_total_hour'] += task['audio_duration_second'] / 3600
                    continue

                for task in successful_tasks:
                    task_key = get_task_key(task)
                    tasks["processing"].pop(task_key, None)
                    duration_hour = task['audio_duration_second'] / 3600
                    tasks['complete'].append(task)
                    tasks['complete_num'] += 1
                    tasks['complete_total_hour'] += duration_hour

                for task in failed_tasks:
                    task_key = get_task_key(task)
                    tasks["processing"].pop(task_key, None)
                    duration_hour = task['audio_duration_second'] / 3600
                    tasks['failed'].append(task)
                    tasks['failed_num'] += 1
                    tasks['failed_total_hour'] += duration_hour

            print_progress(tasks)

        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
            logger.debug("Saving tasks state (GPU-only mode)")
            save_tasks(tasks, TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE)
            last_save_time = current_time

def main():
    ray.init(ignore_reinit_error=True)
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")

if __name__ == '__main__':
    main()
