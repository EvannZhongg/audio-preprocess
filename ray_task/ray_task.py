import json
import math
import os
import shutil
import sys
import time
import traceback

import ray

from utils import msg_bot
from utils.logger import Logger

logger = Logger.get_logger(f"ray-task")

from ray_task.config import (BATCH_SIZE, CONFIG_PATH, CPU_PER_TASK_GPU,
                             DATASET_NAME, GPU_PER_TASK,
                             MAX_AUDIO_DURATION_SECONDS, MAX_POOL_SIZE,
                             OUTPUT_PATH, OUTPUT_ROOT_DIR, PODCAST_PATH,
                             TASK_RESULT_BACKUP_FILE, TASK_RESULT_FILE)
from ray_task.load_task import load_tasks
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline

# ------------------------------------------------------------------
# --- Global Settings ---
# ------------------------------------------------------------------

SAVE_INTERVAL_SECONDS = 3 * 3600

# ------------------------------------------------------------------
# --- Utilities ---
# ------------------------------------------------------------------

def get_ray_available_resources():
    available_resources = ray.available_resources()
    available_cpus = available_resources.get('CPU', 0)
    available_gpus = available_resources.get('GPU', 0)
    return available_cpus, available_gpus

def save_tasks(tasks, main_file_path, backup_file_path=None):
    if 'todo' in tasks and isinstance(tasks['todo'], list):
        tasks['todo'] = [item for item in tasks['todo'] if isinstance(item, dict)]

    temp_file = main_file_path + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False) 
        
    shutil.move(temp_file, main_file_path)
    if backup_file_path:
        shutil.copy(main_file_path, backup_file_path)

def get_task_key(task):
    return task["relative_path"]

# ------------------------------------------------------------------
# --- Task Handling Logic ---
# ------------------------------------------------------------------

def handle_task(config_path, task_batch, prefix_path, output_path):
    successful_tasks = []
    failed_tasks = []
    
    padding_tasks = [task for task in task_batch if str(task.get("relative_path", "")).startswith("padding_")]
    real_tasks = [task for task in task_batch if not str(task.get("relative_path", "")).startswith("padding_")]

    for task in padding_tasks:
        task["pipeline_status"] = "SUCCESS"

    if not real_tasks:
        return padding_tasks, []

    try:
        batch_status = run_audio_preprocess_pipeline(config_path, real_tasks, prefix_path, output_path)
        if batch_status != "SUCCESS":
            logger.error(f"Batch task failed with status: {batch_status}")
            return successful_tasks, task_batch # 全部标记失败

    except Exception as e:
        logger.error(f"handle Batch task error {traceback.format_exc()}")
        return successful_tasks, task_batch


    for task in real_tasks:
        task_status = task.get("pipeline_status", "UNKNOWN")
        if task_status == "SUCCESS":
            successful_tasks.append(task)
        else:
            logger.warning(f"Sub-task {get_task_key(task)} failed: {task_status}")
            failed_tasks.append(task)
            
    successful_tasks.extend(padding_tasks)

    return successful_tasks, failed_tasks

@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, scheduling_strategy="SPREAD", max_retries=0)
def handle_task_ray_gpu(config_path, task_batch, audio_prefix_path, output_path):
    return handle_task(config_path, task_batch, audio_prefix_path, output_path)

# ------------------------------------------------------------------
# --- Main Run Loop ---
# ------------------------------------------------------------------

last_send_bot_msg = 0

def print_progress(tasks):
    global last_send_bot_msg
    total_hour = tasks.get("total_hour", 0)
    handled_hour = tasks.get("complete_total_hour", 0)
    failed_hour = tasks.get("failed_total_hour", 0)
    
    if total_hour > 0:
        percent = (handled_hour / total_hour) * 100
    else:
        percent = 0

    log_str = (f"{DATASET_NAME}: Progress {percent:.2f}% | "
               f"Done: {round(handled_hour, 2)}h | Failed: {round(failed_hour, 2)}h | "
               f"Pending Batch: {len(tasks['todo']) // BATCH_SIZE}")
    
    logger.debug(log_str)
    if time.time() - last_send_bot_msg > 3 * 3600: 
        last_send_bot_msg = time.time()
        msg_bot.send_msg(log_str)


def run():
    if os.path.exists(OUTPUT_PATH) and os.listdir(OUTPUT_PATH):
        logger.warning(f"OUTPUT_PATH is not empty: {OUTPUT_PATH}. Clearing...")
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    tasks = load_tasks(TASK_RESULT_FILE)
    tasks['todo'].sort(key=lambda x: x['audio_duration_second'])
    
    result_refs = []
    result_ref_map = {}
    last_save_time = time.time()

    logger.info(f"Starting GPU processing with {len(tasks['todo'])} tasks remaining.")

    # 主循环条件：只要还有待办任务，或者还有正在运行的任务结果没取回
    while tasks['todo'] or result_refs:
        
        # 1. 提交任务 (Submission)
        while len(result_refs) < MAX_POOL_SIZE and tasks['todo']:
            
            task_batch = []
            
            # 过滤掉正在处理的任务和无效任务（预处理）
            valid_batch_candidates = []
            
            idx_to_remove = []
            for i, task in enumerate(tasks['todo']):
                task_key = get_task_key(task)
                
                if task_key in tasks["processing"]:
                    idx_to_remove.append(i)
                    continue
                if task["audio_duration_second"] < 10:
                    logger.debug(f"Skipping short task {task_key}")
                    idx_to_remove.append(i) 
                    continue
                
                if task["audio_duration_second"] > MAX_AUDIO_DURATION_SECONDS:
                    logger.warning(f"Skipping too long task {task_key}: {task['audio_duration_second']}s > {MAX_AUDIO_DURATION_SECONDS}s")
                    tasks['failed'].append(task)
                    tasks['failed_num'] += 1
                    tasks['failed_total_hour'] += task['audio_duration_second'] / 3600
                    
                    idx_to_remove.append(i) 
                    continue
                
                valid_batch_candidates.append(task)
                idx_to_remove.append(i)
                
                if len(valid_batch_candidates) >= BATCH_SIZE:
                    break
            
            # 从 todo 中移除已被选走或跳过的项 (倒序移除以防索引偏移)
            for i in sorted(idx_to_remove, reverse=True):
                tasks['todo'].pop(i)
            
            task_batch = valid_batch_candidates

            if not task_batch:
                # todo 扫完了都没凑出有效任务，跳出提交循环
                break
                

            if len(task_batch) < BATCH_SIZE:
                logger.info(f"Padding last batch (size {len(task_batch)}) to {BATCH_SIZE}")
                needed = BATCH_SIZE - len(task_batch)
                for p_i in range(needed):
                    task_batch.append({
                        "relative_path": f"padding_{time.time()}_{p_i}",
                        "audio_duration_second": 15
                    })

            for task in task_batch:
                if not task["relative_path"].startswith("padding_"):
                    tasks["processing"][get_task_key(task)] = task

            result_ref = handle_task_ray_gpu.remote(CONFIG_PATH, task_batch, PODCAST_PATH, OUTPUT_ROOT_DIR)
            result_refs.append(result_ref)
            result_ref_map[result_ref] = task_batch
        
        # 如果没有任务在跑且 todo 为空，直接结束
        if not result_refs and not tasks['todo']:
            break


        wait_timeout = 10 if len(result_refs) >= MAX_POOL_SIZE else 0.5
        ready_refs, result_refs = ray.wait(result_refs, num_returns=1, timeout=wait_timeout)
        
        if not ready_refs:
            # 没结果返回，打印进度并继续循环尝试提交
            print_progress(tasks)
            continue

        for ready_ref in ready_refs:
            task_batch = result_ref_map.pop(ready_ref, None)
            
            try:
                successful_tasks, failed_tasks = ray.get(ready_ref)
            except Exception as e:
                logger.error(f"Ray get exception: {e}")
                successful_tasks = []
                failed_tasks = task_batch # 悲观策略，全算失败

            # 处理成功任务
            for task in successful_tasks:
                if task["relative_path"].startswith("padding_"): continue
                
                task_key = get_task_key(task)
                tasks["processing"].pop(task_key, None)
                tasks['complete'].append(task)
                tasks['complete_num'] += 1
                tasks['complete_total_hour'] += task['audio_duration_second'] / 3600

            # 处理失败任务
            for task in failed_tasks:
                if task["relative_path"].startswith("padding_"): continue

                task_key = get_task_key(task)
                tasks["processing"].pop(task_key, None)
                tasks['failed'].append(task)
                tasks['failed_num'] += 1
                tasks['failed_total_hour'] += task['audio_duration_second'] / 3600

        current_time = time.time()
        if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
            logger.info("Auto-saving tasks state...")
            save_tasks(tasks, TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE)
            last_save_time = current_time
            print_progress(tasks)

    logger.info("All tasks finished. Saving final state.")
    save_tasks(tasks, TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE)

def main():
    ray.init(ignore_reinit_error=True) 
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")

if __name__ == '__main__':
    main()
