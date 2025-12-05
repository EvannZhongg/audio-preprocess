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

from ray_task.config import (CONFIG_PATH, CPU_PER_TASK_CPU, CPU_PER_TASK_GPU,
                             DATASET_NAME, GPU_PER_TASK, OUTPUT_PATH,
                             OUTPUT_ROOT_DIR, PODCAST_PATH,
                             TASK_RESULT_BACKUP_FILE, TASK_RESULT_FILE)
from ray_task.load_task import load_tasks
from ray_task.podcast_sort import sort_podcast_todo_tasks
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline

ray.init(ignore_reinit_error=True)


def get_ray_available_resources():
    """获取 Ray 集群的可用资源."""
    available_resources = ray.available_resources()
    available_cpus = available_resources.get('CPU', 0)
    available_gpus = available_resources.get('GPU', 0)
    return available_cpus, available_gpus

def save_tasks(file_path, backup_file_path, data):
    with open(backup_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    shutil.move(backup_file_path, file_path)

def get_task_key(task):
    task_key = task["relative_path"]
    return task_key

def handle_task(config_path, task, prefix_path, output_path):
    """
    实际处理任务的逻辑
    """
    try:
        input_audio_path = task["audio_path"]
        ret = run_audio_preprocess_pipeline(config_path, input_audio_path, prefix_path, output_path)
        return ret, task
    except Exception as e:
        logger.error(f"handle task={get_task_key(task)} error {traceback.format_exc()}")
        return "EXCEPTION", task

# CPU 任务
@ray.remote(num_cpus=CPU_PER_TASK_CPU, max_retries=0)
def handle_task_ray_cpu(config_path, task, audio_prefix_path, output_path):
    return handle_task(config_path, task, audio_prefix_path, output_path)

# GPU 任务
@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, max_retries=0)
def handle_task_ray_gpu(config_path, task, audio_prefix_path, output_path):
    return handle_task(config_path, task, audio_prefix_path, output_path)

# --- 调度策略函数 ---
def get_optimal_task_strategy():
    """根据可用资源返回最优的任务函数和最大并发数。"""
    available_cpus, available_gpus = get_ray_available_resources()
    
    optimal_task_func = handle_task_ray_cpu 
    MAX_NUM_PENDING_TASKS = 1 

    if available_gpus >= GPU_PER_TASK and available_cpus >= CPU_PER_TASK_GPU:
        optimal_task_func = handle_task_ray_gpu
        MAX_NUM_PENDING_TASKS = int(available_gpus / GPU_PER_TASK)
        
        # 确保 GPU 并发数至少为 1 (处理浮点误差)
        if MAX_NUM_PENDING_TASKS == 0:
            MAX_NUM_PENDING_TASKS = 1
        
        required_cpu_for_gpu_tasks = MAX_NUM_PENDING_TASKS * CPU_PER_TASK_GPU
        if available_cpus < required_cpu_for_gpu_tasks:
             MAX_NUM_PENDING_TASKS = int(available_cpus / CPU_PER_TASK_GPU)
             
             if MAX_NUM_PENDING_TASKS == 0:
                 optimal_task_func = handle_task_ray_cpu

    if optimal_task_func == handle_task_ray_cpu:
        MAX_NUM_PENDING_TASKS = int(available_cpus / CPU_PER_TASK_CPU)

    if MAX_NUM_PENDING_TASKS == 0:
        MAX_NUM_PENDING_TASKS = 1

    return optimal_task_func, MAX_NUM_PENDING_TASKS


last_send_bot_msg = time.time()
def print_progress(tasks):
    global last_send_bot_msg
    total_hour = tasks["total_hour"]
    handled_hour = tasks["complete_total_hour"]
    log_str = f"{DATASET_NAME}: audio-pipeline handled_hour/total_hour={round(handled_hour, 2)}/{round(total_hour, 2)}"
    logger.debug(log_str)
    if time.time() - last_send_bot_msg > 3600 * 6:
        last_send_bot_msg = time.time()
        msg_bot.send_msg(log_str)


def check_dirty_data(task):
    relative_path = task["relative_path"]
    cur_task_dir = f"{OUTPUT_PATH}/{relative_path}"
    if os.path.exists(cur_task_dir):
        logger.error(f"unexcepted dir {cur_task_dir}")
        sys.exit(1)


def run():
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tasks = load_tasks(TASK_RESULT_FILE)

    result_refs = []
    result_ref_map = {}
    
    while tasks['todo'] or result_refs:
        # sort_podcast_todo_tasks(tasks['todo']) # 保持原有的排序逻辑

        optimal_task_func, MAX_NUM_PENDING_TASKS = get_optimal_task_strategy()
        
        logger.debug(f"Current optimal_task_func={optimal_task_func._function.__name__}, MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")

        while tasks['todo'] and len(result_refs) < MAX_NUM_PENDING_TASKS:
            
            task = tasks['todo'][0] 
            task_key = get_task_key(task)

            if task_key in tasks["processing"]:
                logger.warning(f"Task {task_key} is already processing but found in todo list. Skipping.")
                tasks['todo'].pop(0) 
                continue

            if task["audio_duration_second"] < 10: 
                logger.info(f"Skipping short task {task_key} (duration={task['audio_duration_second']}s).")
                tasks['todo'].pop(0)
                continue

            # check_dirty_data(task) 

            logger.debug(f"Submitting task={task_key} using {optimal_task_func._function.__name__}")

            tasks["processing"][task_key] = tasks['todo'].pop(0) 
            save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks) 

            result_ref = optimal_task_func.remote(CONFIG_PATH, task, PODCAST_PATH, OUTPUT_ROOT_DIR)
            result_refs.append(result_ref)
            result_ref_map[result_ref] = task
        
        
        if not tasks['todo'] and not result_refs:
            logger.info("All tasks completed or failed.")
            break
        
        if len(result_refs) > 0:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1, timeout=5) 
            if not ready_refs:
                 print_progress(tasks)
                 time.sleep(1)
                 continue

            logger.debug(f"ray_wait returned {len(ready_refs)} completed task(s). Remaining pending: {len(result_refs)}")
            
            for ready_ref in ready_refs:
                ready_task = result_ref_map.pop(ready_ref, None)
                
                if ready_task is None:
                    logger.error(f"ready_ref can not find task in map.")
                    continue

                ready_task_key = get_task_key(ready_task)

                try:
                    task_result_status, ret_task = ray.get(ready_ref)
                    logger.debug(f"Task {ready_task_key} completed with status: {task_result_status}")
                    
                    ret_task_key = get_task_key(ret_task)
                    tasks["processing"].pop(ret_task_key, None) # 从 processing 移除

                    duration_hour = ret_task['audio_duration_second'] / 3600
                    
                    if task_result_status == "SUCCESS":
                        tasks['complete'].append(ret_task)
                        tasks['complete_num'] += 1
                        tasks['complete_total_hour'] += duration_hour
                    else:
                        tasks['failed'].append(ret_task)
                        tasks['failed_num'] += 1
                        tasks['failed_total_hour'] += duration_hour
                
                except Exception as e:
                    logger.error(f"Failed to get result for task {ready_task_key}: {traceback.format_exc()}")
                    
                    tasks["processing"].pop(ready_task_key, None)
                    # 将任务标记为失败
                    tasks['failed'].append(ready_task)
                    tasks['failed_num'] += 1
                    tasks['failed_total_hour'] += ready_task['audio_duration_second'] / 3600
                       
        print_progress(tasks)
        
        # 任务处理完成后保存状态
        save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks)

        time.sleep(1)

def main():
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")
