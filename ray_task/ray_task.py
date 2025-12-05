# ray_task/main.py

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

# --- Configuration Imports (Assume these are defined elsewhere) ---
from ray_task.config import (BATCH_SIZE, CONFIG_PATH, CPU_PER_TASK_CPU,
                             CPU_PER_TASK_GPU, DATASET_NAME, GPU_PER_TASK,
                             OUTPUT_PATH, OUTPUT_ROOT_DIR, PODCAST_PATH,
                             TASK_RESULT_BACKUP_FILE, TASK_RESULT_FILE)
from ray_task.load_task import load_tasks
# from ray_task.podcast_sort import sort_podcast_todo_tasks # 排序可选
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline

ray.init(ignore_reinit_error=True)

# ----------------------------------------------------------------------
# --- Global Settings ---
# ----------------------------------------------------------------------


# **I/O 优化**: 任务状态保存频率 (秒)
SAVE_INTERVAL_SECONDS = 60 

# ----------------------------------------------------------------------
# --- Utilities ---
# ----------------------------------------------------------------------

def get_ray_available_resources():
    """获取 Ray 集群的可用资源."""
    available_resources = ray.available_resources()
    available_cpus = available_resources.get('CPU', 0)
    available_gpus = available_resources.get('GPU', 0)
    return available_cpus, available_gpus

def save_tasks(file_path, backup_file_path, data):
    """原子性地保存任务状态到文件."""
    with open(backup_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    shutil.move(backup_file_path, file_path)

def get_task_key(task):
    """获取任务的唯一标识键."""
    task_key = task["relative_path"]
    return task_key

# ----------------------------------------------------------------------
# --- Task Handling Logic (Batch-aware) ---
# ----------------------------------------------------------------------

def handle_task(config_path, task_batch, prefix_path, output_path):
    """
    实际处理任务的逻辑。接收 Batch，调用管道，解析子任务状态，返回成功和失败的列表。
    """
    successful_tasks = []
    failed_tasks = []
    
    try:
        # 调用 Batch 版本的音频预处理管线
        batch_status = run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_path)
        
        # 1. 检查 run_audio_preprocess_pipeline 自身是否失败 (如初始化失败)
        if batch_status != "SUCCESS":
            logger.error(f"Batch task failed with status: {batch_status}")
            # 将 Batch 中所有任务都视为失败
            for task in task_batch:
                failed_tasks.append(task)
            return successful_tasks, failed_tasks 

    except Exception as e:
        # 2. 捕获管道执行时的任何意外异常
        logger.error(f"handle Batch task error {traceback.format_exc()}")
        for task in task_batch:
            failed_tasks.append(task)
        return successful_tasks, failed_tasks 

    # 3. 解析 Batch 内部的子任务状态 (依赖管道在 task 字典中设置的 "pipeline_status")
    for task in task_batch:
        task_status = task.get("pipeline_status", "UNKNOWN")
        task_key = get_task_key(task)
        
        if task_status == "SUCCESS":
            successful_tasks.append(task)
        else:
            logger.warning(f"Sub-task {task_key} failed or unknown status: {task_status}")
            failed_tasks.append(task)

    return successful_tasks, failed_tasks 

# CPU 任务 (现在处理 Batch)
@ray.remote(num_cpus=CPU_PER_TASK_CPU, max_retries=0)
def handle_task_ray_cpu(config_path, task_batch, audio_prefix_path, output_path):
    return handle_task(config_path, task_batch, audio_prefix_path, output_path)

# GPU 任务 (现在处理 Batch)
@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, max_retries=0)
def handle_task_ray_gpu(config_path, task_batch, audio_prefix_path, output_path):
    return handle_task(config_path, task_batch, audio_prefix_path, output_path)

# ----------------------------------------------------------------------
# --- Scheduling Policy and Progress Reporting ---
# ----------------------------------------------------------------------

def get_optimal_task_strategy():
    """根据可用资源返回最优的任务函数和最大并发数。"""
    available_cpus, available_gpus = get_ray_available_resources()
    
    optimal_task_func = handle_task_ray_cpu 
    MAX_NUM_PENDING_TASKS = 1 

    if available_gpus >= GPU_PER_TASK and available_cpus >= CPU_PER_TASK_GPU:
        optimal_task_func = handle_task_ray_gpu
        MAX_NUM_PENDING_TASKS = int(available_gpus / GPU_PER_TASK)
        
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

# ----------------------------------------------------------------------
# --- Main Run Loop (Optimized) ---
# ----------------------------------------------------------------------

def run():
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    tasks = load_tasks(TASK_RESULT_FILE)

    result_refs = []
    result_ref_map = {}
    
    # I/O 优化状态变量
    last_save_time = time.time()
    batches_since_last_save = 0

    while tasks['todo'] or result_refs:

        optimal_task_func, MAX_NUM_PENDING_TASKS = get_optimal_task_strategy()
        
        logger.debug(f"Current optimal_task_func={optimal_task_func._function.__name__}, MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")

        while tasks['todo'] and len(result_refs) < MAX_NUM_PENDING_TASKS:
            
            # 1. 筛选任务并构建 Batch
            task_batch = []
            
            # 从 todo 列表中收集 BATCH_SIZE 个合格任务并移除不合格任务
            i = 0
            while i < len(tasks['todo']):
                task = tasks['todo'][i]
                task_key = get_task_key(task)
                
                if task_key in tasks["processing"]:
                    logger.warning(f"Task {task_key} is already processing. Removing from todo.")
                    tasks['todo'].pop(i) 
                    continue # 不递增 i
                    
                if task["audio_duration_second"] < 10: 
                    logger.info(f"Skipping short task {task_key}. Removing from todo.")
                    tasks['todo'].pop(i)
                    continue # 不递增 i
                    
                task_batch.append(task)
                i += 1 # 只有合格任务才递增 i
                
                if len(task_batch) >= BATCH_SIZE:
                    break
            
            # 如果构建的 batch 是空的，跳出提交循环
            if not task_batch:
                continue

            # 2. 提交 Batch 任务
            batch_key = get_task_key(task_batch[0])
            
            # 将 Batch 中的所有任务移入 processing 状态
            for task in task_batch:
                tasks["processing"][get_task_key(task)] = task
                
            # 从 todo 列表中移除已提交的任务
            tasks['todo'] = tasks['todo'][len(task_batch):] 
            
            # 注意：移除此处 I/O，转为定时保存！
            # save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks) 

            logger.debug(f"Submitting Batch (size={len(task_batch)}) starting with task={batch_key}...")

            result_ref = optimal_task_func.remote(CONFIG_PATH, task_batch, PODCAST_PATH, OUTPUT_ROOT_DIR)
            result_refs.append(result_ref)
            result_ref_map[result_ref] = task_batch
            
            batches_since_last_save += 1 # 增加计数器
        
        
        if not tasks['todo'] and not result_refs:
            logger.info("All tasks completed or failed.")
            break
        
        if len(result_refs) > 0:
            # **调度优化**: 动态调整 num_returns，提高资源回收效率
            num_to_wait = max(10, int(len(result_refs) * 0.2))
            num_to_wait = min(num_to_wait, 20) 

            ready_refs, result_refs = ray.wait(result_refs, num_returns=num_to_wait, timeout=5) 
            
            if not ready_refs:
                 print_progress(tasks)
                 time.sleep(0.1) # **调度优化**: 缩短等待时间
                 continue

            logger.debug(f"ray_wait returned {len(ready_refs)} completed Batch(es). Remaining pending: {len(result_refs)}")
            
            for ready_ref in ready_refs:
                ready_task_batch = result_ref_map.pop(ready_ref, None)
                
                if ready_task_batch is None:
                    logger.error(f"ready_ref can not find Batch in map.")
                    continue

                try:
                    # 获取 Batch 的结果：(成功的任务列表, 失败的任务列表)
                    successful_tasks, failed_tasks = ray.get(ready_ref)
                    
                    # 遍历成功的任务并更新状态
                    for task in successful_tasks:
                        task_key = get_task_key(task)
                        tasks["processing"].pop(task_key, None) 
                        duration_hour = task['audio_duration_second'] / 3600
                        tasks['complete'].append(task)
                        tasks['complete_num'] += 1
                        tasks['complete_total_hour'] += duration_hour

                    # 遍历失败的任务并更新状态
                    for task in failed_tasks:
                        task_key = get_task_key(task)
                        tasks["processing"].pop(task_key, None) 
                        duration_hour = task['audio_duration_second'] / 3600
                        tasks['failed'].append(task)
                        tasks['failed_num'] += 1
                        tasks['failed_total_hour'] += duration_hour
                
                except Exception as e:
                    # 整个 Batch 任务获取结果失败
                    logger.error(f"Failed to get result for an entire Batch: {traceback.format_exc()}")
                    
                    # 将 Batch 中所有的任务都标记为失败
                    for task in ready_task_batch:
                        task_key = get_task_key(task)
                        tasks["processing"].pop(task_key, None)
                        
                        tasks['failed'].append(task)
                        tasks['failed_num'] += 1
                        tasks['failed_total_hour'] += task['audio_duration_second'] / 3600
                       
        print_progress(tasks)
        
        # **I/O 优化**: 定时保存状态
        current_time = time.time()
        should_save = batches_since_last_save > 0 and (current_time - last_save_time >= SAVE_INTERVAL_SECONDS)
        should_save_on_exit = (not tasks['todo'] and not result_refs) # 任务全部完成时也要保存
        
        if should_save or should_save_on_exit:
            logger.debug(f"Saving tasks state (Batches since last save: {batches_since_last_save}).")
            save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks)
            last_save_time = current_time
            batches_since_last_save = 0

        # time.sleep(1) # **调度优化**: 移除此处的 sleep，保持循环高频运行

def main():
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")

if __name__ == '__main__':
    main()
