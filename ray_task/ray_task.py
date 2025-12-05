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


def get_ray_total_cpu():
    nodes = ray.nodes()
    total_cpus = sum(node['Resources'].get('CPU', 0) for node in nodes)
    return total_cpus

def get_ray_available_gpu():
    available_resources = ray.available_resources()
    available_gpus = available_resources.get('GPU', 0)
    return available_gpus

def save_tasks(file_path, backup_file_path, data):
    with open(backup_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    shutil.move(backup_file_path, file_path)

def get_task_key(task):
    task_key = task["relative_path"]
    return task_key

def handle_task(config_path, task, prefix_path, output_path):
    """
    task
    {
        "relative_path": "xx",
        "audio_path": "xx",
        "audio_duration_second": 0 
    }
    """

    try:
        relative_path = task["relative_path"]
        input_audio_path = task["audio_path"]

        task_key = get_task_key(task)
        # 实际运行音频预处理管线
        ret = run_audio_preprocess_pipeline(config_path, input_audio_path, prefix_path, output_path)
        # logger.debug(f"handle task={task} ret={ret}")
        return ret, task
    except Exception as e:
        logger.error(f"handle task={task} error {traceback.format_exc()}")
        return "EXCEPTION", task

# CPU 任务：保持原有的 4 CPU 占用
@ray.remote(num_cpus=CPU_PER_TASK_CPU, max_retries=0)
def handle_task_ray_cpu(config_path, task, audio_prefix_path, output_path):
    return handle_task(config_path, task, audio_prefix_path, output_path)

# GPU 任务：每个任务占用 5 CPU 和 0.5 GPU，实现一卡双任务
@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, max_retries=0)
def handle_task_ray_gpu(config_path, task, audio_prefix_path, output_path):
    return handle_task(config_path, task, audio_prefix_path, output_path)

def get_optimal_task_function():
    """根据可用资源返回最优的任务函数"""
    available_gpus = get_ray_available_gpu()
    if available_gpus >= GPU_PER_TASK: # 确保至少有 0.5 个 GPU 可用
        return handle_task_ray_gpu
    else:
        return handle_task_ray_cpu


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
    while tasks['todo']:
        # sort_podcast_todo_tasks(tasks['todo'])

        need_delete_task_key = []

        # 计算最大并发任务数
        optimal_task_func = get_optimal_task_function()
        
        if optimal_task_func == handle_task_ray_gpu:
            # 如果使用 GPU 任务 (0.5 GPU)，则并发数由可用 GPU 资源决定
            available_gpus = get_ray_available_gpu()
            MAX_NUM_PENDING_TASKS = int(available_gpus / GPU_PER_TASK)
            if MAX_NUM_PENDING_TASKS == 0 and available_gpus >= 0.1:
                # 浮点数误差处理，至少允许运行一个
                 MAX_NUM_PENDING_TASKS = 1
            if MAX_NUM_PENDING_TASKS == 0:
                 # 如果 GPU 资源不足，退回 CPU 任务的并发计算
                 MAX_NUM_PENDING_TASKS = int(get_ray_total_cpu() / CPU_PER_TASK_CPU)
        else:
            # 如果使用 CPU 任务 (4 CPU)，则并发数由总 CPU 资源决定
            MAX_NUM_PENDING_TASKS = int(get_ray_total_cpu() / CPU_PER_TASK_CPU)
            
        # 确保并发数至少为 1
        if MAX_NUM_PENDING_TASKS == 0:
            MAX_NUM_PENDING_TASKS = 1
        
        logger.debug(f"Current optimal_task_func={optimal_task_func.__name__}, MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")


        for task in tasks['todo']:
            # save processing data
            task_key = get_task_key(task)
            if task_key in tasks["processing"]:
                continue

            # if task["audio_duration_second"] < 600:
            if task["audio_duration_second"] < 10:
                need_delete_task_key.append(task_key)
                continue

            # check_dirty_data(task)

            # handle ray task
            if len(result_refs) < MAX_NUM_PENDING_TASKS:
                logger.debug(f"append task={task}")

                tasks["processing"][task_key] = task
                save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks)

                # 在循环内部重新获取一次，防止在等待过程中资源发生变化
                current_optimal_task_func = get_optimal_task_function()
                result_ref = current_optimal_task_func.remote(CONFIG_PATH, task, PODCAST_PATH, OUTPUT_ROOT_DIR)
                result_refs.append(result_ref)
                result_ref_map[result_ref] = task
            else:
                break

        if len(result_refs) == 0:
            if tasks['todo']:
                logger.error(f"no task, but todo not empty")
            return

        if len(result_refs) > 0:
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            logger.debug(f"ray_wait len(ready_refs)={len(ready_refs)} len(result_refs)={len(result_refs)} MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")
            if len(ready_refs) > 0:
                for ready_ref in ready_refs:
                    ready_task = result_ref_map.get(ready_ref, None)
                    result_ref_map.pop(ready_ref, None)
                    if ready_task is None:
                        ready_task_key = None
                        logger.error(f"ready_ref can not find task")
                    else:
                        ready_task_key = get_task_key(ready_task)

                    try:
                        task_result_status, ret_task = ray.get(ready_ref)
                        logger.debug(f"get_ray_task_result status={task_result_status} task={ret_task}")
                        ret_task_key = get_task_key(ret_task)
                        need_delete_task_key.append(ret_task_key)
                        tasks["processing"].pop(ret_task_key, None)
                        if task_result_status == "SUCCESS":
                            tasks['complete'].append(ret_task)
                            tasks['complete_num'] += 1
                            tasks['complete_total_hour'] += ret_task['audio_duration_second'] / 3600
                        else:
                            tasks['failed'].append(ret_task)
                            tasks['failed_num'] += 1
                            tasks['failed_total_hour'] += ret_task['audio_duration_second'] / 3600
                    except Exception as e:
                        logger.error(f"get ready_ref {ready_task} exception {traceback.format_exc()}")
                        if ready_task is not None and ready_task_key is not None:
                            need_delete_task_key.append(ready_task_key)
                            tasks["processing"].pop(ready_task_key, None)
                            tasks['failed'].append(ready_task)
                            tasks['failed_num'] += 1
                            tasks['failed_total_hour'] += ready_task['audio_duration_second'] / 3600
                       
        print_progress(tasks)

        # delete todo task
        for delete_task_key in need_delete_task_key:
            i = 0
            while i < len(tasks['todo']):
                task = tasks['todo'][i]
                cur_task_key = get_task_key(task)
                if cur_task_key == delete_task_key:
                    logger.debug(f"delete todo task {delete_task_key}")
                    del tasks['todo'][i]
                    break
                else:
                    i += 1
        save_tasks(TASK_RESULT_FILE, TASK_RESULT_BACKUP_FILE, tasks)

        time.sleep(1)

def main():
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")
