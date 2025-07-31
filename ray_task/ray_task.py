import sys
import ray
import json
import traceback
import os
import time

from utils.logger import Logger
logger = Logger.get_logger(f"ray-task")

from ray_task.config import OUTPUT_PATH, PODCAST_PATH, TASK_RESULT_FILE
from ray_task.run_pipeline_cmd import run_audio_preprocess_pipeline
from ray_task.load_task import load_tasks

ray.init(ignore_reinit_error=True)

def get_ray_total_cpu():
    nodes = ray.nodes()
    total_cpus = sum(node['Resources'].get('CPU', 0) for node in nodes)
    return total_cpus

def save_tasks(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_task_key(task):
    task_key = task["podcast_name"] + "/" + task["episode_name"]
    return task_key

def handle_task(task, prefix_path, output_path):
    """
    task
    {
        "podcast_name": "xx",
        "episode_name": "xx",
        "audio_duration_second": 0 
    }
    """

    try:
        podcast_name = task["podcast_name"]
        episode_name = task["episode_name"]
        input_audio_path = f"{prefix_path}/{podcast_name}/{episode_name}"

        task_key = get_task_key(task)
        ret = run_audio_preprocess_pipeline(input_audio_path, output_path, task_key)
        # logger.debug(f"handle task={task} ret={ret}")
        return ret["status"], task
    except Exception as e:
        logger.error(f"handle task={task} error {traceback.format_exc()}")
        return "EXCEPTION", task

@ray.remote(num_cpus=4, max_retries=0)
def handle_task_ray(task, audio_prefix_path, output_path):
    return handle_task(task, audio_prefix_path, output_path)


def print_progress(tasks):
    total_hour = tasks["total_hour"]
    handled_hour = tasks["complete_total_hour"]
    logger.debug(f"task_stat handled_hour/total_hour={handled_hour}/{total_hour}")


def check_dirty_data(task):
    podcast_name = task["podcast_name"]
    episode_name = os.path.splitext(os.path.basename(task["episode_name"]))[0]
    cur_task_dir = f"{OUTPUT_PATH}/{podcast_name}/{episode_name}"
    if os.path.exists(cur_task_dir):
        logger.error(f"unexcepted dir {cur_task_dir}")
        sys.exit(1)


def run():
    tasks = load_tasks(TASK_RESULT_FILE)

    result_refs = []
    while tasks['todo']:
        need_delete_task_key = []

        # logger.debug(f"all tasks {tasks}")

        for task in tasks['todo']:
            check_dirty_data(task)

            # save processing data
            task_key = get_task_key(task)
            if task_key in tasks["processing"]:
                continue

            # handle ray task
            MAX_NUM_PENDING_TASKS = int(get_ray_total_cpu() / 4) + 1
            if len(result_refs) < MAX_NUM_PENDING_TASKS:
                logger.debug(f"append task={task} MAX_NUM_PENDING_TASKS={MAX_NUM_PENDING_TASKS}")

                tasks["processing"][task_key] = task
                save_tasks(TASK_RESULT_FILE, tasks)

                result_refs.append(handle_task_ray.remote(task, PODCAST_PATH, OUTPUT_PATH))
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
                task_result_list = ray.get(ready_refs)
                for task_result_status, ret_task in task_result_list:
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
                else:
                    i += 1
        save_tasks(TASK_RESULT_FILE, tasks)

        time.sleep(1)

def main():
    try:
        run()
    except Exception as e:
        logger.error(f"main exception {traceback.format_exc()}")