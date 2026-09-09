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
# --- Daily Summary ---
# ------------------------------------------------------------------

class DailySummary:
    def __init__(self, dataset_name, report_hour=10):
        self.dataset_name = dataset_name
        self.report_hour = report_hour
        self.last_report_date = None

    def check_and_report(self, tasks):
        """检查是否到达每日报告时间，如果是则发送汇总报告"""
        from datetime import datetime, timedelta

        now = datetime.now()
        current_date = now.date()
        current_hour = now.hour

        # 检查是否已经过了报告时间且今天还没有报告过
        if current_hour >= self.report_hour and self.last_report_date != current_date:
            # 计算统计时间范围：昨天10:00到今天10:00
            today_report_time = datetime.combine(current_date, datetime.min.time()).replace(hour=self.report_hour)
            yesterday_report_time = today_report_time - timedelta(days=1)

            start_timestamp = yesterday_report_time.timestamp()
            end_timestamp = today_report_time.timestamp()

            # 统计该时间范围内的数据
            stats = self._calculate_stats(tasks, start_timestamp, end_timestamp)

            # 发送报告
            self._send_daily_report(stats, yesterday_report_time, today_report_time)

            # 更新最后报告日期
            self.last_report_date = current_date

    def _calculate_stats(self, tasks, start_time, end_time):
        """计算指定时间范围内的统计数据"""
        success_count = 0
        success_hours = 0.0
        failed_count = 0
        failed_hours = 0.0

        # 统计成功的任务
        for task in tasks.get('complete', []):
            completed_at = task.get('completed_at', 0)
            if start_time <= completed_at <= end_time:
                success_count += 1
                success_hours += task.get('audio_duration_second', 0) / 3600

        # 统计失败的任务
        for task in tasks.get('failed', []):
            completed_at = task.get('completed_at', 0)
            if start_time <= completed_at <= end_time:
                failed_count += 1
                failed_hours += task.get('audio_duration_second', 0) / 3600

        return {
            'success_count': success_count,
            'success_hours': success_hours,
            'failed_count': failed_count,
            'failed_hours': failed_hours,
            'total_count': success_count + failed_count,
            'total_hours': success_hours + failed_hours
        }

    def _send_daily_report(self, stats, start_time, end_time):
        """发送每日汇总报告"""
        from datetime import datetime

        start_str = start_time.strftime('%Y-%m-%d %H:%M')
        end_str = end_time.strftime('%Y-%m-%d %H:%M')

        msg = (
            f"📊 {self.dataset_name} 每日数据处理汇总\n"
            f"时间范围: {start_str} - {end_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ 成功: {stats['success_count']}个任务, {stats['success_hours']:.2f}小时\n"
            f"❌ 失败: {stats['failed_count']}个任务, {stats['failed_hours']:.2f}小时\n"
            f"📈 总计: {stats['total_count']}个任务, {stats['total_hours']:.2f}小时"
        )

        try:
            msg_bot.send_msg(msg)
            logger.info(f"Daily summary sent: {stats['total_hours']:.2f}h processed")
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

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
    filename = os.path.basename(task["audio_path"])
    return f"{task['relative_path']}/{filename}"

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

@ray.remote(num_cpus=CPU_PER_TASK_GPU, num_gpus=GPU_PER_TASK, scheduling_strategy="SPREAD", max_calls=1, max_retries=0)
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
    daily_summary = DailySummary(DATASET_NAME, report_hour=10)

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
            daily_summary.check_and_report(tasks)
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
                task['completed_at'] = time.time()
                tasks['complete'].append(task)
                tasks['complete_num'] += 1
                tasks['complete_total_hour'] += task['audio_duration_second'] / 3600

            for task in failed_tasks:
                task_key = get_task_key(task)
                tasks["processing"].pop(task_key, None)
                task['completed_at'] = time.time()
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
    ray.init(address="127.0.0.1:6379", ignore_reinit_error=True) 
    try:
        run()
    except Exception:
        logger.error(f"Main Crashed: {traceback.format_exc()}")

if __name__ == '__main__':
    main()
