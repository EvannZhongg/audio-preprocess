import json
import os
import re
import shutil
from pathlib import Path

from ray_task.config import OUTPUT_ROOT_DIR, PODCAST_DATA_FILE, TASK_RESULT_BACKUP_FILE
from utils.logger import Logger

logger = Logger.get_logger()

# ==============================================================================
# True: 启动时会将之前 'failed' 的任务重新捞回 'todo' 队列
# False: 忽略失败任务，只跑剩下的
# ==============================================================================
RETRY_FAILED = True 

def load_tasks(file_path):
    """
    加载任务列表，处理断点续传、失败重试及与原始数据同步。
    """
    tasks = {}
    loaded = False

    # ==============================================================================
    # 1. 加载文件 (主文件 -> 备份文件)
    # ==============================================================================
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            loaded = True
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Main task file is corrupted: {file_path}")
    
    # 如果主文件挂了，尝试加载备份
    if not loaded and os.path.exists(TASK_RESULT_BACKUP_FILE):
        try:
            logger.info(f"🔄 Attempting to restore from backup: {TASK_RESULT_BACKUP_FILE}")
            with open(TASK_RESULT_BACKUP_FILE, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            # 恢复成功，立即修复主文件
            shutil.copy(TASK_RESULT_BACKUP_FILE, file_path)
            loaded = True
            logger.info("✅ Restored from backup successfully.")
        except Exception:
            logger.error("❌ Backup file is also missing or corrupted.")

    # ==============================================================================
    # 2. 结构初始化 (兼容首次运行)
    # ==============================================================================
    keys_to_ensure = ['todo', 'complete', 'failed', 'processing']
    for key in keys_to_ensure:
        if key not in tasks: tasks[key] = {} if key == 'processing' else []

    if not loaded:
        logger.info("🆕 No history found (First Run). Initializing stats...")
        tasks.update({
            'total_num': 0, 'total_hour': 0, 
            'complete_num': 0, 'complete_total_hour': 0, 
            'failed_num': 0, 'failed_total_hour': 0
        })

    # ==============================================================================
    # 3. 恢复中断任务 (Processing -> Todo)
    # ==============================================================================
    # 将上次程序崩溃/停止时正在运行的任务，清理脏数据后放回待办列表
    if tasks.get("processing"):
        interrupted_tasks = tasks["processing"]
        logger.info(f"⚠️ Found {len(interrupted_tasks)} interrupted tasks. Re-queueing...")
        
        for k, task_info in interrupted_tasks.items():
            # A. 清理可能残留的脏文件目录
            try:
                relative_path = task_info.get("relative_path")
                audio_path = task_info.get("audio_path")
                if relative_path and audio_path:
                    # 使用文件名作为唯一目录标识 (适配 ray_task.py 的逻辑)
                    fid = re.sub(r"['\"\s]", "", Path(audio_path).stem)
                    processing_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path, fid)
                    
                    if os.path.exists(processing_dir):
                        shutil.rmtree(processing_dir)
            except Exception as e:
                logger.warning(f"Failed to clean dir for task {k}: {e}")

            # B. 放回待办列表
            tasks['todo'].append(task_info)
        
        # C. 清空进行中状态
        tasks["processing"] = {}

    # ==============================================================================
    # 4. 捞回失败任务 (Failed -> Todo)
    # ==============================================================================
    if RETRY_FAILED and tasks.get('failed'):
        failed_count = len(tasks['failed'])
        logger.info(f"♻️  [RETRY MODE] Recycling {failed_count} previously failed tasks to Todo queue...")
        
        tasks['todo'].extend(tasks['failed'])
        
        # 重置失败统计
        tasks['failed'] = []
        tasks['failed_num'] = 0
        tasks['failed_total_hour'] = 0

    # ==============================================================================
    # 5. 全量同步 (Sync with Manifest)
    # ==============================================================================
    logger.info("🔄 Syncing with original manifest...")
    
    try:
        if not os.path.exists(PODCAST_DATA_FILE):
             raise FileNotFoundError(f"Source data file missing: {PODCAST_DATA_FILE}")

        with open(PODCAST_DATA_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # 获取原始数据中的列表
            all_podcasts = raw_data.get('podcast_data', [])
    except Exception as e:
        logger.error(f"Failed to load raw manifest: {e}")
        all_podcasts = []

    if all_podcasts:
        existing_audio_paths = set()
        
        for t in tasks['complete']: existing_audio_paths.add(t['audio_path'])
        for t in tasks['failed']: existing_audio_paths.add(t['audio_path'])
        for t in tasks['todo']: existing_audio_paths.add(t['audio_path'])
        
        added_count = 0
        
        for podcast in all_podcasts:
            if podcast['audio_path'] not in existing_audio_paths:
                task_item = {
                    "relative_path": podcast['relative_path'], 
                    "audio_path": podcast['audio_path'],       
                    "audio_duration_second": podcast['audio_duration_second']
                }
                tasks['todo'].append(task_item)
                existing_audio_paths.add(podcast['audio_path'])
                added_count += 1

        if added_count > 0:
            logger.info(f"✨ Sync added {added_count} tasks (Recovered or New).")
        else:
            logger.info("✅ Verification complete. No missing tasks found.")

        # 更新总统计数据 (基于原始 manifest 计算，确保精确)
        tasks['total_num'] = len(all_podcasts)
        tasks['total_hour'] = sum(ep['audio_duration_second'] for ep in all_podcasts) / 3600
        
    return tasks