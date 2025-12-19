import json
import os
import re
import shutil
from pathlib import Path

# 引入配置
from ray_task.config import OUTPUT_ROOT_DIR, PODCAST_DATA_FILE, TASK_RESULT_BACKUP_FILE
from utils.logger import Logger

logger = Logger.get_logger()

def load_tasks(file_path):
    tasks = {}
    loaded = False

    # ==============================================================================
    # 1. 加载历史记录 (主文件 -> 备份文件)
    # ==============================================================================
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            loaded = True
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Main task file is corrupted: {file_path}")
    
    if not loaded and os.path.exists(TASK_RESULT_BACKUP_FILE):
        try:
            logger.info(f"🔄 Attempting to restore from backup: {TASK_RESULT_BACKUP_FILE}")
            with open(TASK_RESULT_BACKUP_FILE, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            shutil.copy(TASK_RESULT_BACKUP_FILE, file_path)
            loaded = True
            logger.info("✅ Restored from backup successfully.")
        except Exception:
            logger.error("❌ Backup file is also corrupted or missing.")

    # ==============================================================================
    # 2. 结构初始化 (兼容首次运行)
    # ==============================================================================
    # 无论是否加载成功，都必须确保字典结构完整，防止 KeyError
    
    keys_to_ensure = ['todo', 'complete', 'failed', 'processing']
    for key in keys_to_ensure:
        if key not in tasks:
            # processing 是字典，其他是列表
            tasks[key] = {} if key == 'processing' else []

    # 如果是第一次运行，或者文件全坏了，初始化统计数据
    if not loaded:
        logger.info("🆕 No history found (First Run). Initializing structure...")
        tasks['total_num'] = 0
        tasks['total_hour'] = 0
        tasks['complete_num'] = 0
        tasks['complete_total_hour'] = 0
        tasks['failed_num'] = 0
        tasks['failed_total_hour'] = 0

    # ==============================================================================
    # 3. 状态恢复 (把中断的任务捞回来)
    # ==============================================================================

    if tasks.get("processing"):
        interrupted_tasks = tasks["processing"]
        logger.info(f"⚠️ Found {len(interrupted_tasks)} interrupted tasks. Re-queueing...")
        
        for k, task_info in interrupted_tasks.items():
            # 清理脏目录
            try:
                relative_path = task_info.get("relative_path")
                audio_path = task_info.get("audio_path")
                if relative_path and audio_path:
                    fid = re.sub(r"['\"\s]", "", Path(audio_path).stem)
                    processing_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path, fid)
                    if os.path.exists(processing_dir):
                        shutil.rmtree(processing_dir)
            except Exception:
                pass
            
            # 放回待办列表
            tasks['todo'].append(task_info)
        
        tasks["processing"] = {} # 清空进行中状态

    # ==============================================================================
    # 4. [全能同步] 账目核对与任务生成 (Sync)
    # ==============================================================================
    # 第一次运行时：这里会读取 manifest 并生成所有任务放入 todo
    # 断点续传时：这里会检查是否有新文件，或者找回丢失的任务
    
    logger.info("🔄 Syncing with original manifest...")
    
    try:
        if not os.path.exists(PODCAST_DATA_FILE):
             # 如果连原始数据都没有，抛出异常
             raise FileNotFoundError(f"Source data file missing: {PODCAST_DATA_FILE}")

        with open(PODCAST_DATA_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            all_podcasts = raw_data.get('podcast_data', [])
    except Exception as e:
        logger.error(f"Failed to load raw manifest: {e}")
        all_podcasts = []

    if all_podcasts:
        # 建立索引，避免重复添加
        existing_paths = set()
        
        for t in tasks['complete']: existing_paths.add(t['relative_path'])
        for t in tasks['failed']: existing_paths.add(t['relative_path'])
        for t in tasks['todo']: existing_paths.add(t['relative_path'])
        
        added_count = 0
        
        for podcast in all_podcasts:
            r_path = podcast['relative_path']
            
            # 只要不在 (完成/失败/待办) 里，就加进去
            if r_path not in existing_paths:
                task_item = {
                    "relative_path": r_path,
                    "audio_path": podcast['audio_path'],
                    "audio_duration_second": podcast['audio_duration_second']
                }
                tasks['todo'].append(task_item)
                existing_paths.add(r_path)
                added_count += 1

        if added_count > 0:
            logger.info(f"✨ Added {added_count} tasks (First run or Recovery).")
        else:
            logger.info("✅ All tasks are accounted for.")

        # 更新总统计
        tasks['total_num'] = len(all_podcasts)
        tasks['total_hour'] = sum(ep['audio_duration_second'] for ep in all_podcasts) / 3600

    return tasks