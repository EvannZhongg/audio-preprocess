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
    # 1. 加载文件 (主文件 -> 备份文件 -> 失败则初始化)
    # ==============================================================================
    
    # 尝试加载主文件
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            loaded = True
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Main task file is corrupted: {file_path}")
    
    # 尝试加载备份文件
    if not loaded and os.path.exists(TASK_RESULT_BACKUP_FILE):
        try:
            logger.info(f"🔄 Attempting to restore from backup: {TASK_RESULT_BACKUP_FILE}")
            with open(TASK_RESULT_BACKUP_FILE, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            shutil.copy(TASK_RESULT_BACKUP_FILE, file_path)
            loaded = True
            logger.info("✅ Restored from backup successfully.")
        except Exception as e:
            logger.error(f"❌ Backup file is also corrupted: {e}")

    # ==============================================================================
    # 2. 状态恢复与清理 (Re-queue Processing)
    # ==============================================================================

    if loaded:
        # 确保基本结构存在
        for key in ['todo', 'complete', 'failed', 'processing']:
            if key not in tasks: tasks[key] = [] if key != 'processing' else {}

        # A. 把上次中断的 'processing' 任务全部扔回 'todo'
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
    # 3. [关键] 账目核对与补全 (Sync with Manifest)
    # ==============================================================================
    # 解决 "todo" 为空但任务没跑完的问题
    
    logger.info("🔄 Syncing with original manifest to ensure no tasks are lost...")
    
    try:
        with open(PODCAST_DATA_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            all_podcasts = raw_data.get('podcast_data', [])
    except Exception as e:
        logger.error(f"Failed to load raw manifest: {e}")
        all_podcasts = []

    if all_podcasts:
        # 建立已存在任务的索引 (使用 relative_path 作为唯一键)
        # 注意：这里我们只关心 relative_path 就能区分任务
        existing_paths = set()
        
        # 记录已完成的
        for t in tasks.get('complete', []): existing_paths.add(t['relative_path'])
        # 记录失败的
        for t in tasks.get('failed', []): existing_paths.add(t['relative_path'])
        # 记录已经在 todo 里的
        for t in tasks.get('todo', []): existing_paths.add(t['relative_path'])
        
        recovered_count = 0
        new_todo = []
        
        # 遍历原始总表
        for podcast in all_podcasts:
            r_path = podcast['relative_path']
            
            # 如果这个任务不在 (完成 + 失败 + 待办) 里，说明它丢了
            if r_path not in existing_paths:
                task_item = {
                    "relative_path": r_path,
                    "audio_path": podcast['audio_path'],
                    "audio_duration_second": podcast['audio_duration_second']
                }
                tasks['todo'].append(task_item)
                existing_paths.add(r_path) # 防止重复添加
                recovered_count += 1

        if recovered_count > 0:
            logger.info(f"✨ Recovered {recovered_count} tasks that were missing from JSON!")
        else:
            logger.info("✅ Verification complete. No missing tasks found.")

        # 更新统计数据
        tasks['total_num'] = len(all_podcasts)
        tasks['total_hour'] = sum(ep['audio_duration_second'] for ep in all_podcasts) / 3600
        
        # 可选：重新清洗 todo 格式
        cleaned_todo = [t for t in tasks['todo'] if isinstance(t, dict) and "relative_path" in t]
        tasks['todo'] = cleaned_todo

    return tasks