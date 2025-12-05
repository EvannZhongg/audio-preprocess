import os
import traceback
from dataclasses import dataclass
from pathlib import Path

# 假设这些配置和工具函数都已正确导入
from ray_task.config import REPORT_PATH 
from utils.tool import load_cfg
from pipeline import global_var
from pipeline.main_process import main_process # 假设 main_process 在这里导入

from utils.logger import Logger
logger = Logger.get_logger(f"ray-task")


def get_audio_manifest(audio_path, base_dir):   
    """为单个音频文件生成 manifest entry."""
    # 注意：这里的 manifest entry 结构必须与你的 main_process 期望的结构一致
    relative_path = os.path.relpath(os.path.dirname(audio_path), base_dir)
    return {
        "RelativePath": relative_path,
        "FilePath": audio_path
    }

@dataclass
class TaskCmdArgs:
    batch_size: int = 8
    compute_type: str = 'float16'
    whisper_arch: str = 'medium'
    threads: int = 4


def run_audio_preprocess_pipeline(config_path, task_batch, prefix_path, output_dir):
    """
    运行音频预处理管线，适配 Batching 机制。
    
    参数:
        config_path (str): 配置文件路径
        task_batch (list): 任务列表，每个元素是一个任务字典，包含 "audio_path"
        prefix_path (str): 音频路径的前缀（用于计算相对路径）
        output_dir (str): 输出根目录
        
    返回:
        str: "SUCCESS" 或 "FAILURE" (用于 Ray 调度层判断，但实际状态管理在 handle_task 中)
    """
    
    # 只需要在 Ray 任务启动时初始化一次全局配置
    try:
        main_cfg = load_cfg(config_path)
        # 注意: 这里的 cli_args.batch_size=8 是指模型内部的 Batch Size，与 Ray 的任务 Batch Size 不是一回事
        cli_args = TaskCmdArgs(batch_size=8, compute_type='float16', threads=4)
        logger.info(f"pipeline config={main_cfg} cli_args={cli_args}")
        global_var.init_pipeline_global(main_cfg, cli_args)
    
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processed data will be saved in: {output_dir}")
        
    except Exception:
        logger.error(f"Global pipeline initialization failed: {traceback.format_exc()}")
        # 如果初始化失败，则整个 Batch 无法处理，返回 FAILURE
        return "FAILURE_INIT"


    # 循环处理 Batch 中的每一个子任务
    for task in task_batch:
        input_audio_path = task["audio_path"]
        task_key = get_task_key(task) # 假设 get_task_key 函数在外部可用

        try:
            # 1. 为当前音频生成 Manifest
            manifest_entry = get_audio_manifest(input_audio_path, prefix_path)
            
            # 2. 调用核心处理函数
            # 注意: main_process 必须能够正确处理单个文件，并且其输出目录应该基于 relative_path 隔离
            # 如果 main_process 内部能够处理 relative_path，则 output_dir 可以是根目录
            main_process(manifest_entry, output_dir, "processing_report.csv")
            
            logger.info(f"Task {task_key} processed successfully within the Batch.")
            
            # 假设处理成功后，在 task 字典中加入成功标记 (可选, Ray 调度层主要看返回值)
            task["pipeline_status"] = "SUCCESS" 

        except Exception as e:
            logger.error(f"Error processing single task {task_key} in Batch: {traceback.format_exc()}")
            # 记录失败状态，但不中断整个 Batch 的处理
            task["pipeline_status"] = "FAILURE" 
            # 继续处理 Batch 中的下一个任务


    logger.info("--- All files in Batch have been submitted to main_process. ---")
    
    # 返回 SUCCESS 告诉 Ray 调度层：Batch 任务本身执行完成（至于 Batch 内部任务的成功失败，由 handle_task 负责解析）
    return "SUCCESS"
