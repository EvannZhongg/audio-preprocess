import json
import os
import subprocess

import numpy as np

from pipeline.global_var import PipelineParam
from ray_task.config import (CPU_PER_TASK_CPU, FFMPEG_TIME_OUT,
                             MAX_AUDIO_DURATION_SECONDS)
from utils.logger import time_logger


@time_logger
def standardization(audio_path, num_threads=CPU_PER_TASK_CPU, timeout=FFMPEG_TIME_OUT):

    logger = PipelineParam.logger
    cfg = PipelineParam.cfg
    target_sample_rate = cfg["entrypoint"]["SAMPLE_RATE"]
    target_dBFS = -20
    
    name = os.path.basename(audio_path)
    
    dynamic_timeout = timeout
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", audio_path]
        probe_out = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT, timeout=5)
        duration_info = json.loads(probe_out)
        duration_sec = float(duration_info['format']['duration'])
        
        #  OOM 熔断机制：超过最大时长直接跳过
        if duration_sec > MAX_AUDIO_DURATION_SECONDS:
            logger.warning(f"SKIP Huge Audio (>{MAX_AUDIO_DURATION_SECONDS}): {name} ({duration_sec:.1f}s)")
            return None
            
        calc_timeout = 60 + int(duration_sec / 24)
        dynamic_timeout = max(timeout, calc_timeout)
    
    except subprocess.TimeoutExpired:
        logger.error(f"IO Hang detected during probe: {name} took too long. Skipping.")
        return None
        
    except Exception as e:
        # 如果 ffprobe 失败 (如文件头损坏)，交给后续 ffmpeg 尝试处理，使用默认超时
        logger.debug(f"Probe failed for {name}, using default logic: {e}")
        dynamic_timeout = timeout
    # ===============================================================

    cmd = [
        "ffmpeg",
        "-threads", str(num_threads),         
        "-i", audio_path,
        "-ar", str(target_sample_rate),
        "-ac", "1",
        "-f", "s16le",         
        "-acodec", "pcm_s16le",  
        "-loglevel", "error",
        "-"
    ]
    
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        
        try:
            # 使用动态计算的超时时间
            raw_data, stderr = proc.communicate(timeout=dynamic_timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT ({dynamic_timeout}s): Killing process for {name}")
            proc.kill() 
            proc.communicate() 
            return None

        if proc.returncode != 0:
            logger.error(f"FFmpeg Error for {name}: {stderr.decode()}")
            return None
        
        if not raw_data:
            logger.warning(f"Empty output for {name}")
            return None

        # --- 内存管理关键区 ---
        waveform_int16 = np.frombuffer(raw_data, dtype=np.int16)
        del raw_data  # 立即释放原始字节流
        
        waveform = waveform_int16.astype(np.float32) / 32768.0
        del waveform_int16 # 立即释放
        # --------------------
        
        duration = len(waveform) / target_sample_rate
 
        # 优化 RMS 计算，避免生成中间大数组
        mean_square = np.mean(np.square(waveform))
        rms = np.sqrt(mean_square)

        if rms > 0:
            current_dBFS = 20 * np.log10(rms)
        else:
            current_dBFS = -float('inf')
            
        gain_db = target_dBFS - current_dBFS
        limited_gain_db = min(max(gain_db, -3), 3)
        gain_factor = 10 ** (limited_gain_db / 20)
        
        # In-place 乘法，稍微节省一点内存
        waveform *= gain_factor
        
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude > 1.0:
            waveform /= max_amplitude

        return {
            "waveform": waveform,
            "name": name,
            "sample_rate": target_sample_rate,
            "duration": duration,
            "original_path": audio_path
        }

    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
        logger.error(f"Exception processing {name}: {e}")
        return None
