import os
import subprocess

import numpy as np

from pipeline.global_var import PipelineParam
from utils.logger import time_logger


@time_logger
def standardization(audio_path):
    logger = PipelineParam.logger
    cfg = PipelineParam.cfg
    target_sample_rate = cfg["entrypoint"]["SAMPLE_RATE"]
    target_dBFS = -20
    

    FFMPEG_TIMEOUT = 900  # 10分钟
    
    name = os.path.basename(audio_path)
    
    cmd = [
        "ffmpeg",
        "-i", audio_path,
        "-ar", str(target_sample_rate),
        "-ac", "1",
        "-f", "s16le",         
        "-acodec", "pcm_s16le",  
        "-loglevel", "error",
        "-"
    ]
    
    logger.info(f"Stream processing via FFmpeg (Raw PCM): {name}")
    
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        
        try:
            raw_data, stderr = proc.communicate(timeout=FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timed out after {FFMPEG_TIMEOUT}s: {name}")
            proc.kill() 
            proc.communicate() 
            raise RuntimeError(f"FFmpeg timeout processing {name}")

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
        
        if not raw_data:
            logger.warning("Empty audio output from FFmpeg.")
            return {"waveform": np.array([], dtype=np.float32), "name": name, "sample_rate": target_sample_rate, "duration": 0}

        waveform_int16 = np.frombuffer(raw_data, dtype=np.int16)
        waveform = waveform_int16.astype(np.float32) / 32768.0
        duration = len(waveform) / target_sample_rate
 
        rms = np.sqrt(np.mean(waveform**2))
        if rms > 0:
            current_dBFS = 20 * np.log10(rms)
        else:
            current_dBFS = -float('inf')
            
        gain_db = target_dBFS - current_dBFS
        limited_gain_db = min(max(gain_db, -3), 3)
        gain_factor = 10 ** (limited_gain_db / 20)
        
        waveform = waveform * gain_factor
        
        max_amplitude = np.max(np.abs(waveform))
        if max_amplitude > 1.0:
            waveform /= max_amplitude

        return {
            "waveform": waveform,
            "name": name,
            "sample_rate": target_sample_rate,
            "duration": duration
        }

    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
        logger.error(f"Error processing {audio_path}: {e}")
        raise e
