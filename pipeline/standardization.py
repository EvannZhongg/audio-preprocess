import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pydub import AudioSegment

from pipeline.global_var import PipelineParam
from utils.logger import Logger, time_logger

logger = Logger.get_logger(__name__)

audio_count = 0

CHUNK_DURATION_MS = 5000 #  5s


def standardize_chunk(audio_segment: AudioSegment, limited_gain: float, target_sample_rate: int, chunk_index: int):
    """
    Worker function to standardize a single audio chunk (关键修复：移除块内归一化)
    """
    chunk = (
        audio_segment
        .set_frame_rate(target_sample_rate)
        .set_sample_width(2)  # 16-bit
        .set_channels(1)      # 单声道
    )
    
    normalized_chunk = chunk.apply_gain(limited_gain)
    waveform = np.array(normalized_chunk.get_array_of_samples(), dtype=np.float32)
    return chunk_index, waveform


@time_logger
def standardization(audio, max_workers=4):

    global audio_count
    
    cfg = PipelineParam.cfg
    target_sample_rate = cfg["entrypoint"]["SAMPLE_RATE"]
    target_dBFS = -20
    name = "audio"

    # --- 1. 加载整个音频并计算全局增益 (串行步骤) ---
    if isinstance(audio, str):
        name = os.path.basename(audio)
        full_audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
        full_audio = audio
    else:
        raise ValueError("Invalid audio type")
        
    logger.debug("Starting gain calculation for full audio...")
    
    gain = target_dBFS - full_audio.dBFS
    limited_gain = min(max(gain, -3), 3)
    logger.info(f"Global gain calculated: {limited_gain:.2f} dB (target: -20dBFS)")

    # --- 2. 分割音频并提交并行任务 ---
    chunks = []
    total_duration = len(full_audio)  # 总长度（毫秒）
    
    # 分割音频为固定时长的块
    for i in range(0, total_duration, CHUNK_DURATION_MS):
        chunk = full_audio[i:i + CHUNK_DURATION_MS]
        chunks.append(chunk)
    
    logger.debug(f"Split audio into {len(chunks)} chunks ({CHUNK_DURATION_MS}ms each)")

    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                standardize_chunk, 
                chunk, 
                limited_gain, 
                target_sample_rate, 
                i
            ) for i, chunk in enumerate(chunks)
        }
        
        for future in as_completed(futures):
            try:
                idx, waveform_chunk = future.result()
                results[idx] = waveform_chunk
            except Exception as e:
                logger.error(f"Chunk {idx} processing failed: {str(e)}")
    
    if not results:
        logger.warning("No chunks were successfully processed. Using empty waveform.")
        final_waveform = np.array([], dtype=np.float32)
    else:
        sorted_waveforms = [results[i] for i in sorted(results.keys())]
        final_waveform = np.concatenate(sorted_waveforms)
        logger.debug(f"Successfully merged {len(sorted_waveforms)} chunks into final waveform")

    max_amplitude = np.max(np.abs(final_waveform))
    if max_amplitude > 0:
        final_waveform /= max_amplitude
        logger.info(f"Global normalization applied: max amplitude set to 1.0 (was {max_amplitude:.4f})")
    else:
        logger.warning("Audio is completely silent, skipping normalization.")

    logger.debug(f"Final waveform shape: {final_waveform.shape}")
    logger.debug(f"Final waveform dtype: {final_waveform.dtype}")

    return {
        "waveform": final_waveform,
        "name": name,
        "sample_rate": target_sample_rate,
        "duration": full_audio.duration_seconds
    }
