import librosa
import tqdm
import numpy as np

from pipeline.global_var import PipelineParam
from utils.tool import calculate_audio_stats
from utils.logger import time_logger

logger = PipelineParam.logger

@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        tuple: A tuple containing the average MOS and the updated VAD segments with MOS scores.
    """
    cfg = PipelineParam.cfg
    dnsmos_compute_score = PipelineParam.dnsmos_compute_score

    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")

    return predict_dnsmos, vad_list


def filter_by_mos(mos_list, mos_filter_cfg):
    """
    Filter out segments based on a configurable MOS strategy, followed by other quality checks.

    Args:
        mos_list (list): List of VAD segments with MOS scores.
        mos_filter_cfg (dict): Configuration for MOS filtering.

    Returns:
        list: A list of VAD segments that passed all filtering stages.
    """
    # 检查输入是否为空
    if not mos_list:
        logger.warning("No segments to filter - mos_list is empty")
        return []

    # --- Step 1: Filter by MOS score based on the chosen strategy ---
    strategy = mos_filter_cfg.get("strategy", "average")
    list_after_mos_filter = []

    if strategy == "fixed":
        threshold = mos_filter_cfg.get("fixed_threshold", 3.0)
        logger.info(f"Filtering with fixed MOS threshold: >={threshold}")
        list_after_mos_filter = [seg for seg in mos_list if seg.get('dnsmos', 0) >= threshold]
    else:  # "average" strategy (default)
        if not mos_list:
            return []
        threshold = np.mean([vad["dnsmos"] for vad in mos_list])
        logger.info(f"Filtering with average MOS threshold: >={threshold:.2f}")
        list_after_mos_filter = [seg for seg in mos_list if seg.get('dnsmos', 0) >= threshold]
    
    logger.info(f"MOS Filter: {len(mos_list) - len(list_after_mos_filter)} segments removed.")

    # 如果没有任何段通过MOS过滤，提前返回
    if not list_after_mos_filter:
        logger.warning("No segments passed the MOS filtering stage.")
        return []

    # --- Step 2: Perform other quality checks (e.g., char duration) ---
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(list_after_mos_filter)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    
    if all_segment == 0:
        logger.warning("No valid segments found after secondary quality checks (calculate_audio_stats)")
        return []
    
    filter_percentage = (all_segment - filtered_segment) / all_segment
    logger.debug(
        f"> Secondary filters (char rate, etc.) removed: {all_segment - filtered_segment}/{all_segment} ({filter_percentage:.2%}) segments."
    )
    
    final_filtered_list = [list_after_mos_filter[idx] for idx, _ in filtered_audio_stats]
    
    if not final_filtered_list:
        logger.warning("All segments were filtered out by secondary quality checks.")
    
    return final_filtered_list