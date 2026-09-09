import librosa
import numpy as np
import tqdm

from utils.logger import time_logger
from utils.tool import calculate_audio_stats


@time_logger
def metrics_prediction(audio, vad_list, metrics_filter_cfg):
    """
    Predict the audio quality scores for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.
        metrics_filter_cfg (dict): Configuration for metrics filtering.

    Returns:
        tuple: A tuple containing the average audio quality scores and the updated VAD segments with audio quality scores.
    """
    
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger
    
    cfg = PipelineParam.cfg
    dnsmos_compute_score = PipelineParam.dnsmos_compute_score
    brouhaha_metric = PipelineParam.brouhaha_metric

    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="METRICS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]
        vad_list[index]["dnsmos"] = dnsmos
        if brouhaha_metric is not None:
            c50, snr = brouhaha_metric(segment, sample_rate)
            vad_list[index]["c50"] = c50
            vad_list[index]["snr"] = snr
        else:
            vad_list[index]["c50"] = metrics_filter_cfg.get("fixed_c50_threshold", 40.0)
            vad_list[index]["snr"] = metrics_filter_cfg.get("fixed_snr_threshold", 40.0)

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])
    predict_c50 = np.mean([vad["c50"] for vad in vad_list])
    predict_snr = np.mean([vad["snr"] for vad in vad_list])
    logger.debug(f"avg predict_metrics for whole audio: {predict_dnsmos:.2f}, {predict_c50:.2f}, {predict_snr:.2f}")
    return (predict_dnsmos, predict_c50, predict_snr), vad_list



def filter_by_metrics(metrics_list, metrics_filter_cfg):
    """
    Filter out segments based on a configurable audio quality strategy, followed by other quality checks.

    Args:
        mos_list (list): List of VAD segments with audio quality scores.
        mos_filter_cfg (dict): Configuration for MOS filtering.

    Returns:
        list: A list of VAD segments that passed all filtering stages.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger
    # 检查输入是否为空
    if not metrics_list:
        logger.warning("No segments to filter - metrics_list is empty")
        return []

    # --- Step 1: Filter by metrics score based on the chosen strategy ---
    strategy = metrics_filter_cfg.get("strategy", "average")

    if strategy == "fixed":
        dnsmos_threshold = metrics_filter_cfg.get("fixed_dnsmos_threshold", 3.0)
        c50_threshold = metrics_filter_cfg.get("fixed_c50_threshold", 40.0)
        snr_threshold = metrics_filter_cfg.get("fixed_snr_threshold", 30.0)
        logger.info(f"Filtering with fixed metric threshold: dnsmos>={dnsmos_threshold}, c50>={c50_threshold}, snr>={snr_threshold}")
        list_after_metrics_filter = [seg for seg in metrics_list if seg.get('dnsmos', 3.0) >= dnsmos_threshold and seg.get('c50', 40.0) >= c50_threshold and seg.get('snr', 30.0) >= snr_threshold]
    else:  # "average" strategy (default)
        if not metrics_list:
            return []
        threshold = np.mean([vad["dnsmos"] for vad in metrics_list])
        logger.info(f"Filtering with average MOS threshold: >={threshold:.2f}")
        list_after_metrics_filter = [seg for seg in metrics_list if seg.get('dnsmos', 0) >= threshold]
    
    logger.info(f"Metrics Filter: {len(metrics_list) - len(list_after_metrics_filter)} segments removed.")

    # 如果没有任何段通过MOS过滤，提前返回
    if not list_after_metrics_filter:
        logger.warning("No segments passed the Metrics filtering stage.")
        return []

    # --- Step 2: Perform other quality checks (e.g., char duration) ---
    filtered_audio_stats, all_audio_stats, avg_char_durations = calculate_audio_stats(list_after_metrics_filter, metrics_filter_cfg)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    
    if all_segment == 0:
        logger.warning("No valid segments found after secondary quality checks (calculate_audio_stats)")
        return []
    
    filter_percentage = (all_segment - filtered_segment) / all_segment
    logger.debug(
        f"> Secondary filters (char rate, etc.) removed: {all_segment - filtered_segment}/{all_segment} ({filter_percentage:.2%}) segments."
    )
    
    final_filtered_list = []
    for idx_dur, avg_char_duration in zip(filtered_audio_stats, avg_char_durations):
        valid_segment = list_after_metrics_filter[idx_dur[0]]
        valid_segment['avg_char_duration'] = avg_char_duration
        final_filtered_list.append(valid_segment)
    
    if not final_filtered_list:
        logger.warning("All segments were filtered out by secondary quality checks.")
    
    return final_filtered_list
