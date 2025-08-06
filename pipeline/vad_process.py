import librosa
import torch
from pipeline.global_var import PipelineParam
from utils.logger import time_logger

logger = PipelineParam.logger

@time_logger
def refine_vad_list_by_embedding(
    vad_list, audio, refinement_model, feature_extractor, device
):
    """
    Args:
        vad_list (list): 从 vad.vad() 得到的原始VAD切片列表。
        audio (dict): 音频数据。
        refinement_model: 用于优化的ERes2NetV2模型。
        feature_extractor: 模型的FBank特征提取器。
        device: 运行模型的torch设备 (CPU或GPU)。

    Returns:
        list: 经过筛选后，逻辑与旧版本一致的VAD切片新列表。
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from torch.nn.utils.rnn import pad_sequence

    refined_vad_list = []
    MIN_SEGMENT_DURATION_S = 1.0
    WINDOW_SIZE_S = 1.1
    WINDOW_STEP_S = 0.4
    SIMILARITY_THRESHOLD = 0.6
    MAX_REFINEMENT_BATCH_SIZE = 64 

    def _get_embedding_single(waveform_segment):
        if len(waveform_segment) / audio["sample_rate"] < 0.1:
            return None
        
        waveform_16k = librosa.resample(
            waveform_segment, orig_sr=audio["sample_rate"], target_sr=16000
        )
        
        features = feature_extractor(torch.tensor(waveform_16k, dtype=torch.float32).to(device))
        with torch.no_grad():
            # 使用 unsqueeze(0) 创建一个 batch_size=1 的批次
            embedding = refinement_model(features.unsqueeze(0)).cpu().numpy()
        return embedding

    def _get_embeddings_batched(waveforms):
        if not waveforms:
            return np.array([])

        all_embeddings = []
        for i in range(0, len(waveforms), MAX_REFINEMENT_BATCH_SIZE):
            batch_waveforms = waveforms[i:i + MAX_REFINEMENT_BATCH_SIZE]
            
            waveforms_16k = [
                librosa.resample(wf, orig_sr=audio["sample_rate"], target_sr=16000)
                for wf in batch_waveforms
            ]

            feature_tensors = [
                feature_extractor(torch.tensor(wf, dtype=torch.float32).to(device))
                for wf in waveforms_16k
            ]
            
            padded_features = pad_sequence(feature_tensors, batch_first=True, padding_value=0.0)

            with torch.no_grad():
                embeddings_batch = refinement_model(padded_features).cpu().numpy()
                all_embeddings.append(embeddings_batch)
        
        return np.vstack(all_embeddings)

    for segment in vad_list:
        duration = segment["end"] - segment["start"]
        if duration < MIN_SEGMENT_DURATION_S:
            # 对于太短的 segment，设置默认相似度值
            segment_with_similarity = segment.copy()
            segment_with_similarity["min_similarity"] = 0.61  # 默认相似度
            refined_vad_list.append(segment_with_similarity)
            continue

        start_frame_main = int(segment["start"] * audio["sample_rate"])
        end_frame_main = int(segment["end"] * audio["sample_rate"])
        segment_waveform = audio["waveform"][start_frame_main:end_frame_main]

        # 1. 单独计算参考嵌入，确保逻辑与旧版一致
        reference_embedding = _get_embedding_single(segment_waveform)
        if reference_embedding is None:
            # 如果整个片段无法获取embedding，则直接保留，设置默认相似度
            segment_with_similarity = segment.copy()
            segment_with_similarity["min_similarity"] = 0.61  # 默认相似度
            refined_vad_list.append(segment_with_similarity)
            continue

        # 2. 收集所有窗口的波形用于批处理
        window_waveforms = []
        window_start_s = 0
        while window_start_s + WINDOW_SIZE_S <= duration:
            window_start_frame = int(window_start_s * audio["sample_rate"])
            window_end_frame = int((window_start_s + WINDOW_SIZE_S) * audio["sample_rate"])
            window_waveform = segment_waveform[window_start_frame:window_end_frame]
            
            # 同样进行时长检查
            if len(window_waveform) / audio["sample_rate"] >= 0.1:
                window_waveforms.append(window_waveform)

            window_start_s += WINDOW_STEP_S
        
        # 如果没有有效的窗口，则直接保留原片段，设置默认相似度
        if not window_waveforms:
            segment_with_similarity = segment.copy()
            segment_with_similarity["min_similarity"] = 0.61  # 默认相似度
            refined_vad_list.append(segment_with_similarity)
            continue

        # 3. 对所有窗口进行批处理计算
        window_embeddings = _get_embeddings_batched(window_waveforms)
        
        # 4. 逐一比较，记录最小相似度
        is_consistent = True
        min_similarity = float('inf')
        
        for window_embedding in window_embeddings:
            similarity = cosine_similarity(reference_embedding, window_embedding.reshape(1, -1))[0, 0]
            min_similarity = min(min_similarity, similarity)

            if similarity < SIMILARITY_THRESHOLD:
                is_consistent = False
                logger.debug(
                    f"Discarding VAD segment from {segment['start']:.2f}s to {segment['end']:.2f}s "
                    f"due to internal inconsistency. Similarity: {similarity:.2f}"
                )
                break 
        
        if is_consistent:
            # 添加最小相似度到 segment 中
            segment_with_similarity = segment.copy()
            segment_with_similarity["min_similarity"] = float(min_similarity)
            refined_vad_list.append(segment_with_similarity)

    return refined_vad_list


@time_logger
def cut_by_speaker_label(vad_list, audio_duration, stats, step_name="post_process_vad"):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.
    Also adds a grace period to the end of segments to reduce cut-offs.
    This function now internally tracks and updates statistics.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.
        audio_duration (float): Total duration of the audio in seconds.
        stats (dict): The main statistics dictionary to be updated.
        step_name (str): The name of the step for statistics tracking.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds
    GRACE_PERIOD_START_S = 0.00
    GRACE_PERIOD_END_S = 0.02
    updated_list = []

    # --- Internal Statistics ---
    discarded_long_count = 0
    discarded_long_duration = 0.0

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            duration = vad["end"] - vad["start"]
            logger.warning(

                f"cut_by_speaker_label > Discarding segment for speaker {vad['speaker']} "
                f"because its duration ({duration:.2f}s) is longer than "
                f"MAX_SEGMENT_LENGTH ({MAX_SEGMENT_LENGTH}s)."
            )
            # Track discard due to max length
            discarded_long_count += 1
            discarded_long_duration += duration
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    # Calculate discards from the final length filtering
    count_before_min_len_filter = len(updated_list)
    duration_before_min_len_filter = sum(s["end"] - s["start"] for s in updated_list)

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]
    
    count_after_min_len_filter = len(filter_list)
    duration_after_min_len_filter = sum(s["end"] - s["start"] for s in filter_list)

    discarded_short_count = count_before_min_len_filter - count_after_min_len_filter
    discarded_short_duration = duration_before_min_len_filter - duration_after_min_len_filter

    logger.debug(
        f"cut_by_speaker_label > removed: {discarded_short_count} segments by length"
    )

    # Update the main statistics dictionary
    stats['steps'][step_name]['discarded_count'] = discarded_long_count + discarded_short_count
    stats['steps'][step_name]['discarded_duration'] = discarded_long_duration + discarded_short_duration

    # --- Add Grace Period Logic ---
    if not filter_list:
        return filter_list

    logger.debug(
        f"cut_by_speaker_label > Applying {GRACE_PERIOD_START_S}s grace period to segment starts and {GRACE_PERIOD_END_S}s to ends."
    )

    # First, handle the end times to avoid overlap with the *next* segment
    # Iterate up to the second to last segment
    for i in range(len(filter_list) - 1):
        current_segment = filter_list[i]
        next_segment_start = filter_list[i + 1]["start"]

        # Add grace period, ensuring it doesn't extend into the next segment
        new_end = current_segment["end"] + GRACE_PERIOD_END_S
        current_segment["end"] = min(new_end, next_segment_start)

    # Handle the last segment's end, ensuring it doesn't extend beyond the total audio duration
    last_segment = filter_list[-1]
    new_end = last_segment["end"] + GRACE_PERIOD_END_S
    last_segment["end"] = min(new_end, audio_duration)

    # Second, handle the start times to avoid overlap with the *previous* segment
    # Handle the first segment's start time, ensuring it doesn't go below zero
    first_segment = filter_list[0]
    new_start = first_segment["start"] - GRACE_PERIOD_START_S
    first_segment["start"] = max(0.0, new_start)

    # Iterate from the second segment onwards
    for i in range(1, len(filter_list)):
        current_segment = filter_list[i]
        previous_segment_end = filter_list[i - 1]["end"]

        # Subtract grace period, ensuring it doesn't overlap with the previous segment
        new_start = current_segment["start"] - GRACE_PERIOD_START_S
        current_segment["start"] = max(new_start, previous_segment_end)

    return filter_list
