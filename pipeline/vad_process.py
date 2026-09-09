import librosa
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence

from utils.logger import time_logger


@time_logger
def refine_vad_list_by_embedding(
    vad_list, audio, refinement_model, similarity_threshold, refinement_batch_size, feature_extractor, device
):
    """
    Args:
        vad_list (list): 从 vad.vad() 得到的原始VAD切片列表。
        audio (dict): 音频数据。
        refinement_model: 用于优化的ERes2NetV2模型。
        similarity_threshold: 相似度阈值。
        feature_extractor: 模型的FBank特征提取器。
        device: 运行模型的torch设备 (CPU或GPU)。

    Returns:
        list: 经过筛选后，逻辑与旧版本一致的VAD切片新列表。
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

    refined_vad_list = []
    MIN_SEGMENT_DURATION_S = 1.0
    WINDOW_SIZE_S = 1.1
    WINDOW_STEP_S = 0.4
    SIMILARITY_THRESHOLD = similarity_threshold
    MAX_REFINEMENT_BATCH_SIZE = refinement_batch_size

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
            segment_with_similarity["min_similarity"] = similarity_threshold
            segment_with_similarity["reference_embedding"] = None
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
            segment_with_similarity["min_similarity"] = similarity_threshold
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
            segment_with_similarity["min_similarity"] = similarity_threshold 
            segment_with_similarity["reference_embedding"] = None
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
            segment_with_similarity["reference_embedding"] = reference_embedding
            refined_vad_list.append(segment_with_similarity)

    return refined_vad_list


def _split_long_span_at_silences(vad, audio, vad_model, max_len, logger):
    """Split an overlong VAD span at silero-detected internal silences.

    Strategy:
      1. Run silero-vad on the audio slice belonging to this span.
      2. Compute internal gaps between consecutive speech intervals.
      3. Pick the K longest gaps so each resulting chunk is < max_len,
         and split at the midpoint of each chosen gap.
      4. If silero returns < 2 speech intervals (no internal silence),
         return None — caller should drop the segment.

    Returns:
        list[dict] of sub-segments (copies of `vad` with adjusted start/end),
        or None if no valid cut points exist.
    """
    if audio is None or vad_model is None:
        return None

    waveform = audio.get("waveform")
    src_sr = audio.get("sample_rate")
    if waveform is None or src_sr is None:
        return None

    seg_start = float(vad["start"])
    seg_end = float(vad["end"])
    duration = seg_end - seg_start
    if duration < max_len:
        return [vad]

    # Slice and resample to 16k for silero
    s_idx = max(0, int(seg_start * src_sr))
    e_idx = min(len(waveform), int(seg_end * src_sr))
    if e_idx <= s_idx:
        return None
    seg_audio = waveform[s_idx:e_idx]
    if seg_audio.ndim > 1:
        seg_audio = seg_audio.mean(axis=0)
    if src_sr != 16000:
        seg_audio_16k = librosa.resample(seg_audio, orig_sr=src_sr, target_sr=16000)
    else:
        seg_audio_16k = seg_audio

    try:
        intervals = vad_model._get_speech_timestamps_wrapper(seg_audio_16k, 16000)
    except Exception as e:
        logger.warning(f"silero re-detect failed on long span: {e}")
        return None

    if not intervals or len(intervals) < 2:
        return None

    # Compute gaps in seconds (relative to seg_start), as (gap_dur, idx)
    # idx is the index of the interval BEFORE the gap.
    gaps = []
    for i in range(len(intervals) - 1):
        prev_end_s = intervals[i].get("end", 0) / 16000.0
        next_start_s = intervals[i + 1].get("start", 0) / 16000.0
        g = next_start_s - prev_end_s
        if g > 0:
            gaps.append((g, i, prev_end_s, next_start_s))

    if not gaps:
        return None

    # We need ceil(duration / max_len) chunks → ceil(duration / max_len) - 1 cuts.
    # To be safe target slightly below max_len.
    target_chunk = max_len * 0.9
    n_chunks_needed = max(2, int(np.ceil(duration / target_chunk)))
    n_cuts_needed = n_chunks_needed - 1

    # Pick the longest gaps as cut points
    gaps_sorted = sorted(gaps, key=lambda x: -x[0])
    chosen = sorted(gaps_sorted[:n_cuts_needed], key=lambda x: x[1])

    # Cut at midpoint of each chosen gap (in absolute seconds)
    cut_times = [seg_start + (prev_end + next_start) / 2.0
                 for _, _, prev_end, next_start in chosen]

    # Build sub-segments
    sub_segments = []
    prev_t = seg_start
    for ct in cut_times:
        chunk = vad.copy()
        chunk["start"] = prev_t
        chunk["end"] = ct
        sub_segments.append(chunk)
        prev_t = ct
    final_chunk = vad.copy()
    final_chunk["start"] = prev_t
    final_chunk["end"] = seg_end
    sub_segments.append(final_chunk)

    # Sanity: ensure each sub-chunk is < max_len. If not, the longest chosen
    # gaps were not enough (shouldn't happen with n_chunks_needed math, but
    # guard anyway). Returning None lets caller drop, which is safer than
    # producing a still-too-long segment that would loop forever upstream.
    for c in sub_segments:
        if c["end"] - c["start"] >= max_len:
            return None

    return sub_segments


@time_logger
def cut_by_speaker_label(vad_list, audio_duration, stats, postprocess_cfg, step_name="post_process_vad", audio=None):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.
    Also adds a grace period to the end of segments to reduce cut-offs.
    This function now internally tracks and updates statistics.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.
        audio_duration (float): Total duration of the audio in seconds.
        stats (dict): The main statistics dictionary to be updated.
        parameters_cfg(dict): The configuration for the VAD post-process.
        step_name (str): The name of the step for statistics tracking.
        audio (dict, optional): {"waveform", "sample_rate"} for splitting overlong
            spans at silero-detected internal silences. If None, overlong spans
            are dropped (no audio = no way to find natural cut points).

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger
    vad_model = PipelineParam.vad_model

    MERGE_GAP = postprocess_cfg.get("merge_gap", 2)   # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = postprocess_cfg.get("min_segment_length", 3)  # min segment length in seconds
    MAX_SEGMENT_LENGTH = postprocess_cfg.get("max_segment_length", 30)  # max segment length in seconds
    MIN_SIMILARITY = postprocess_cfg.get("intra_similarity_threshold", 0.64) # min similarity between segments
    GRACE_PERIOD_START_S = 0.00
    GRACE_PERIOD_END_S = 0.02
    updated_list = []

    # --- Internal Statistics ---
    discarded_long_count = 0
    discarded_long_duration = 0.0
    split_long_count = 0

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None
        last_embedding = updated_list[-1]["reference_embedding"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            duration = vad["end"] - vad["start"]
            # Try to split at silero-detected internal silences. If none are
            # found (continuous reading with no breath/pause), DROP the
            # segment — we don't fabricate cut points in the middle of words.
            sub_segments = _split_long_span_at_silences(
                vad, audio, vad_model, MAX_SEGMENT_LENGTH, logger
            )
            if sub_segments is None:
                logger.warning(
                    f"cut_by_speaker_label > Discarding segment for speaker "
                    f"{vad['speaker']} ({duration:.2f}s > MAX={MAX_SEGMENT_LENGTH}s) "
                    f"— no internal silence found, refusing to cut mid-word."
                )
                discarded_long_count += 1
                discarded_long_duration += duration
                continue
            logger.info(
                f"cut_by_speaker_label > Split long segment for speaker "
                f"{vad['speaker']} ({duration:.2f}s > MAX={MAX_SEGMENT_LENGTH}s) "
                f"into {len(sub_segments)} pieces at silero-detected silences."
            )
            updated_list.extend(sub_segments)
            split_long_count += 1
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
        ):
            updated_list.append(vad)
            continue

        if last_embedding is None or vad["reference_embedding"] is None or cosine_similarity(last_embedding, vad["reference_embedding"])[0, 0] < MIN_SIMILARITY:
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
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments; "
        f"split {split_long_count} long VAD spans; dropped {discarded_long_count} unsplittable."
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
    # Note: long VAD spans are now hard-chunked rather than discarded, so
    # discarded_long_count stays 0 by design. Only short segments below
    # min_segment_length are dropped here.
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
