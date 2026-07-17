"""
Abnormal silence detection & filtering.

Uses silero-vad to find non-speech gaps inside each segment, then checks
how many of them can be "explained" by pause-justifying punctuation in
the segment's text. Unexplained long gaps are flagged as abnormal — they
typically indicate:

  - VAD post-processing merged across a real pause (merge_gap absorbed it)
  - Speaker hesitation / dead air
  - Music or silence inserted in the middle of speech
  - Cross-talk dropped by ASR but the silence remains in audio

Single output field per segment: `abnormal_silence_count` (int)
  -1: VAD failed / could not compute
   0: no abnormal silences (all long internal pauses explained by punctuation)
   N: N unexplained long pauses

Failures degrade gracefully (-1) and are NOT dropped by the filter.

Note on scope: only INTERNAL gaps are counted. Leading/trailing silence is
the responsibility of VAD post-processing (cut_by_speaker_label /
GRACE_PERIOD), not this filter.
"""
import re

import librosa
import numpy as np
import tqdm

from utils.logger import time_logger

_VAD_SR = 16000

# Punctuation that justifies a pause in spoken text.
# Sentence-final, clause separators, ellipsis, dash. Quotes/parens excluded.
# `+` quantifier so '...' or '……' counts as one pause cue, not three.
_PAUSE_PUNCT = re.compile(r"[,.!?;:。、！？；：，…—–\-]+")


def _count_pause_punct(text: str) -> int:
    """Count contiguous groups of pause-justifying punctuation."""
    if not text:
        return 0
    return len(_PAUSE_PUNCT.findall(text))


def _find_long_internal_silences(seg_audio_16k: np.ndarray, vad_model, min_sec: float):
    """Return list of internal silence durations (in seconds) >= min_sec.

    Returns:
        None if VAD itself failed.
        []   if VAD succeeded but no long internal pause.
        [d1, d2, ...] each >= min_sec.

    Leading/trailing silences are intentionally ignored.
    """
    seg_dur = len(seg_audio_16k) / _VAD_SR
    if seg_dur < 0.05:
        return []
    try:
        intervals = vad_model._get_speech_timestamps_wrapper(seg_audio_16k, _VAD_SR)
    except Exception:
        return None
    if not intervals or len(intervals) < 2:
        # 0 or 1 speech chunk → no internal gap
        return []

    silences = []
    for i in range(len(intervals) - 1):
        prev_end = intervals[i].get("end", 0) / _VAD_SR
        next_start = intervals[i + 1].get("start", 0) / _VAD_SR
        gap = next_start - prev_end
        if gap >= min_sec:
            silences.append(gap)
    return silences


@time_logger
def detect_abnormal_silence(audio: dict, asr_result: list, silence_cfg: dict) -> list:
    """Attach `abnormal_silence_count` to every segment in-place.

    Args:
        audio: dict with "waveform" (np.ndarray, native SR) and "sample_rate".
        asr_result: list of segment dicts (must have text, start, end).
        silence_cfg: cfg["silence_filter"] sub-dict.

    Returns:
        The same asr_result with `abnormal_silence_count` set per segment.
    """
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    vad_model = PipelineParam.vad_model

    if not asr_result:
        return asr_result

    # default for all segments
    for seg in asr_result:
        seg.setdefault("abnormal_silence_count", -1)

    if vad_model is None:
        logger.warning("VAD model not loaded; skipping abnormal_silence detection.")
        return asr_result

    waveform = audio.get("waveform")
    src_sr = audio.get("sample_rate")
    if waveform is None or src_sr is None:
        logger.warning("audio missing waveform/sample_rate; skipping abnormal_silence.")
        return asr_result

    # resample once to 16k for VAD
    if src_sr != _VAD_SR:
        wav16 = librosa.resample(waveform, orig_sr=src_sr, target_sr=_VAD_SR)
    else:
        wav16 = waveform
    if wav16.ndim > 1:
        wav16 = wav16.mean(axis=0)

    total_samples = len(wav16)
    min_silence_ms = int(silence_cfg.get("min_silence_ms", 300))
    min_silence_sec = min_silence_ms / 1000.0

    abnormal_segments = 0
    for seg in tqdm.tqdm(asr_result, desc="ABNORMAL_SILENCE"):
        try:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            if end - start < 0.05:
                seg["abnormal_silence_count"] = 0
                continue

            s_idx = max(0, int(start * _VAD_SR))
            e_idx = min(total_samples, int(end * _VAD_SR))
            if e_idx <= s_idx:
                continue
            seg_audio = wav16[s_idx:e_idx]

            silences = _find_long_internal_silences(seg_audio, vad_model, min_silence_sec)
            if silences is None:
                # VAD failed for this segment — leave -1
                continue

            n_punct = _count_pause_punct(seg.get("text", ""))
            unexplained = max(0, len(silences) - n_punct)
            seg["abnormal_silence_count"] = unexplained
            if unexplained > 0:
                abnormal_segments += 1
        except Exception as e:
            logger.warning(f"abnormal_silence failed for a segment: {e}")
            # leave -1

    logger.info(
        f"Abnormal silence: {abnormal_segments}/{len(asr_result)} segments have "
        f"unexplained pauses >= {min_silence_ms}ms"
    )
    return asr_result


def filter_by_abnormal_silence(segments: list, silence_cfg: dict) -> list:
    """Drop segments whose `abnormal_silence_count` exceeds the threshold.

    Threshold = None means filter is off.
    abnormal_silence_count = -1 (not computed / failed) is NOT dropped.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

    th = silence_cfg.get("thresholds", {}) or {}
    max_count = th.get("abnormal_silence_count_max")
    if max_count is None:
        logger.debug("No abnormal_silence_count_max threshold; skipping filter.")
        return segments

    filtered = []
    drop = 0
    for seg in segments:
        n = seg.get("abnormal_silence_count", -1)
        if n >= 0 and n > max_count:
            drop += 1
            continue
        filtered.append(seg)

    logger.info(
        f"Abnormal-silence filter: kept {len(filtered)}/{len(segments)}, dropped {drop}"
    )
    return filtered
