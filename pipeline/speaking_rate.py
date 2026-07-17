"""
Speaking rate analyzer.

Computes a single canonical metric per segment: `speaking_rate` defined as
chars / voiced_duration, where voiced_duration is measured by silero-vad
(excludes silence within the segment).

Why use voiced duration instead of raw segment duration?
  Raw `(end - start)` includes pauses/silence. A segment with `"hello"` (5
  chars) lasting 5s could be 1 char/s gross — but if the speaker actually
  said "hello" in 1s and was silent for 4s, the real articulation rate is
  5 char/s. The "effective" rate is what matters for TTS data quality.

Output (added to each segment):
  - speaking_rate: float, chars per voiced second; -1 on failure
"""
import re

import librosa
import numpy as np
import tqdm

from utils.logger import time_logger

_VAD_SR = 16000
_PUNCT_REGEX = re.compile(r"[,.!?\"'，。！？“”‘’ \t\n\r]")


def _count_chars(text: str) -> int:
    """Strip punctuation/spaces and return remaining char count."""
    if not text:
        return 0
    return len(_PUNCT_REGEX.sub("", text))


def _voiced_duration(seg_audio_16k: np.ndarray, vad_model) -> float:
    """Return total voiced (non-silence) duration in seconds for the slice."""
    seg_dur = len(seg_audio_16k) / _VAD_SR
    if seg_dur < 0.05:
        return 0.0
    try:
        intervals = vad_model._get_speech_timestamps_wrapper(seg_audio_16k, _VAD_SR)
    except Exception:
        return seg_dur  # vad failed: assume all voiced
    if not intervals:
        return 0.0
    voiced = 0.0
    for it in intervals:
        s = it.get("start", 0) / _VAD_SR
        e = it.get("end", 0) / _VAD_SR
        if e > s:
            voiced += (e - s)
    return voiced


@time_logger
def analyze_speaking_rate(audio: dict, asr_result: list, speaking_rate_cfg: dict) -> list:
    """Attach `speaking_rate` to every segment in-place.

    Args:
        audio: dict with "waveform" and "sample_rate".
        asr_result: list of segment dicts (must have text, start, end).
        speaking_rate_cfg: cfg["speaking_rate"] sub-dict (currently unused
            here, only consumed by `filter_by_speaking_rate`).

    Returns:
        The same asr_result with `speaking_rate` set per segment.
    """
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    vad_model = PipelineParam.vad_model

    if not asr_result:
        return asr_result

    # default for all segments
    for seg in asr_result:
        seg.setdefault("speaking_rate", -1.0)

    if vad_model is None:
        logger.warning("VAD model not loaded; skipping speaking_rate.")
        return asr_result

    waveform = audio.get("waveform")
    src_sr = audio.get("sample_rate")
    if waveform is None or src_sr is None:
        logger.warning("audio missing waveform/sample_rate; skipping speaking_rate.")
        return asr_result

    # resample once to 16k for VAD
    if src_sr != _VAD_SR:
        wav16 = librosa.resample(waveform, orig_sr=src_sr, target_sr=_VAD_SR)
    else:
        wav16 = waveform

    total_samples = len(wav16)

    for seg in tqdm.tqdm(asr_result, desc="SPEAKING_RATE"):
        try:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            if end - start < 0.05:
                continue

            s_idx = max(0, int(start * _VAD_SR))
            e_idx = min(total_samples, int(end * _VAD_SR))
            if e_idx <= s_idx:
                continue
            seg_audio = wav16[s_idx:e_idx]

            voiced = _voiced_duration(seg_audio, vad_model)
            char_count = _count_chars(seg.get("text", ""))
            if voiced > 0 and char_count > 0:
                seg["speaking_rate"] = round(char_count / voiced, 4)
        except Exception as e:
            logger.warning(f"speaking_rate failed for a segment: {e}")
            # leave -1.0 default

    vals = [s["speaking_rate"] for s in asr_result if s.get("speaking_rate", -1) >= 0]
    if vals:
        logger.info(
            f"Speaking rate: avg={sum(vals)/len(vals):.2f} char/s, "
            f"n={len(vals)}/{len(asr_result)}"
        )
    return asr_result


def filter_by_speaking_rate(segments: list, speaking_rate_cfg: dict) -> list:
    """Drop segments whose speaking_rate is outside [min, max].

    Threshold = None means that bound is not used.
    speaking_rate = -1 (not computed / failed) is NOT dropped.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

    th = speaking_rate_cfg.get("thresholds", {}) or {}
    sr_min = th.get("speaking_rate_min")
    sr_max = th.get("speaking_rate_max")

    if sr_min is None and sr_max is None:
        logger.debug("No speaking_rate thresholds; skipping filter.")
        return segments

    filtered = []
    drop_low = drop_high = 0
    for seg in segments:
        sr = seg.get("speaking_rate", -1)
        if sr_min is not None and sr >= 0 and sr < sr_min:
            drop_low += 1
            continue
        if sr_max is not None and sr >= 0 and sr > sr_max:
            drop_high += 1
            continue
        filtered.append(seg)

    logger.info(
        f"Speaking-rate filter: kept {len(filtered)}/{len(segments)}, "
        f"dropped low={drop_low}, high={drop_high}"
    )
    return filtered
