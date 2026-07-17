"""
Audio-text alignment scoring & filtering.

Uses WhisperX (wav2vec2-CTC) to forced-align ASR segments. Single output
field per segment: `alignment_score` (mean per-word CTC confidence,
range [0, 1]; -1.0 = scoring failed).

Catches problems no other evaluator can:
  - ASR missing words (audio said more than text)
  - ASR hallucinated words (text contains words not in audio)
  - VAD boundary mis-cuts
  - Background speech overlap on the segment
  - Word-order errors (rare but severe)

Failures degrade gracefully (-1) and are NOT dropped.
"""
from collections import defaultdict

import librosa
import numpy as np

from utils.logger import time_logger

_ALIGN_SR = 16000


@time_logger
def compute_alignment_score(audio: dict, asr_result: list, alignment_cfg: dict) -> list:
    """Attach `alignment_score` to every segment in-place.

    Args:
        audio: dict with "waveform" (np.ndarray, native SR) and "sample_rate".
        asr_result: list of segment dicts (must have text, start, end, language).
        alignment_cfg: cfg["alignment"] sub-dict.

    Returns:
        The same asr_result with `alignment_score` set per segment.
    """
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    aligner = PipelineParam.aligner

    if not asr_result:
        return asr_result

    # default
    for seg in asr_result:
        seg.setdefault("alignment_score", -1.0)

    if aligner is None:
        logger.warning("Aligner not loaded; skipping alignment_score.")
        return asr_result

    waveform = audio.get("waveform")
    src_sr = audio.get("sample_rate")
    if waveform is None or src_sr is None:
        logger.warning("audio missing waveform/sample_rate; skipping alignment_score.")
        return asr_result

    # resample once to 16k for wav2vec2
    if src_sr != _ALIGN_SR:
        wav16 = librosa.resample(waveform, orig_sr=src_sr, target_sr=_ALIGN_SR)
    else:
        wav16 = waveform

    # ensure mono float32 / numpy contiguous
    if wav16.ndim > 1:
        wav16 = wav16.mean(axis=0)
    wav16 = np.ascontiguousarray(wav16.astype(np.float32))

    # group by language so we load each align model only once
    by_language = defaultdict(list)
    default_lang = alignment_cfg.get("default_language", "ru")
    for i, seg in enumerate(asr_result):
        lang = seg.get("language") or default_lang
        # whisperx language codes are 2-letter ISO; truncate if needed
        if lang and lang != "unknown":
            by_language[lang].append((i, seg))

    if not by_language:
        return asr_result

    for lang, lang_segs in by_language.items():
        whisper_segments = [
            {
                "text": seg.get("text", ""),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
            }
            for _, seg in lang_segs
        ]
        try:
            scores = aligner.align_segments(wav16, lang, whisper_segments)
        except Exception as e:
            logger.warning(f"alignment batch failed for lang={lang}: {e}")
            scores = [-1.0] * len(lang_segs)

        for (idx, _), score in zip(lang_segs, scores):
            asr_result[idx]["alignment_score"] = round(score, 4) if score >= 0 else -1.0

    vals = [s["alignment_score"] for s in asr_result if s.get("alignment_score", -1) >= 0]
    if vals:
        logger.info(
            f"Alignment score: avg={sum(vals)/len(vals):.3f}, "
            f"n={len(vals)}/{len(asr_result)}"
        )
    return asr_result


def filter_by_alignment(segments: list, alignment_cfg: dict) -> list:
    """Drop segments whose alignment_score is below `alignment_score_min`.

    Threshold = None means filter is off.
    alignment_score = -1 (not computed / failed) is NOT dropped.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

    th = alignment_cfg.get("thresholds", {}) or {}
    min_score = th.get("alignment_score_min")
    if min_score is None:
        logger.debug("No alignment_score_min threshold; skipping filter.")
        return segments

    filtered = []
    drop = 0
    for seg in segments:
        s = seg.get("alignment_score", -1)
        if s >= 0 and s < min_score:
            drop += 1
            continue
        filtered.append(seg)

    logger.info(f"Alignment filter: kept {len(filtered)}/{len(segments)}, dropped {drop}")
    return filtered
