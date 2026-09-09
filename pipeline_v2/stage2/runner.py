"""Stage-2 chunk runner: remote ASR + v1 post-processing.

Given one stage-1 chunk wav plus its stage-1 segments (VAD/diarization
output, no text yet), this module:

  1. Runs remote Qwen3 ASR per-segment (segment boundaries = stage-1 VAD
     boundaries; no re-VAD, no cross-validation).
  2. Runs v1's post-processing functions unmodified, via
     `pipeline.global_var.PipelineParam` (domain annotation, speaking rate,
     abnormal silence, forced alignment, text quality), in the exact same
     order as v1's `main_process.py` (domain -> speaking_rate -> silence ->
     alignment -> text_quality), where each `filter_by_*` step narrows the
     working set fed to the next step (matches v1's cost/quality tradeoff:
     a segment dropped early is not re-scored downstream).
  3. Because stage 2 must "write back, not delete" (segments must stay
     1:1 with stage-1 output), a dropped segment is never removed from the
     returned list. Instead we diff each `filter_by_*` result against its
     input by object identity and stamp a `dropped_by_*` boolean flag.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import librosa
import numpy as np

from pipeline.alignment_filter import compute_alignment_score, filter_by_alignment
from pipeline.domain_annotation import annotate_domains
from pipeline.silence_filter import (detect_abnormal_silence,
                                     filter_by_abnormal_silence)
from pipeline.speaking_rate import analyze_speaking_rate, filter_by_speaking_rate
from pipeline.text_quality_filtering import (filter_by_text_quality,
                                             text_quality_prediction)
from pipeline_v2.params import Stage2Params
from pipeline_v2.stage2.models import Stage2Models
from pipeline_v2.state import ASR_ACCESS_FAILED_MARKER

_ASR_SR = 16000


class AsrAccessFailedError(RuntimeError):
    """Raised when the remote ASR service could not be reached / did not
    answer usably for at least one segment of a chunk.

    This is deliberately an exception rather than a degraded result: the empty
    text such a failure produces is indistinguishable from a genuinely silent
    segment once written out, so persisting it would silently lose data forever
    (resume would never revisit the chunk). Raising instead makes the chunk a
    failed chunk whose `error` carries `ASR_ACCESS_FAILED_MARKER`, which the
    resume logic recognizes as retriable.

    Whole-chunk granularity is intentional: segments of a chunk share batch
    requests and service instances, so a failure is usually batch- or
    instance-level, and retrying the chunk matches the resume key
    (`chunk_audio_path`) exactly.
    """

# Fields possibly attached by any post-processing stage. Set on every
# segment up front so that a segment dropped early (and thus skipped by
# later stages) still has a complete, schema-stable dict for the stage-2
# parquet writer instead of a KeyError / missing column.
_DEFAULT_FIELDS: dict[str, Any] = {
    "domain_info": None,
    "speaking_rate": -1.0,
    "abnormal_silence_count": -1,
    "alignment_score": -1.0,
    "ppl": -1.0,
    "spell_score": -1.0,
    "llm_quality": -1.0,
    "semantic_completeness": -1.0,
    "tts_suitability": -1.0,
    "dropped_by_speaking_rate": False,
    "dropped_by_silence": False,
    "dropped_by_alignment": False,
    "dropped_by_text_quality": False,
    "dropped_by_asr_validation": False,
    "asr_wer": -1.0,
    "asr_val_text": None,
}


def _mark_dropped(before: list, after: list, flag: str) -> None:
    """Set `seg[flag] = True` for every dict in `before` absent from `after`
    (identity comparison), else `seg[flag] = False`. Mutates `before`
    in-place; never removes anything.
    """
    kept_ids = {id(seg) for seg in after}
    for seg in before:
        seg[flag] = id(seg) not in kept_ids


def _validate_asr_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list,
    params: Stage2Params,
    models: Stage2Models,
    logger: logging.Logger,
) -> list:
    """Cross-validate stage-2 ASR text against a second, independently
    configured ASR model (`models.validation_asr_model`).

    Filtering logic -- language-mismatch skip; CER for zh/ja/ko else WER;
    `wer_threshold` cutoff -- is lifted verbatim from
    `pipeline.asr_process.asr()`'s "ASR Cross-Validation Logic" block, just
    re-pointed at the primary text already produced by `run_stage2_asr`
    instead of re-running the primary model.

    Returns the survivor sublist (segments passing both checks); the caller
    diffs it against `segments` via `_mark_dropped` like every other stage-2
    filter, since v1 deletes failing segments outright but stage 2 must stay
    1:1 with stage-1 output.
    """
    import jiwer

    from pipeline.asr_process import normalize_text_for_cer, normalize_text_for_wer

    val_cfg = params.asr_validation
    target_language = val_cfg.get("language", "zh")
    wer_threshold = val_cfg.get("wer_threshold", 0.15)

    if sample_rate != _ASR_SR:
        wav_16k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=_ASR_SR)
    else:
        wav_16k = waveform
    if wav_16k.ndim > 1:
        wav_16k = wav_16k.mean(axis=0)

    vad_segments = [
        {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": seg.get("speaker"),
        }
        for seg in segments
    ]
    val_out = models.validation_asr_model.transcribe(
        wav_16k, vad_segments, batch_size=None, language=target_language, print_progress=False
    )
    val_segments = val_out.get("segments") or []

    if len(val_segments) != len(segments):
        logger.warning(
            f"Stage2 ASR validation returned {len(val_segments)} segments for "
            f"{len(segments)} inputs; dropping the whole batch (mirrors v1's "
            "asr() behaviour on a segment-count mismatch)."
        )
        return []

    survivors = []
    for seg, val_seg in zip(segments, val_segments):
        seg["asr_val_text"] = val_seg.get("text")

        seg_detected_lang = seg.get("detected_language")
        if seg_detected_lang and seg_detected_lang != "unknown" and seg_detected_lang != target_language:
            logger.debug(
                "Stage2 ASR validation: segment skipped due to language "
                f"mismatch (detected {seg_detected_lang!r}, expected {target_language!r})"
            )
            continue

        if target_language in ("zh", "ja", "ko"):
            ref = normalize_text_for_cer(seg.get("text") or "")
            hyp = normalize_text_for_cer(val_seg.get("text") or "")
            error_rate = jiwer.cer(ref, hyp) if ref and hyp else 1.0
        else:
            ref = normalize_text_for_wer(seg.get("text") or "")
            hyp = normalize_text_for_wer(val_seg.get("text") or "")
            error_rate = jiwer.wer(ref, hyp) if ref and hyp else 1.0
        seg["asr_wer"] = error_rate

        if error_rate <= wer_threshold:
            survivors.append(seg)
        else:
            logger.debug(f"Stage2 ASR validation: segment dropped due to high error rate {error_rate:.2f}")

    return survivors


def run_stage2_asr(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_segments: list,
    params: Stage2Params,
    models: Stage2Models,
    logger: Optional[logging.Logger] = None,
) -> list:
    """Run only step 1 (remote ASR) for one chunk wav.

    This is pure network I/O (Qwen3ASR HTTP calls) and does not touch the
    local GPU, so callers should NOT hold any GPU-serializing lock around
    this call -- it's safe (and desired) to run many of these concurrently
    across actor threads.

    Args:
        waveform: chunk audio, native sample rate, mono float32.
        sample_rate: native sample rate of `waveform`.
        chunk_segments: stage-1 segments for this chunk, each a dict with at
            least "start", "end" (seconds, relative to this chunk's wav) and
            "speaker_id". Order is preserved throughout; the returned list
            has exactly the same length and order.
        params: Stage2Params (post-processing config sub-dicts, same keys as
            v1's config.json).
        models: Stage2Models bundle (loaded once per actor at init time).
        logger: optional logger; falls back to `PipelineParam.logger`.

    Returns:
        A list parallel to `chunk_segments` (same length/order), each a
        fresh dict with ASR text/language plus all `_DEFAULT_FIELDS`
        pre-filled (to be overwritten by `run_stage2_postprocess`).

    Raises:
        AsrAccessFailedError: at least one segment's remote ASR call failed
            (timeout / HTTP error / malformed response). Nothing is written
            back for the chunk so it can be retried on a later run.
    """
    from pipeline.global_var import PipelineParam

    if logger is None:
        logger = PipelineParam.logger or logging.getLogger(__name__)

    if not chunk_segments:
        return []

    # Qwen3ASR.transcribe() expects audio already resampled to 16kHz; the
    # v1 postprocessing functions instead resample internally from the
    # native rate, so we keep `waveform`/`sample_rate` (native) for them and
    # build a separate 16k copy only for ASR.
    if sample_rate != _ASR_SR:
        wav_16k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=_ASR_SR)
    else:
        wav_16k = waveform
    if wav_16k.ndim > 1:
        wav_16k = wav_16k.mean(axis=0)

    vad_segments = [
        {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": seg.get("speaker_id") or seg.get("speaker"),
        }
        for seg in chunk_segments
    ]
    asr_out = models.asr_model.transcribe(wav_16k, vad_segments)
    asr_segments = asr_out.get("segments") or []
    default_language = asr_out.get("language") or params.alignment.get(
        "default_language", "en"
    )

    # Bail out before any post-processing / write-back if the remote service
    # failed on any segment: the resulting empty text would be
    # indistinguishable from real silence downstream, so we'd rather fail the
    # chunk (retriable) than persist a partially-lost transcript.
    n_failed = sum(1 for seg in asr_segments if isinstance(seg, dict) and seg.get("asr_failed"))
    if n_failed:
        raise AsrAccessFailedError(
            f"{ASR_ACCESS_FAILED_MARKER}: remote ASR failed for {n_failed}/"
            f"{len(asr_segments)} segments of this chunk"
        )

    if len(asr_segments) != len(chunk_segments):
        logger.warning(
            f"Stage2 ASR returned {len(asr_segments)} segments for "
            f"{len(chunk_segments)} inputs; padding/truncating to input length"
        )
        if len(asr_segments) < len(chunk_segments):
            asr_segments = asr_segments + [
                {} for _ in range(len(chunk_segments) - len(asr_segments))
            ]
        else:
            asr_segments = asr_segments[: len(chunk_segments)]

    for seg, src in zip(asr_segments, chunk_segments):
        seg.setdefault("start", src.get("start", 0.0))
        seg.setdefault("end", src.get("end", 0.0))
        seg["speaker"] = src.get("speaker_id") or seg.get("speaker")
        lang = seg.get("detected_language") or default_language
        seg["language"] = lang if lang and lang != "unknown" else default_language
        for key, default in _DEFAULT_FIELDS.items():
            seg[key] = default() if callable(default) else default

    return asr_segments


def run_stage2_postprocess(
    waveform: np.ndarray,
    sample_rate: int,
    asr_segments: list,
    params: Stage2Params,
    models: Stage2Models,
    logger: Optional[logging.Logger] = None,
) -> list:
    """Run steps 2 & 3 (v1 GPU-touching post-processing, cascading filters)
    on top of `run_stage2_asr`'s output. Mutates and returns `asr_segments`.

    Callers SHOULD serialize concurrent invocations of this function (e.g.
    with a per-actor GPU lock) since it touches local GPU models
    (alignment / ppl scorer, etc.), unlike `run_stage2_asr`.
    """
    from pipeline.global_var import PipelineParam

    if logger is None:
        logger = PipelineParam.logger or logging.getLogger(__name__)

    if not asr_segments:
        return asr_segments

    audio_native = {"waveform": waveform, "sample_rate": sample_rate}

    # Mirrors pipeline/main_process.py Step 5.5 -> 5.7 -> 5.75 -> 5.8 -> 6.5.
    current = asr_segments

    # ASR cross-validation, lifted from pipeline/asr_process.py's "ASR
    # Cross-Validation Logic" block. Placed first in the cascade (like v1,
    # where it runs immediately after the primary ASR call) since a segment
    # whose transcript can't be corroborated is not a good candidate for the
    # more expensive downstream scoring steps either.
    if params.asr_validation.get("enable", False) and models.validation_asr_model is not None:
        try:
            survivors = _validate_asr_segments(waveform, sample_rate, current, params, models, logger)
            _mark_dropped(current, survivors, "dropped_by_asr_validation")
            current = survivors
        except Exception as e:
            logger.warning(f"Stage2 ASR cross-validation failed: {e}")

    if params.domain_annotation.get("enable", False):
        try:
            annotate_domains(audio_native, current, params.domain_annotation)
        except Exception as e:
            logger.warning(f"Stage2 domain annotation failed: {e}")

    if params.speaking_rate.get("enable", False):
        try:
            analyze_speaking_rate(audio_native, current, params.speaking_rate)
            survivors = filter_by_speaking_rate(current, params.speaking_rate)
            _mark_dropped(current, survivors, "dropped_by_speaking_rate")
            current = survivors
        except Exception as e:
            logger.warning(f"Stage2 speaking-rate analysis failed: {e}")

    if params.silence_filter.get("enable", False):
        try:
            detect_abnormal_silence(audio_native, current, params.silence_filter)
            survivors = filter_by_abnormal_silence(current, params.silence_filter)
            _mark_dropped(current, survivors, "dropped_by_silence")
            current = survivors
        except Exception as e:
            logger.warning(f"Stage2 abnormal-silence detection failed: {e}")

    if params.alignment.get("enable", False):
        try:
            compute_alignment_score(audio_native, current, params.alignment)
            survivors = filter_by_alignment(current, params.alignment)
            _mark_dropped(current, survivors, "dropped_by_alignment")
            current = survivors
        except Exception as e:
            logger.warning(f"Stage2 alignment scoring failed: {e}")

    if params.text_quality.get("enable", False):
        try:
            text_quality_prediction(current, params.text_quality)
            survivors = filter_by_text_quality(current, params.text_quality)
            _mark_dropped(current, survivors, "dropped_by_text_quality")
            current = survivors
        except Exception as e:
            logger.warning(f"Stage2 text-quality scoring failed: {e}")

    return asr_segments


def run_stage2_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_segments: list,
    params: Stage2Params,
    models: Stage2Models,
    logger: Optional[logging.Logger] = None,
) -> list:
    """Run the full stage-2 pipeline (ASR + post-processing) for one chunk
    wav, back-to-back with no lock. Kept for callers that don't care about
    separating the network-bound ASR step from the GPU-bound post-processing
    step; `Stage2Actor` calls `run_stage2_asr`/`run_stage2_postprocess`
    directly instead so it can serialize only the GPU-touching part.
    """
    asr_segments = run_stage2_asr(
        waveform, sample_rate, chunk_segments, params, models, logger=logger
    )
    return run_stage2_postprocess(
        waveform, sample_rate, asr_segments, params, models, logger=logger
    )
