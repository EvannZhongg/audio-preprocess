"""
Domain annotation pipeline step.

Produces per-segment `domain_info` with three nested groups:
  - text_domain      (domain / scenario / style)
  - acoustic_domain  (environment / background / quality)
  - speaker_domain   (gender / age_group / accent)

Strategy (cost-optimized):
  - File-level call (1x): text_domain + acoustic_domain, using the first N
    seconds of audio + concatenated transcript.
  - Speaker-level call (k x = number of speakers): speaker_domain, using the
    longest segment of that speaker's audio.
  - Denormalize into every segment's `domain_info`.

A typical file with 2 speakers and 30 segments needs only 3 API calls
instead of 30, saving ~90% cost.

Failures degrade softly: any field falls back to "unknown".
"""
from collections import defaultdict
from typing import Dict, List

from models.domain_classifier import _wav_b64, slice_audio
from utils.logger import time_logger


_DEFAULT_DOMAIN_INFO = {
    "text_domain": {"domain": "unknown", "scenario": "unknown", "style": "unknown"},
    "acoustic_domain": {"environment": "unknown", "background": "unknown", "quality": "unknown"},
    "speaker_domain": {"gender": "unknown", "age_group": "unknown", "accent": "unknown"},
}


@time_logger
def annotate_domains(audio: dict, asr_result: List[dict], domain_cfg: dict) -> List[dict]:
    """Attach `domain_info` to every segment in asr_result.

    Args:
        audio: dict with "waveform" (np.ndarray) and "sample_rate" (int)
        asr_result: list of segment dicts. Each must have "text", "start", "end",
                    "speaker" (optional but expected for speaker_domain).
        domain_cfg: cfg["domain_annotation"] sub-dict.

    Returns:
        The same asr_result list with `domain_info` field added per segment.
    """
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    classifier = PipelineParam.domain_classifier

    # initialize default for every segment first (defensive)
    for seg in asr_result:
        seg["domain_info"] = {
            "text_domain": dict(_DEFAULT_DOMAIN_INFO["text_domain"]),
            "acoustic_domain": dict(_DEFAULT_DOMAIN_INFO["acoustic_domain"]),
            "speaker_domain": dict(_DEFAULT_DOMAIN_INFO["speaker_domain"]),
        }

    if classifier is None:
        logger.debug("Domain classifier not loaded; skipping domain annotation.")
        return asr_result
    if not asr_result:
        return asr_result

    waveform = audio["waveform"]
    sr = audio["sample_rate"]

    text_enabled = domain_cfg.get("text_domain", {}).get("enable", False)
    acoustic_enabled = domain_cfg.get("acoustic_domain", {}).get("enable", False)
    speaker_enabled = domain_cfg.get("speaker_domain", {}).get("enable", False)

    # ---------- File-level call: text_domain + acoustic_domain ----------
    if text_enabled or acoustic_enabled:
        file_audio_seconds = max(
            domain_cfg.get("acoustic_domain", {}).get("audio_sample_seconds", 30),
            domain_cfg.get("text_domain", {}).get("audio_sample_seconds", 0),
        )
        try:
            audio_slice_arr = waveform[: int(file_audio_seconds * sr)]
            audio_b64 = _wav_b64(audio_slice_arr, sr, max_seconds=file_audio_seconds)
            transcript = " ".join(s.get("text", "") for s in asr_result[:25])
            file_domains = classifier.classify_file_level(audio_b64, transcript)
            file_text = file_domains.get("text_domain", _DEFAULT_DOMAIN_INFO["text_domain"])
            file_acoustic = file_domains.get("acoustic_domain", _DEFAULT_DOMAIN_INFO["acoustic_domain"])
        except Exception as e:
            logger.warning(f"Domain file-level call failed: {e}")
            file_text = dict(_DEFAULT_DOMAIN_INFO["text_domain"])
            file_acoustic = dict(_DEFAULT_DOMAIN_INFO["acoustic_domain"])
    else:
        file_text = dict(_DEFAULT_DOMAIN_INFO["text_domain"])
        file_acoustic = dict(_DEFAULT_DOMAIN_INFO["acoustic_domain"])

    # ---------- Speaker-level calls: one per unique speaker ----------
    speaker_domains: Dict[str, dict] = {}
    if speaker_enabled:
        spk_max_seconds = domain_cfg.get("speaker_domain", {}).get("audio_sample_seconds", 15)

        # group segments by speaker, pick the longest segment per speaker
        spk_to_longest: Dict[str, dict] = {}
        for seg in asr_result:
            spk = seg.get("speaker") or "UNKNOWN"
            dur = seg.get("end", 0) - seg.get("start", 0)
            if spk not in spk_to_longest or dur > (spk_to_longest[spk]["end"] - spk_to_longest[spk]["start"]):
                spk_to_longest[spk] = seg

        for spk, seg in spk_to_longest.items():
            try:
                seg_audio = slice_audio(
                    waveform, sr,
                    seg.get("start", 0.0), seg.get("end", 0.0),
                    max_seconds=spk_max_seconds,
                )
                if len(seg_audio) < sr:  # less than 1s, skip
                    speaker_domains[spk] = dict(_DEFAULT_DOMAIN_INFO["speaker_domain"])
                    continue
                audio_b64 = _wav_b64(seg_audio, sr)
                spk_domain = classifier.classify_speaker(audio_b64)
                speaker_domains[spk] = spk_domain.get(
                    "speaker_domain", _DEFAULT_DOMAIN_INFO["speaker_domain"]
                )
            except Exception as e:
                logger.warning(f"Domain speaker-level call failed for {spk}: {e}")
                speaker_domains[spk] = dict(_DEFAULT_DOMAIN_INFO["speaker_domain"])

    # ---------- Denormalize into every segment ----------
    for seg in asr_result:
        if text_enabled:
            seg["domain_info"]["text_domain"] = dict(file_text)
        if acoustic_enabled:
            seg["domain_info"]["acoustic_domain"] = dict(file_acoustic)
        if speaker_enabled:
            spk = seg.get("speaker") or "UNKNOWN"
            seg["domain_info"]["speaker_domain"] = dict(
                speaker_domains.get(spk, _DEFAULT_DOMAIN_INFO["speaker_domain"])
            )

    # log summary
    logger.info(
        f"Domain annotation: text={file_text.get('domain')}/{file_text.get('scenario')}, "
        f"acoustic={file_acoustic.get('environment')}/{file_acoustic.get('quality')}, "
        f"speakers={len(speaker_domains)}"
    )
    return asr_result
