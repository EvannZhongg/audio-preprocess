"""
Forced alignment using WhisperX (wav2vec2-CTC).

Wraps `whisperx.load_align_model()` + `whisperx.align()` with:
  - Lazy per-language model loading (each lang ~1.5GB FP16)
  - Thread-safe init (multi-worker safe)
  - Permanent failure marker (don't retry languages that failed)
  - Soft degrade: returns -1.0 instead of raising

Public API:
  - WhisperXAligner.align_segments(audio_16k, language, segments) -> list[float]
    Returns one alignment_score per segment (mean per-word CTC confidence).
    Score range [0, 1]; -1.0 if alignment failed for that segment.

Why WhisperX:
  - Multilingual (uses wav2vec2-CTC per-language models)
  - GPU-accelerated batch alignment
  - Output: word-level (or char-level for CJK) timestamps + scores
  - Reuses existing Whisper ASR result; only loads wav2vec2 part
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WhisperXAligner:
    """Lazy multi-language WhisperX wrapper."""

    def __init__(self, device: str = "cuda", model_dir: Optional[str] = None):
        """
        Args:
            device: torch device string, e.g. "cuda", "cuda:0", "cpu"
            model_dir: optional cache dir for align models
        """
        self.device = device
        self.model_dir = model_dir
        self._models: Dict[str, Tuple[object, dict]] = {}
        self._failed_langs = set()
        self._lock = threading.Lock()

    def _get_model(self, language: str) -> Tuple[Optional[object], Optional[dict]]:
        """Return (model, metadata) for language, or (None, None) on failure."""
        with self._lock:
            if language in self._failed_langs:
                return None, None
            if language in self._models:
                return self._models[language]
            try:
                import whisperx
                logger.info(f"Loading WhisperX align model for language: {language}")
                kwargs = {"language_code": language, "device": self.device}
                if self.model_dir:
                    kwargs["model_dir"] = self.model_dir
                model_a, metadata = whisperx.load_align_model(**kwargs)
                self._models[language] = (model_a, metadata)
                return model_a, metadata
            except Exception as e:
                logger.warning(f"Failed to load WhisperX align model for {language}: {e}")
                self._failed_langs.add(language)
                return None, None

    def align_segments(
        self,
        audio_16k,
        language: str,
        segments: List[dict],
    ) -> List[float]:
        """Forced-align all segments in the given language.

        Args:
            audio_16k: full audio waveform (np.ndarray, 16kHz, mono float32)
            language: ISO language code (e.g. "ru", "zh", "en")
            segments: list of {"text": str, "start": float, "end": float}
                      timestamps relative to the audio_16k passed here.

        Returns:
            List of alignment_score values, parallel to `segments`.
            Each score is mean CTC confidence in [0, 1]; -1.0 on failure.
        """
        if not segments:
            return []

        model_a, metadata = self._get_model(language)
        if model_a is None or metadata is None:
            return [-1.0] * len(segments)

        try:
            import whisperx

            # Whisperx expects 'whisper-style' segment dicts
            whisper_segments = [
                {
                    "text": s.get("text", ""),
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                }
                for s in segments
            ]

            result = whisperx.align(
                whisper_segments,
                model_a,
                metadata,
                audio_16k,
                self.device,
                return_char_alignments=False,
            )

            aligned_segs = result.get("segments", []) if isinstance(result, dict) else []
            scores = []
            # match output length to input by index. WhisperX preserves order.
            for i, _ in enumerate(segments):
                if i >= len(aligned_segs):
                    scores.append(-1.0)
                    continue
                words = aligned_segs[i].get("words", [])
                word_scores = [
                    float(w["score"])
                    for w in words
                    if isinstance(w, dict) and "score" in w and w["score"] is not None
                ]
                if not word_scores:
                    scores.append(-1.0)
                else:
                    avg = sum(word_scores) / len(word_scores)
                    # clamp to [0, 1] just in case
                    scores.append(max(0.0, min(1.0, avg)))
            return scores
        except Exception as e:
            logger.warning(f"WhisperX align failed for language={language}: {e}")
            return [-1.0] * len(segments)
