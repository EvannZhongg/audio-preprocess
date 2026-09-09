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

    def __init__(
        self,
        device: str = "cuda",
        model_dir: Optional[str] = None,
        language_models: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            device: torch device string, e.g. "cuda", "cuda:0", "cpu"
            model_dir: legacy passthrough — forwarded to whisperx as cache_dir
                if no per-language path matches.
            language_models: dict of {lang_code: local_path}. Each path may be
                either a HF cache repo root ("models--<org>--<name>") or an
                already-resolved snapshot dir. The class auto-resolves repo
                roots to their snapshot dir before handing to from_pretrained.
                Languages absent from this dict fall back to whisperx's
                default HF repo id (which requires network).
        """
        self.device = device
        self.model_dir = model_dir
        self.language_models = self._normalize_language_models(language_models or {})
        self._models: Dict[str, Tuple[object, dict]] = {}
        self._failed_langs = set()
        self._lock = threading.Lock()
        if self.language_models:
            logger.info(
                f"WhisperXAligner: per-language local paths configured for "
                f"{sorted(self.language_models.keys())}"
            )

    @staticmethod
    def _normalize_language_models(mapping: Dict[str, str]) -> Dict[str, str]:
        """Resolve any HF cache repo-root paths to their snapshot subdirectory.

        Accepts both forms; HuggingFace cache layout is:
            <hub>/models--<org>--<name>/snapshots/<commit>/{config.json,...}
        from_pretrained() with a local path expects a directory directly
        containing config.json — i.e. the snapshot dir.
        """
        import os

        resolved = {}
        for lang, path in mapping.items():
            if not path:
                continue
            if not os.path.isabs(path):
                # Treat as HF repo id; pass through unchanged.
                resolved[lang] = path
                continue
            # Already resolved snapshot dir?
            if os.path.isfile(os.path.join(path, "config.json")):
                resolved[lang] = path
                continue
            # Repo root with snapshots/ inside?
            snap = os.path.join(path, "snapshots")
            if os.path.isdir(snap):
                ref_main = os.path.join(path, "refs", "main")
                chosen = None
                if os.path.isfile(ref_main):
                    try:
                        commit = open(ref_main).read().strip()
                        candidate = os.path.join(snap, commit)
                        if os.path.isfile(os.path.join(candidate, "config.json")):
                            chosen = candidate
                    except Exception:
                        pass
                if chosen is None:
                    for d in sorted(os.listdir(snap)):
                        candidate = os.path.join(snap, d)
                        if os.path.isfile(os.path.join(candidate, "config.json")):
                            chosen = candidate
                            break
                if chosen:
                    logger.info(f"WhisperXAligner: resolved {lang} → {chosen}")
                    resolved[lang] = chosen
                else:
                    logger.warning(
                        f"WhisperXAligner: {lang} path {path} has snapshots/ "
                        f"but no usable commit; falling back to default repo id"
                    )
            else:
                # Path exists but neither a snapshot dir nor a repo root — pass
                # through and let from_pretrained surface the error.
                resolved[lang] = path
        return resolved

    def _get_model(self, language: str) -> Tuple[Optional[object], Optional[dict]]:
        """Return (model, metadata) for language, or (None, None) on failure."""
        with self._lock:
            if language in self._failed_langs:
                return None, None
            if language in self._models:
                return self._models[language]
            try:
                import whisperx
                kwargs = {"language_code": language, "device": self.device}
                # If a local path was configured for this language, force
                # whisperx to use it as model_name — skips HF lookup entirely.
                local_path = self.language_models.get(language)
                if local_path:
                    logger.info(f"Loading WhisperX align model for {language} from {local_path}")
                    kwargs["model_name"] = local_path
                else:
                    logger.info(f"Loading WhisperX align model for {language} (default repo)")
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
        except Exception as e:
            logger.warning(f"WhisperX align failed for language={language}: {e}")
            return [-1.0] * len(segments)

        # NOTE: whisperx.align() internally splits each input segment's text
        # into sentences (nltk punkt, on '.', '!', '?', etc.) and, on
        # successful alignment, emits ONE OUTPUT SEGMENT PER SENTENCE. So a
        # single multi-sentence input segment can expand into N output
        # segments — the output length is NOT guaranteed to match the input
        # length, and output[i] does NOT reliably correspond to input[i].
        # Aligning by index (as a naive batched call would) silently drops
        # words from multi-sentence segments and shifts all subsequent
        # segments' scores onto the wrong text.
        #
        # To stay correct regardless of how many sentences a segment
        # contains, we align ONE segment at a time and pool the words from
        # ALL sub-segments whisperx returns for that call. whisperx already
        # slices audio_16k per-segment internally, so this has no extra
        # compute cost vs. batching multiple segments in one call.
        scores = []
        for s in segments:
            whisper_segment = {
                "text": s.get("text", ""),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
            }
            try:
                result = whisperx.align(
                    [whisper_segment],
                    model_a,
                    metadata,
                    audio_16k,
                    self.device,
                    return_char_alignments=False,
                )
            except Exception as e:
                logger.warning(f"WhisperX align failed for language={language}: {e}")
                scores.append(-1.0)
                continue

            aligned_segs = result.get("segments", []) if isinstance(result, dict) else []
            word_scores = []
            for aseg in aligned_segs:
                words = aseg.get("words", [])
                word_scores.extend(
                    float(w["score"])
                    for w in words
                    if isinstance(w, dict) and "score" in w and w["score"] is not None
                )
            if not word_scores:
                scores.append(-1.0)
            else:
                avg = sum(word_scores) / len(word_scores)
                # clamp to [0, 1] just in case
                scores.append(max(0.0, min(1.0, avg)))
        return scores
