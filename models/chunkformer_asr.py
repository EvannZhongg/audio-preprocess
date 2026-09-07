"""ChunkFormer ASR wrapper (validation provider for Vietnamese).

Wraps `khanhld/chunkformer-ctc-large-vie` -- a Vietnamese-only CTC model
(non-autoregressive, so per-segment decode is much cheaper than an LLM-style
decoder) -- behind the same duck-typed `transcribe(...)` contract every other
ASR provider in this repo exposes, so it can be selected purely via config:

    "validation_asr_provider": "chunkformer"

Decode path mirrors tmp/verify_chunkformer.py's sentence-level mode (write the
VAD segment to a temp wav, `endless_decode(return_timestamps=False)`), which
is the usage that model was validated with on this data.
"""

import logging
import os
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch
from filelock import FileLock

logger = logging.getLogger(__name__)

# The pipeline contract feeds transcribe() a 16 kHz mono float32 waveform
# (stage-2's `_validate_asr_segments` resamples before calling), matching
# every other provider wrapper.
_SAMPLE_RATE = 16000


class ChunkFormerASR:
    """Vietnamese ChunkFormer CTC ASR, duck-typed like FunASR/whisper wrappers.

    The underlying model is Vietnamese-only: the `language` argument of
    `transcribe()` is accepted for contract compatibility but ignored.
    """

    def __init__(self, model_dir: str, device: str = "cpu", **kwargs):
        # Imported lazily so this module (and therefore configs that merely
        # mention chunkformer) works in environments without the pip package;
        # only actually *loading* the model requires it.
        from chunkformer import ChunkFormerModel

        # Same reasoning as models/funasr_asr.py: concurrent actors on one
        # machine must serialize the (possible) hub download / first load of
        # the same model directory.
        lock_dir = os.path.join(tempfile.gettempdir(), "chunkformer_model_locks")
        os.makedirs(lock_dir, exist_ok=True)
        lock_key = os.path.basename(model_dir.rstrip("/")) or model_dir.replace("/", "_")
        lock_file = os.path.join(lock_dir, f"{lock_key}.lock")

        with FileLock(lock_file):
            logger.info(f"Loading ChunkFormer model from: {model_dir}")
            self.model = ChunkFormerModel.from_pretrained(model_dir)
        self.device = device
        self.model = self.model.to(device).eval()
        logger.info(f"ChunkFormer model loaded on {device}")

    def detect_language(self, audio: np.ndarray):
        """Contract stub: single-language model, no detection."""
        return None, 0.0

    @staticmethod
    def _extract_text(out) -> str:
        """`endless_decode` returns a list of {'decode': ..., 'timestamp': ...}
        dicts (possibly several for long audio), a bare string, or None on
        empty input -- normalize all of that to one text line."""
        if isinstance(out, list):
            parts = []
            for o in out:
                if isinstance(o, dict):
                    parts.append(str(o.get("decode", "") or ""))
                else:
                    parts.append(str(o or ""))
            return " ".join(p for p in parts if p).strip()
        return str(out or "").strip()

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        batch_size=None,
        language: str = None,
        print_progress: bool = False,
        **kwargs,
    ) -> dict:
        """Transcribe pre-cut VAD segments, 1:1, from one 16 kHz waveform.

        Always returns exactly one segment dict per `vad_segments` entry (a
        failed/empty segment yields empty text, never a missing row), because
        stage-2's `_validate_asr_segments` drops the whole chunk on a
        segment-count mismatch.
        """
        if not vad_segments:
            return {"segments": [], "language": "unknown"}

        if language and language not in ("vi", "auto"):
            logger.debug(
                f"ChunkFormer is a Vietnamese-only model; ignoring language={language!r}"
            )

        # Prefer the large scratch dir when set (Ray actors redirect TMPDIR
        # there); mirrors funasr_asr's LARGE_TEMP_DIR handling.
        custom_temp_dir = os.environ.get("LARGE_TEMP_DIR", None)
        if custom_temp_dir:
            try:
                os.makedirs(custom_temp_dir, exist_ok=True)
            except Exception as e:
                logger.warning(
                    f"Could not create LARGE_TEMP_DIR '{custom_temp_dir}', "
                    f"falling back to system default: {e}"
                )
                custom_temp_dir = None

        segments = []
        with tempfile.TemporaryDirectory(dir=custom_temp_dir) as temp_dir:
            with torch.no_grad():
                for idx, seg_info in enumerate(vad_segments):
                    start = seg_info.get("start", 0.0)
                    end = seg_info.get("end", 0.0)
                    seg_audio = audio[int(start * _SAMPLE_RATE): int(end * _SAMPLE_RATE)]

                    text = ""
                    if len(seg_audio) > 0:
                        seg_path = os.path.join(temp_dir, f"{idx}.wav")
                        try:
                            sf.write(seg_path, seg_audio, _SAMPLE_RATE)
                            out = self.model.endless_decode(
                                audio_path=seg_path, return_timestamps=False
                            )
                            text = self._extract_text(out)
                        except Exception as e:
                            logger.error(
                                f"ChunkFormer decode failed for segment {idx} "
                                f"[{start:.2f}s-{end:.2f}s]: {e}"
                            )
                            text = ""

                    segments.append(
                        {
                            "text": text,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "speaker": seg_info.get("speaker", None),
                        }
                    )

        return {"segments": segments, "language": "unknown"}


def load_asr_model(model_dir: str, device: str = "cpu", **kwargs) -> ChunkFormerASR:
    """Factory matching the other provider wrappers' shape."""
    return ChunkFormerASR(model_dir=model_dir, device=device, **kwargs)
