"""Source separation (denoise / dereverb).

Two providers, same interface:
  - SMRU  (models.smru_separate.Predictor)  — neural denoise+dereverb at 48k
  - UVR   (models.separate_fast.Predictor)  — ONNX MDX-Net at 44.1k

Both expect a stereo `(C, T)` float32 mix at 44100 Hz and return
`(vocals, no_vocals)`. We resample the input from its native SR to 44100,
run prediction, then resample the vocals back. NaN/Inf in the output
falls back to the original waveform.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Optional

import librosa
import numpy as np

import logger
from pipeline_v2.params import SourceSeparationParams


_INTERNAL_SR = 44100


class Separator:
    """Loads the chosen provider once; reused per file."""

    def __init__(self, params: SourceSeparationParams, device: str) -> None:
        self.params = params
        self.device = device
        self.predictor: Any = self._load_predictor()

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> np.ndarray:
        """Return a denoised mono waveform at `sample_rate`.

        On any model error or non-finite output, returns the input waveform
        unchanged so the pipeline can continue.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        mix_44k = self._to_internal_sr(waveform, sample_rate)
        resample_in_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        vocals = self._predict(mix_44k, log_tag)
        predict_ms = int((time.perf_counter() - t0) * 1000)
        if vocals is None:
            logger.error("sep_fallback_input", extra=log_tag)
            return waveform

        t0 = time.perf_counter()
        vocals_native = self._to_native_sr(vocals, sample_rate)
        resample_out_ms = int((time.perf_counter() - t0) * 1000)

        if not np.all(np.isfinite(vocals_native)):
            logger.error("sep_nonfinite_output", extra=log_tag)
            return waveform

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"sep_time_cost provider {self.params.provider} "
            f"resample_in_ms {resample_in_ms} predict_ms {predict_ms} "
            f"resample_out_ms {resample_out_ms} total_ms {total_ms}",
            extra=log_tag,
        )
        return vocals_native

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _load_predictor(self) -> Any:
        provider = self.params.provider
        if provider == "smru":
            from models import smru_separate

            return smru_separate.Predictor(args=dict(self.params.smru_conf), device=self.device)
        elif provider == "uvr":
            from models import separate_fast

            return separate_fast.Predictor(args=dict(self.params.uvr_conf), device=self.device)
        else:
            raise ValueError(f"unknown source-separation provider: {provider!r}")

    def _to_internal_sr(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == _INTERNAL_SR:
            return waveform
        return librosa.resample(waveform, orig_sr=sample_rate, target_sr=_INTERNAL_SR)

    def _to_native_sr(self, vocals: np.ndarray, sample_rate: int) -> np.ndarray:
        """Bring predictor output back to `sample_rate`, mono `(T,)`."""
        if vocals.ndim == 1:
            mono = vocals
        else:
            mono = vocals[:, 0]

        if sample_rate == _INTERNAL_SR:
            return mono.astype(np.float32, copy=False)
        return librosa.resample(
            mono, orig_sr=_INTERNAL_SR, target_sr=sample_rate
        ).astype(np.float32, copy=False)

    def _predict(self, mix_44k: np.ndarray, log_tag: Optional[dict]) -> Optional[np.ndarray]:
        try:
            vocals, _ = self.predictor.predict(mix_44k)
            return vocals
        except RuntimeError as e:
            logger.error(f"sep_runtime_error {e}", extra=log_tag)
            return None
        except Exception:
            logger.error(f"sep_unknown_error {traceback.format_exc()}", extra=log_tag)
            return None
