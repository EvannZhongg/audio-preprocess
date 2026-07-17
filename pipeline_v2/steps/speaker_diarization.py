"""Speaker diarization via pyannote.

Loads `pyannote/speaker-diarization-3.1` once and exposes a `run` that
turns a mono waveform into a per-segment DataFrame plus per-speaker
embedding centroids. Failures return `None` with an error log line; the
orchestrator decides what to do with that.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from pyannote.audio import Pipeline as PyannotePipeline

import logger
from pipeline_v2.params import DiarizationParams


class Diarizer:
    """Loads pyannote once; reused per file."""

    def __init__(self, params: DiarizationParams, device: str) -> None:
        self.params = params
        self.device = device
        self.dia_pipeline: PyannotePipeline = self._load_pipeline()

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[Tuple[pd.DataFrame, dict[str, np.ndarray]]]:
        t_total = time.perf_counter()

        try:
            tensor = torch.from_numpy(waveform).to(self.dia_pipeline.device).unsqueeze(0)
        except Exception as e:
            logger.error(f"dia_input_convert_failed {e}", extra=log_tag)
            return None

        t0 = time.perf_counter()
        try:
            segments, embeddings = self.dia_pipeline(
                {"waveform": tensor, "sample_rate": sample_rate},
                return_embeddings=True,
            )
        except Exception:
            logger.error(f"dia_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None
        infer_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        diarize_df = self._segments_to_df(segments)
        centroids = {spk: embeddings[i] for i, spk in enumerate(segments.labels())}
        postprocess_ms = int((time.perf_counter() - t0) * 1000)

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"dia_time_cost provider {self.params.provider} "
            f"speakers {len(centroids)} segments {len(diarize_df)} "
            f"infer_ms {infer_ms} postprocess_ms {postprocess_ms} total_ms {total_ms}",
            extra=log_tag,
        )
        return diarize_df, centroids

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> PyannotePipeline:
        provider = self.params.provider
        if provider != "pyannote":
            raise ValueError(f"unknown diarization provider: {provider!r}")

        model_ref = self._resolve_model_ref()
        dia_pipeline = PyannotePipeline.from_pretrained(
            model_ref, use_auth_token=self.params.huggingface_token
        )
        dia_pipeline.to(torch.device(self.device))
        return dia_pipeline

    def _resolve_model_ref(self) -> str:
        """Prefer the local cache yaml if it points at existing weights;
        otherwise fall back to the HF model id."""
        cache = self.params.pyannote_model_dir_cache
        if cache and os.path.exists(cache):
            with open(cache, "r") as fp:
                cfg = yaml.safe_load(fp)
            params = cfg.get("pipeline", {}).get("params", {})
            seg = params.get("segmentation")
            emb = params.get("embedding")
            if seg and emb and os.path.exists(seg) and os.path.exists(emb):
                return cache
        return self.params.pyannote_model

    @staticmethod
    def _segments_to_df(segments) -> pd.DataFrame:
        df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        df["start"] = df["segment"].apply(lambda s: s.start)
        df["end"] = df["segment"].apply(lambda s: s.end)
        return df
