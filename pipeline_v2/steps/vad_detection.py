"""Voice activity detection (silero-vad).

Splits diarization spans into finer speech segments. Wraps the legacy
SileroVAD.vad(speakerdia, audio) which returns raw dicts; we convert
them to `Segment` so downstream stages see structured types.

Failures return None + an error log line; orchestrator decides what to do.
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import torch

import logger
from models.vad import SileroVAD
from pipeline_v2.state import Segment


class VadDetector:
    """Loads silero-vad once; reused per file."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.vad_model: SileroVAD = SileroVAD(device=torch.device(device))

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        diarize_df: pd.DataFrame,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[list[Segment]]:
        if diarize_df is None or len(diarize_df) == 0:
            logger.error("vad_empty_input_diarize_df", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            raw = self.vad_model.vad(
                diarize_df, {"waveform": waveform, "sample_rate": sample_rate}
            )
        except Exception:
            logger.error(f"vad_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        vad_list = [
            Segment(
                index=s["index"],
                start=float(s["start"]),
                end=float(s["end"]),
                speaker=s["speaker"],
            )
            for s in raw
        ]
        total_ms = int((time.perf_counter() - t_total) * 1000)

        total_dur = sum(s.end - s.start for s in vad_list)
        logger.info(
            f"vad_time_cost segments {len(vad_list)} duration_s {total_dur:.2f} "
            f"total_ms {total_ms}",
            extra=log_tag,
        )
        return vad_list
