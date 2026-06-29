"""Shared state passed between PipelineV2 steps.

Each step reads the fields it needs and writes its own outputs.
Fields are Optional so the type tells you which steps have run yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Segment:
    """A single span produced by VAD; mutated through downstream stages.

    Embedding-related fields are filled by `EmbeddingRefiner`:
      - `reference_embedding`: per-segment identity, used by `Segmenter`
        to decide whether adjacent same-speaker segments should be merged.
      - `min_similarity`: per-segment internal consistency score (min
        cosine sim across sliding windows vs reference). Carried forward
        as a quality signal for downstream ASR / metrics stages.
    """
    index: str
    start: float                                    # seconds
    end: float                                      # seconds
    speaker: str
    reference_embedding: Optional[np.ndarray] = None
    min_similarity: float = -1.0
    dnsmos: Optional[float] = None
    c50: Optional[float] = None
    snr: Optional[float] = None


@dataclass
class PipelineState:
    # Step 0
    audio_path: str
    waveform: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None

    # Step 2
    diarize_df: Optional[pd.DataFrame] = None
    speaker_centroids: Optional[dict[str, np.ndarray]] = None

    # Step 3 + 3.5 (VAD + embedding refinement)
    vad_list: Optional[list[Segment]] = None

    # Step 4 (segmenter)
    segment_list: Optional[list[Segment]] = None

    # Step 5 (export)
    export_path: Optional[str] = None

    # Shared structured-log tag; set once at run() entry, threaded
    # into every step's logger calls.
    log_tag: dict = field(default_factory=dict)
