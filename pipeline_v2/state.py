"""Shared state passed between PipelineV2 steps.

Each step reads the fields it needs and writes its own outputs.
Fields are Optional so the type tells you which steps have run yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

# Stamped into every exported record. Bump when processing logic changes.
# Lives here in the core layer so export (producer) and the ray driver (writer
# of failed-file rows) share one definition without an upward import.
PIPELINE_VERSION = "v2"


class SegmentRecord(TypedDict):
    """One flat row of segments_part parquet (mirrors SEGMENT_SCHEMA in
    pipeline_v2_ray/segments.py). Produced by the exporter, accumulated and
    flushed by the ray driver. A TypedDict (not a dataclass) so it needs no
    conversion to cross Ray or feed pa.Table.from_pylist. Defined here in the
    core layer so both pipeline_v2 (producer) and pipeline_v2_ray (writer) can
    import it without an upward dependency."""
    utt_id: Optional[str]             # {base}_chunk{ci}_{seg.index}; None on a failed-file row
    source: str                       # relative_path; join key to stage-0 manifest
    shard: Optional[str]              # manifest shard name; stamped by the driver (its output dir)
    pipeline_version: str
    chunk_index: Optional[int]
    chunk_audio_path: Optional[str]   # wav path RELATIVE to output_root, incl. shard prefix:
                                      #   <shard>/audios/<bucket>/<file>.wav  (None on failed-file rows)
    sample_rate: Optional[int]
    chunk_duration: Optional[float]   # seconds
    speaker_id: Optional[str]
    speaker_min_similarity: Optional[float]
    start: Optional[float]            # seconds, within the chunk
    end: Optional[float]
    seg_duration: Optional[float]
    dnsmos: Optional[float]           # nullable quality metrics
    c50: Optional[float]
    snr: Optional[float]
    # Failure reason. None for a real (successful) segment; set on a single
    # placeholder row emitted per failed file. Also reused by later stages
    # (e.g. ASR) to record a per-segment processing error.
    error: Optional[str]


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
    # Source path relative to the audio root (same key as the stage-0 manifest).
    # Its SHA-1 is the export id `base` -> stable across mount points + joinable.
    # None in --input mode; export falls back to hashing audio_path.
    relative_path: Optional[str] = None
    # Manifest shard this file belongs to (its output subdir). Threaded to export
    # so it fills the segment row's `shard` column and the <shard>/... path prefix.
    shard: Optional[str] = None
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
    # Flat per-segment records for this chunk (filled by export), collected by
    # the actor and shipped to the driver for segments_part parquet.
    export_records: Optional[list[SegmentRecord]] = None

    # Shared structured-log tag; set once at run() entry, threaded
    # into every step's logger calls.
    log_tag: dict = field(default_factory=dict)
