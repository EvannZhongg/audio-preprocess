"""Shared plain-data types for pipeline_v3.

Kept dependency-light (no ray, no pipeline_v2_ray import) so config.py and
stages.py can import this without pulling in the scheduler or ray itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FileItem:
    """One unit of work for one stage's actor pool. Mirrors
    `pipeline_v2_ray.driver.FileItem` (duplicated, not imported: pipeline_v3
    evolves its own multi-stage scheduling independently of pipeline_v2_ray,
    which stays unmodified)."""
    audio_path: str      # full filesystem path to decode/read
    relative_path: str   # resume/dedup key; also the export id (stage_1) or
                          # chunk_audio_path (stage_2, per STAGE2_SEGMENT_SCHEMA.source)
    duration: float = 0.0  # source audio seconds; for RTF throughput / ETA
    # Extra per-file context handed straight to process_file(); unused by
    # stage_1 (raw-audio decode). stage_2 uses it to carry the stage-1
    # segments (start/end/speaker_id/utt_id/...) for the chunk wav being
    # re-processed, so the scheduling/pool/resume machinery stays generic
    # across every stage.
    payload: Any = None
