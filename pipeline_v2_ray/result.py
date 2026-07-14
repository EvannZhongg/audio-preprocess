"""Per-file result, serializable across the Ray boundary as a plain dict."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pipeline_v2.state import SegmentRecord


@dataclass
class FileResult:
    audio_path: str
    success: bool
    n_segments: int = 0
    error: str = ""
    # Flat per-segment records (SEGMENT_SCHEMA rows) the driver accumulates and
    # flushes to segments_part parquet. Ships back over Ray with the result.
    segments: list[SegmentRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileResult":
        return cls(**d)
