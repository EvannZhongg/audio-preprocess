"""Per-file result, serializable across the Ray boundary as a plain dict."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FileResult:
    audio_path: str
    success: bool
    n_segments: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileResult":
        return cls(**d)
