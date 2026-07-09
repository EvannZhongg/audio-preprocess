"""SourceScanner interface.

Stage 1 of manifest building is *enumeration*: turn an arbitrarily-organised
source dataset into a flat stream of audio paths, relative to the dataset root.
All per-dataset layout differences (millions of files in one dir, deep trees,
manifest-driven listings, custom filtering) live in SourceScanner subclasses.

Duration / size / format probing is NOT done here -- that is the generic
stage 2 (see manifest.probe). Keeping enumeration separate means stage 2 knows
the total up front (for progress + sharded resume) and is dataset-agnostic.

Pure interface: no torch / ray / pipeline_v2 imports, so a manifest can be
built on any machine without the GPU stack. Paths are relative -- the root is
supplied by the consumer at read time, so data can move without a rescan.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4", ".ogg", ".webm")


class SourceScanner(ABC):
    """Yields audio file paths (relative to the dataset root) for one dataset."""

    @abstractmethod
    def relative_paths(self) -> Iterator[str]:
        ...
