"""Stage 0: enumerate only the first level under a root.

Splits a huge tree into independent work units so stage 1 has a known total
(for progress) and can resume per top-level entry. Each entry records whether
it is a directory (stage 1 recurses into it) or a file (stage 1 takes it
directly).
"""
from __future__ import annotations

import os
from typing import Iterator

from tqdm import tqdm


def top_entries(root: str) -> Iterator[dict]:
    """Yield {name, is_dir} for each first-level entry under root (no recursion).
    Hidden and .temp entries are skipped, matching DirectoryScanner. Progress is
    reported with a tqdm counter (no total -- the level is streamed)."""
    try:
        it = os.scandir(root)
    except OSError:
        return
    with it:
        for entry in tqdm(it, desc="scan_top entries", unit="entry"):
            name = entry.name
            if name.startswith(".") or ".temp" in name:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            yield {"name": name, "is_dir": is_dir}
