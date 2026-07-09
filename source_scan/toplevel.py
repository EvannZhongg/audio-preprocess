"""Stage 0: enumerate only the first-level entry names under a root.

Splits a huge tree into independent work units so stage 1 has a known total
(for progress) and can resume per top-level entry. Only names are recorded --
file-vs-directory is left to stage 1 (which tries to scandir each name) so that
stage 0 never pays a per-entry stat, which on NFSv4.0 would be a GETATTR RPC per
file and take hours on a directory with millions of entries.
"""
from __future__ import annotations

import os
from typing import Iterator

from tqdm import tqdm


def top_names(root: str) -> Iterator[str]:
    """Yield each first-level entry name under root (no recursion, no stat).

    Deliberately does NOT call is_dir()/stat(): on NFSv4.0 that triggers a
    per-entry GETATTR RPC, which is catastrophic for a directory with millions
    of entries (hours of round-trips). We only read names here; stage 1 decides
    file-vs-dir by trying to scandir each name. Hidden and .temp entries are
    skipped, matching DirectoryScanner."""
    try:
        it = os.scandir(root)
    except OSError:
        return
    with it:
        for entry in tqdm(it, desc="scan_top entries", unit="entry"):
            name = entry.name
            if name.startswith(".") or ".temp" in name:
                continue
            yield name
