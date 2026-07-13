"""TopDirScanner: a SourceScanner driven by stage-0 top-level names.

Reads the top_part-*.parquet shards from scan_top (names only) and, for each
name, stats it to decide file-vs-directory: a directory is recursed with
DirectoryScanner, a file is yielded directly.

Top entries are expanded concurrently on a thread pool. readdir/stat are
IO-bound (they release the GIL), so on a high-latency mount like NFSv4.0 this
overlaps many directories' round-trips instead of paying them one after
another. A bounded submission window keeps memory flat on millions of entries
and preserves top-entry order for a steady progress bar.

(Only useful when the top level has many directories; a single giant flat
directory has one readdir cursor and cannot be parallelised -- shard it upstream
instead.)
"""
from __future__ import annotations

import os
import stat as stat_mod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional

from tqdm import tqdm

from source_scan.base import SourceScanner
from source_scan.directory import DirectoryScanner
from source_scan.manifest import list_shards, read_top


class TopDirScanner(SourceScanner):
    def __init__(
        self,
        root: str,
        top_dir: str,
        extensions: Optional[tuple[str, ...]] = None,
        workers: int = 64,
    ) -> None:
        self.root = os.path.abspath(root)
        self.top_dir = top_dir
        self.extensions = extensions
        self.workers = max(1, workers)

    def _expand(self, name: str) -> list[str]:
        """Resolve one top-level name to its audio paths (relative to root).
        Runs on a worker thread."""
        sub = os.path.join(self.root, name)
        try:
            st = os.stat(sub)
        except OSError:
            return []
        if stat_mod.S_ISDIR(st.st_mode):
            # DirectoryScanner yields paths relative to `sub`; re-root by
            # prefixing the top-level name.
            return [os.path.join(name, rel)
                    for rel in DirectoryScanner(sub, self.extensions).relative_paths()]
        if not self.extensions or name.lower().endswith(self.extensions):
            return [name]
        return []

    def relative_paths(self) -> Iterator[str]:
        # First-level names fit in memory (one string each), so we load them all
        # up front to get a total for the progress bar.
        names = [n for shard in list_shards(self.top_dir, "top")
                 for n in read_top(shard)]
        names_it = iter(names)
        window = self.workers * 2   # bounded in-flight futures: cap memory

        with ThreadPoolExecutor(max_workers=self.workers) as pool, \
                tqdm(total=len(names), desc="scan_paths top entries", unit="entry") as bar:
            # FIFO of futures in submission order, so results come out ordered.
            inflight: deque = deque()
            for _ in range(window):
                nxt = next(names_it, None)
                if nxt is None:
                    break
                inflight.append(pool.submit(self._expand, nxt))

            while inflight:
                paths = inflight.popleft().result()
                nxt = next(names_it, None)
                if nxt is not None:
                    inflight.append(pool.submit(self._expand, nxt))
                for p in paths:
                    yield p
                bar.update(1)
