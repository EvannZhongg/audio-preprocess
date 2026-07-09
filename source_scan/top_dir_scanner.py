"""TopDirScanner: a SourceScanner driven by stage-0 top-level entries.

Reads the top_part-*.parquet shards produced by scan_top, and for each entry:
  * a file      -> yields it directly,
  * a directory -> recurses with DirectoryScanner.
Serial, but the known total number of top entries gives a real i/total progress
readout, which is the whole point of splitting enumeration into two steps.
"""
from __future__ import annotations

import os
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
    ) -> None:
        self.root = os.path.abspath(root)
        self.top_dir = top_dir
        self.extensions = extensions

    def relative_paths(self) -> Iterator[str]:
        # First-level entries fit in memory (one dict per top entry), so we
        # load them all up front to get a total for the progress bar.
        entries = [e for shard in list_shards(self.top_dir, "top")
                   for e in read_top(shard)]
        for entry in tqdm(entries, desc="scan_paths top entries", unit="entry"):
            name = entry["name"]
            if entry["is_dir"]:
                sub = os.path.join(self.root, name)
                # DirectoryScanner yields paths relative to `sub`; re-root them
                # to `self.root` by prefixing the top-level directory name.
                for rel in DirectoryScanner(sub, self.extensions).relative_paths():
                    yield os.path.join(name, rel)
            else:
                if self.extensions and not name.lower().endswith(self.extensions):
                    continue
                yield name
