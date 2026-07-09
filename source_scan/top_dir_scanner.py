"""TopDirScanner: a SourceScanner driven by stage-0 top-level names.

Reads the top_part-*.parquet shards produced by scan_top (names only, no type)
and, for each name, decides file-vs-directory by *attempting* to scandir it:
opening it as a directory succeeds -> recurse; NotADirectoryError -> it's a
file, yield it. This avoids a separate stat/GETATTR per entry (the directory
open we need anyway doubles as the type check), which matters on NFSv4.0 where
each GETATTR is a network round-trip.

Serial, but the known total number of top names gives a real progress readout.
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
        # First-level names fit in memory (one string each), so we load them all
        # up front to get a total for the progress bar.
        names = [n for shard in list_shards(self.top_dir, "top")
                 for n in read_top(shard)]
        for name in tqdm(names, desc="scan_paths top entries", unit="entry"):
            sub = os.path.join(self.root, name)
            try:
                probe = os.scandir(sub)
            except NotADirectoryError:
                # `name` is a file: emit it directly (applying the extension
                # filter, if any).
                if not self.extensions or name.lower().endswith(self.extensions):
                    yield name
                continue
            except OSError:
                # Vanished / unreadable entry -- skip it.
                continue
            probe.close()
            # `name` is a directory: recurse. DirectoryScanner yields paths
            # relative to `sub`, so re-root them by prefixing `name`.
            for rel in DirectoryScanner(sub, self.extensions).relative_paths():
                yield os.path.join(name, rel)
