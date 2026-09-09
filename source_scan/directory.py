"""DirectoryScanner: the common "dataset is a directory tree of audio files"
scanner. Streams paths with os.scandir so a directory holding millions of
entries never has to be materialised in memory at once.

By default it yields every file (bar hidden/temp), so the manifest's `format`
column exposes the full extension distribution for inspection. Pass
`extensions` to restrict to a whitelist when a dataset needs it.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from source_scan.base import SourceScanner


class DirectoryScanner(SourceScanner):
    def __init__(
        self,
        root: str,
        extensions: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.root = os.path.abspath(root)
        # None -> keep every file; a tuple -> keep only these extensions.
        self.extensions = tuple(e.lower() for e in extensions) if extensions else None

    def _walk(self, dirpath: str) -> Iterator[str]:
        """Depth-first stream of files under dirpath, using scandir so no
        single directory's entries are all held in memory at once."""
        try:
            it = os.scandir(dirpath)
        except OSError:
            return
        with it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        yield from self._walk(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        name = entry.name
                        if name.startswith(".") or ".temp" in name:
                            continue
                        if self.extensions and not name.lower().endswith(self.extensions):
                            continue
                        yield entry.path
                except OSError:
                    continue

    def relative_paths(self) -> Iterator[str]:
        for abs_path in self._walk(self.root):
            yield os.path.relpath(abs_path, self.root)
