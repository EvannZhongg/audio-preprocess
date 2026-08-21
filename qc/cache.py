"""Resumable cache for the GPU re-check pass.

An exhaustive `--sample-n 0` run can take hours, so losing it to a preemption or
an OOM is unacceptable. Every scored segment is appended to a jsonl file the
moment it is done, and a restart skips whatever is already there.

Design choices that matter:

  * **one file per worker.** Multiple processes appending to a single file would
    interleave partial lines under load. Per-worker files remove the contention
    entirely, and the reader just globs them.
  * **append + flush per record**, not buffered writes. A crash then costs at
    most the record in flight, and only ever truncates the final line -- which
    the reader tolerates.
  * **verdicts only.** No audio, no transcript text. The cache is diagnostics,
    and copying customer text into a scratch directory is a liability with no
    upside for the statistics being computed.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator, Optional

CACHE_DIR_NAME = "recheck_cache"


def cache_dir(work_dir: str) -> str:
    path = os.path.join(work_dir, CACHE_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def worker_cache_path(work_dir: str, worker_id: int) -> str:
    return os.path.join(cache_dir(work_dir), f"verdicts-w{worker_id:03d}.jsonl")


class VerdictWriter:
    """Append-only jsonl sink for one worker."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Flushed per record so an interrupted run stays resumable. The cost is
        # negligible next to a pyannote forward pass.
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "VerdictWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _iter_cache_files(work_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(cache_dir(work_dir), "verdicts-w*.jsonl")))


def load_done_ids(work_dir: str) -> set:
    """Ids already scored, so a resumed run does not redo them.

    A truncated final line (the crash case this cache exists for) is skipped
    rather than treated as corruption.
    """
    done: set = set()
    for path in _iter_cache_files(work_dir):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partial last line from an interrupted run
                    key = rec.get("id")
                    if key:
                        done.add(key)
        except OSError:
            continue
    return done


def iter_records(work_dir: str, kind: Optional[str] = None) -> Iterator[dict]:
    """Replay cached verdicts, optionally filtered to one `kind`.

    Reduction always reads from the cache rather than from worker return values,
    so a resumed run aggregates the previous run's work and this run's
    identically -- there is one code path, not a fresh one and a resumed one.
    """
    for path in _iter_cache_files(work_dir):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if kind is None or rec.get("kind") == kind:
                        yield rec
        except OSError:
            continue


def cached_count(work_dir: str) -> int:
    return sum(1 for _ in iter_records(work_dir))
