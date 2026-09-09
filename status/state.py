"""Persistent state for the resident monitor, all of it inside `status/`.

Three files, each with a different durability requirement:

  * `history.jsonl` -- one JSON object per scan, append-only. The rate/ETA
    estimator reads a trailing window of it. Append-only (rather than a
    rewritten array) so a crash mid-write can damage at most the final line,
    which the reader skips.
  * `scan_cache.json` -- per-parquet-part aggregates, keyed by
    `path|mtime_ns|size`. Lets a restarted monitor skip re-reading parts it has
    already summed. Rewritten wholesale, so it goes through temp-then-rename.
  * `state.json` -- tiny bookkeeping (`last_push_ts`, `first_seen_ts`) that must
    survive a restart, otherwise the monitor would re-push to WeCom immediately
    on every restart and would lose the observation window's start.

Every whole-file write is temp-then-rename, matching how the pipeline itself
publishes parquet (`pipeline_v2_ray/segments.py:61-63`): a reader never sees a
half-written file, and a crash leaves the previous good version in place.

Only stdlib here -- this module is imported by every other one.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

HISTORY_FILE = "history.jsonl"
SCAN_CACHE_FILE = "scan_cache.json"
STATE_FILE = "state.json"

# A single history line is a handful of floats; anything much larger than this
# is a corrupt/garbage line we refuse to parse.
_MAX_HISTORY_LINE = 8192
# Only ever read the tail of history.jsonl -- the estimator's window is minutes
# wide, but the file grows for the lifetime of the job.
_TAIL_READ_BYTES = 1 << 20  # 1 MiB, ~5000 samples


def _atomic_write_json(path: str, payload: Any) -> None:
    """Serialise `payload` to `path` via temp-then-rename in the same dir.

    Same directory matters: `os.replace` is only atomic within a filesystem,
    and `status/` may well be a different mount from `/tmp`.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leave the previous good file untouched; drop the partial temp.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Optional[Any]:
    """Load a JSON file, returning None if it is absent, empty or corrupt.

    A damaged state file must degrade the monitor (it recomputes) rather than
    stop it, so every failure mode collapses to "no data".
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


class StatusState:
    """Owns the three state files for one `status/` directory."""

    def __init__(self, state_dir: str) -> None:
        self.dir = os.path.abspath(state_dir)
        self.history_path = os.path.join(self.dir, HISTORY_FILE)
        self.scan_cache_path = os.path.join(self.dir, SCAN_CACHE_FILE)
        self.state_path = os.path.join(self.dir, STATE_FILE)
        os.makedirs(self.dir, exist_ok=True)

    # -- history ----------------------------------------------------------
    def append_history(self, sample: Dict[str, Any]) -> None:
        """Append one scan sample. A single `write` of a line under the pipe
        buffer is effectively atomic for our purposes; a torn tail line is
        tolerated by `read_history`."""
        line = json.dumps(sample, ensure_ascii=False)
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def read_history(self) -> List[Dict[str, Any]]:
        """Trailing history samples, oldest first.

        Reads only the last `_TAIL_READ_BYTES` and discards the first
        (possibly mid-line) fragment, so cost stays constant no matter how long
        the job has run. Unparseable lines are skipped rather than fatal.
        """
        try:
            size = os.path.getsize(self.history_path)
            with open(self.history_path, "rb") as handle:
                if size > _TAIL_READ_BYTES:
                    handle.seek(size - _TAIL_READ_BYTES)
                    handle.readline()  # discard the partial first line
                raw = handle.read()
        except OSError:
            return []

        samples: List[Dict[str, Any]] = []
        for chunk in raw.split(b"\n"):
            if not chunk.strip() or len(chunk) > _MAX_HISTORY_LINE:
                continue
            try:
                obj = json.loads(chunk.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue  # torn final line, or hand-edited garbage
            if isinstance(obj, dict) and isinstance(obj.get("ts"), (int, float)):
                samples.append(obj)
        samples.sort(key=lambda s: s["ts"])
        return samples

    # -- scan cache -------------------------------------------------------
    def load_scan_cache(self) -> Dict[str, Any]:
        payload = _read_json(self.scan_cache_path)
        return payload if isinstance(payload, dict) else {}

    def save_scan_cache(self, cache: Dict[str, Any]) -> None:
        _atomic_write_json(self.scan_cache_path, cache)

    # -- misc state -------------------------------------------------------
    def load_state(self) -> Dict[str, Any]:
        payload = _read_json(self.state_path)
        return payload if isinstance(payload, dict) else {}

    def save_state(self, state: Dict[str, Any]) -> None:
        _atomic_write_json(self.state_path, state)
