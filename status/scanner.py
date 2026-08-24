"""One scan pass over a pipeline_v3 output tree.

Produces a `Snapshot`: the three levels of progress (manifest -> stage 1 ->
stage 2) plus enough scan metadata for the estimator to reason about elapsed
time. Deliberately holds no parquet rows -- only aggregates and, for stage 1,
the set of finished source paths, which is orders of magnitude smaller than the
row count (one entry per input file, not per segment).

Two things here are load-bearing and easy to get wrong:

**"Hours of raw audio processed" cannot come from the stage-1 parquet.** That
table has no raw-file duration column: `chunk_duration` is the length of an
exported chunk wav and `seg_duration` the length of a segment, both already
shrunk by denoise/VAD/quality filtering. So stage 1's progress is measured by
taking the distinct `source` values (which are manifest `relative_path`s, see
`pipeline_v2_ray/segments.py:26`) and summing those files' manifest durations.
That is exactly the quantity the driver itself tracks
(`pipeline_v3/driver.py:378` adds `item.duration`, which came from the
manifest), so the monitor and the pipeline's own logs agree.

**Failed files count as processed.** A stage-1 failure is written as one
placeholder row with `source` set and everything else NULL
(`pipeline_v2_ray/segments.py:44-53`), and `resume_state` treats those files as
done, i.e. a rerun will not retry them. They consumed GPU time and they will
never be revisited, so excluding them would make progress permanently
understate itself and the ETA never converge.

Validity semantics are NOT re-implemented: every mask comes from
`qc/loaders.py`, which documents itself as the single source of truth for what
"valid" means at each stage. A monitor that defined "kept segment" slightly
differently from the pipeline would be worse than no monitor at all.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from qc.layout import discover_manifest_parts, discover_shards
from qc.loaders import (DROP_COLUMNS, ParquetReadError, all_true_mask,
                        classify_error, mask_count, masked_sum, read_columns,
                        stage1_error_mask, stage1_valid_mask, stage2_drop_mask,
                        stage2_kept_mask, stage2_retriable_error_mask)

# Only the columns each stage's aggregation actually needs. Column pruning is
# what keeps a 100k-row part cheap to re-read (`SEG_SHARD_SIZE = 100_000`).
_STAGE1_COLUMNS = ("source", "error", "seg_duration")
_STAGE2_COLUMNS = ("seg_duration", "error") + tuple(DROP_COLUMNS)

# Cap on how many distinct stage-1 error strings to carry in a report.
_MAX_ERROR_TYPES = 8
# Cap on how many unreadable part paths to name in a report (the rest are
# counted): a bad mount would otherwise flood the log every five minutes.
_MAX_FAILED_SAMPLES = 5


@dataclass
class Snapshot:
    """Everything one scan pass learned. Aggregates only, no parquet rows."""

    ts: float

    # which output tree this snapshot describes; the report shows its tail so a
    # reader can tell several concurrent monitors apart at a glance.
    output_root: str = ""

    # level 0 -- the raw corpus, from build_manifest.py's manifest
    total_secs: float = 0.0
    total_files: int = 0
    files_unknown_duration: int = 0
    manifest_available: bool = False

    # level 1 -- stage 1
    stage1_done_secs: float = 0.0        # manifest duration of finished files
    stage1_done_files: int = 0           # distinct sources, success + failed
    stage1_failed_files: int = 0
    stage1_matched_files: int = 0        # finished files found in the manifest
    stage1_valid_segments: int = 0
    stage1_valid_secs: float = 0.0       # sum(seg_duration) where error IS NULL
    stage1_error_types: Dict[str, int] = field(default_factory=dict)

    # level 2 -- stage 2, the pipeline's final output
    stage2_total_secs: float = 0.0
    stage2_total_rows: int = 0
    stage2_kept_secs: float = 0.0
    stage2_kept_rows: int = 0
    stage2_drop_secs: Dict[str, float] = field(default_factory=dict)
    stage2_drop_rows: Dict[str, int] = field(default_factory=dict)
    stage2_error_rows: int = 0
    stage2_retriable_error_rows: int = 0

    # scan metadata
    shards_total: int = 0
    shards_with_stage1: int = 0
    shards_with_stage2: int = 0
    parts_total: int = 0
    parts_read: int = 0
    parts_cached: int = 0
    parts_failed: int = 0
    failed_part_samples: List[str] = field(default_factory=list)
    min_part_mtime: Optional[float] = None
    max_part_mtime: Optional[float] = None
    scan_secs: float = 0.0

    @property
    def progress_pct(self) -> Optional[float]:
        if not self.manifest_available or self.total_secs <= 0:
            return None
        return 100.0 * self.stage1_done_secs / self.total_secs

    def history_sample(self) -> Dict[str, float]:
        """The subset the estimator needs, kept deliberately small since it is
        appended to `history.jsonl` on every single scan."""
        return {
            "ts": self.ts,
            "stage1_done_secs": self.stage1_done_secs,
            "stage1_done_files": self.stage1_done_files,
            "stage2_kept_secs": self.stage2_kept_secs,
            "total_secs": self.total_secs,
        }


@dataclass
class _PartAggregate:
    """One parquet part's contribution. Cacheable because a published part is
    immutable: parts are written temp-then-rename, so once `*.parquet` is
    visible its bytes never change again."""

    rows: int = 0
    # stage 1
    sources: Tuple[str, ...] = ()
    failed_sources: Tuple[str, ...] = ()
    error_types: Dict[str, int] = field(default_factory=dict)
    valid_segments: int = 0
    valid_secs: float = 0.0
    # stage 2
    total_secs: float = 0.0
    kept_secs: float = 0.0
    kept_rows: int = 0
    drop_secs: Dict[str, float] = field(default_factory=dict)
    drop_rows: Dict[str, int] = field(default_factory=dict)
    error_rows: int = 0
    retriable_error_rows: int = 0

    def to_json(self) -> dict:
        return {
            "rows": self.rows,
            "sources": list(self.sources),
            "failed_sources": list(self.failed_sources),
            "error_types": self.error_types,
            "valid_segments": self.valid_segments,
            "valid_secs": self.valid_secs,
            "total_secs": self.total_secs,
            "kept_secs": self.kept_secs,
            "kept_rows": self.kept_rows,
            "drop_secs": self.drop_secs,
            "drop_rows": self.drop_rows,
            "error_rows": self.error_rows,
            "retriable_error_rows": self.retriable_error_rows,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "_PartAggregate":
        return cls(
            rows=int(payload.get("rows", 0)),
            sources=tuple(payload.get("sources") or ()),
            failed_sources=tuple(payload.get("failed_sources") or ()),
            error_types=dict(payload.get("error_types") or {}),
            valid_segments=int(payload.get("valid_segments", 0)),
            valid_secs=float(payload.get("valid_secs", 0.0)),
            total_secs=float(payload.get("total_secs", 0.0)),
            kept_secs=float(payload.get("kept_secs", 0.0)),
            kept_rows=int(payload.get("kept_rows", 0)),
            drop_secs=dict(payload.get("drop_secs") or {}),
            drop_rows=dict(payload.get("drop_rows") or {}),
            error_rows=int(payload.get("error_rows", 0)),
            retriable_error_rows=int(payload.get("retriable_error_rows", 0)),
        )


def _cache_key(path: str, stat: os.stat_result) -> str:
    """Identify a part by path + mtime + size.

    Sufficient precisely because published parts are immutable: a differing
    mtime or size means a genuinely different file (a rewritten shard after a
    rerun), which correctly misses the cache.
    """
    return f"{path}|{stat.st_mtime_ns}|{stat.st_size}"


def _scan_stage1_part(path: str) -> _PartAggregate:
    """Aggregate one `segments_part-*.parquet`.

    `pc.unique` does the dedup inside Arrow's kernel and only the deduped
    values are converted to Python, so a 100k-row part typically materialises a
    few thousand strings instead of 100k. That is what makes the thread pool
    (rather than a process pool) the right choice: the expensive parts release
    the GIL and there is no large object to pickle back.
    """
    table = read_columns(path, _STAGE1_COLUMNS)
    agg = _PartAggregate(rows=table.num_rows)
    if table.num_rows == 0:
        return agg

    err_mask = stage1_error_mask(table)
    valid_mask = stage1_valid_mask(table)

    source_col = table["source"]
    if not pa.types.is_null(source_col.type):
        agg.sources = tuple(
            s for s in pc.unique(source_col).to_pylist() if s is not None
        )
        # Failed files are tracked separately so a file that failed can be
        # reported as such, while still counting as processed.
        failed = pc.filter(source_col, err_mask)
        if len(failed) > 0:
            agg.failed_sources = tuple(
                s for s in pc.unique(failed).to_pylist() if s is not None
            )

    agg.valid_segments = mask_count(valid_mask)
    agg.valid_secs = masked_sum(table, "seg_duration", valid_mask)

    error_col = table["error"]
    if not pa.types.is_null(error_col.type):
        # Bucket by category, not raw string: the detail after the first colon
        # is high-cardinality noise (see qc.loaders.classify_error).
        for raw in pc.filter(error_col, err_mask).to_pylist():
            key = classify_error(raw)
            agg.error_types[key] = agg.error_types.get(key, 0) + 1
    return agg


def _scan_stage2_part(path: str) -> _PartAggregate:
    """Aggregate one `stage2_segments_part-*.parquet`.

    "Kept" is `qc.loaders.stage2_kept_mask` verbatim: no error and none of the
    five `dropped_by_*` flags set. `tmp/stat_stage2.py` is an independent
    implementation of the same predicate and produces identical numbers.
    """
    table = read_columns(path, _STAGE2_COLUMNS)
    agg = _PartAggregate(rows=table.num_rows)
    if table.num_rows == 0:
        return agg

    kept_mask = stage2_kept_mask(table)
    agg.total_secs = masked_sum(table, "seg_duration", all_true_mask(table.num_rows))
    agg.kept_secs = masked_sum(table, "seg_duration", kept_mask)
    agg.kept_rows = mask_count(kept_mask)
    # Drop reasons are NOT mutually exclusive -- one segment can trip several --
    # so these sum to more than the dropped total by design.
    for column in DROP_COLUMNS:
        drop_mask = stage2_drop_mask(table, column)
        agg.drop_rows[column] = mask_count(drop_mask)
        agg.drop_secs[column] = masked_sum(table, "seg_duration", drop_mask)
    agg.error_rows = mask_count(stage1_error_mask(table))
    agg.retriable_error_rows = mask_count(stage2_retriable_error_mask(table))
    return agg


class OutputScanner:
    """Scans one output tree repeatedly, reusing per-part aggregates.

    Holds the manifest as a two-column Arrow table for the process lifetime: on
    a ten-million-file corpus a Python `dict` of path -> duration would cost
    gigabytes, whereas the columnar form stays compact and `pc.is_in` builds its
    hash table in Arrow.
    """

    def __init__(self, output_root: str, manifest: Optional[str] = None,
                 workers: int = 16, state=None) -> None:
        self.output_root = os.path.abspath(output_root)
        self.manifest = manifest
        self.workers = max(1, workers)
        self.state = state

        self._manifest_table: Optional[pa.Table] = None
        self._manifest_loaded = False
        self._manifest_total_secs = 0.0
        self._manifest_total_files = 0
        self._manifest_unknown_duration = 0

        self._cache: Dict[str, _PartAggregate] = {}
        if state is not None:
            for key, payload in state.load_scan_cache().items():
                if isinstance(payload, dict):
                    try:
                        self._cache[key] = _PartAggregate.from_json(payload)
                    except (TypeError, ValueError):
                        continue

    # -- manifest ---------------------------------------------------------
    def _load_manifest(self) -> None:
        """Read the manifest once; it does not change while a job runs.

        A partly-readable manifest is still used: whatever shards parsed give a
        denominator that is too small but honest, which is more useful than
        refusing to report progress at all.
        """
        self._manifest_loaded = True
        if not self.manifest:
            return
        parts = discover_manifest_parts(self.manifest)
        if not parts:
            return

        tables = []
        for part in parts:
            try:
                tables.append(read_columns(part, ("relative_path", "duration")))
            except ParquetReadError:
                continue
            except Exception:  # noqa: BLE001 - a bad shard must not stop the monitor
                continue
        if not tables:
            return

        table = pa.concat_tables(tables)
        durations = pc.fill_null(pc.cast(table["duration"], pa.float64()), 0.0)
        table = table.set_column(
            table.schema.get_field_index("duration"), "duration", durations
        )
        self._manifest_table = table
        self._manifest_total_files = table.num_rows
        self._manifest_total_secs = float(pc.sum(durations).as_py() or 0.0)
        # `probe()` records 0.0 when mutagen could not read a header
        # (source_scan/manifest.py:148-157); those files inflate the file count
        # but contribute no seconds, so the denominator is slightly optimistic.
        self._manifest_unknown_duration = mask_count(
            pc.fill_null(pc.less_equal(durations, 0.0), False)
        )

    def _manifest_duration_of(self, sources: Set[str]) -> Tuple[float, int]:
        """Total manifest duration of `sources`, and how many were found.

        The returned count is what makes a silent path-convention mismatch
        visible: if stage 1's `source` values never match the manifest's
        `relative_path` values, matched stays 0 while done_files climbs, and the
        report says so instead of showing an impossible 0% forever.
        """
        table = self._manifest_table
        if table is None or not sources:
            return 0.0, 0
        mask = pc.is_in(table["relative_path"], value_set=pa.array(sorted(sources)))
        mask = pc.fill_null(mask, False)
        matched = mask_count(mask)
        total = float(
            pc.sum(pc.if_else(mask, table["duration"], 0.0)).as_py() or 0.0
        )
        return total, matched

    # -- parts ------------------------------------------------------------
    def _gather_parts(self, snap: Snapshot) -> List[Tuple[str, str]]:
        """All parquet parts to account for, as (stage, path).

        `discover_shards` already excludes `_`-prefixed directories (so
        `_qc_reports` is skipped) and only matches `*.parquet`, so a
        `*.parquet.tmp` still being renamed into place is never picked up.
        """
        shards = discover_shards(self.output_root)
        snap.shards_total = len(shards)
        parts: List[Tuple[str, str]] = []
        for shard in shards:
            if shard.has_stage1:
                snap.shards_with_stage1 += 1
            if shard.has_stage2:
                snap.shards_with_stage2 += 1
            parts.extend(("stage1", p) for p in shard.stage1_parts)
            parts.extend(("stage2", p) for p in shard.stage2_parts)
        return parts

    def _aggregate_part(self, stage: str, path: str) -> Tuple[str, _PartAggregate, bool, Optional[float]]:
        """Return (stage, aggregate, from_cache, mtime); raises on read failure."""
        stat = os.stat(path)
        key = _cache_key(path, stat)
        cached = self._cache.get(key)
        if cached is not None:
            return stage, cached, True, stat.st_mtime
        agg = (_scan_stage1_part if stage == "stage1" else _scan_stage2_part)(path)
        self._cache[key] = agg
        return stage, agg, False, stat.st_mtime

    # -- public API -------------------------------------------------------
    def scan(self) -> Snapshot:
        started = time.time()
        snap = Snapshot(ts=started, output_root=self.output_root)

        if not self._manifest_loaded:
            self._load_manifest()
        snap.manifest_available = self._manifest_table is not None
        snap.total_secs = self._manifest_total_secs
        snap.total_files = self._manifest_total_files
        snap.files_unknown_duration = self._manifest_unknown_duration

        parts = self._gather_parts(snap)
        snap.parts_total = len(parts)

        stage1_sources: Set[str] = set()
        stage1_failed: Set[str] = set()
        error_types: Dict[str, int] = {}
        drop_secs: Dict[str, float] = {c: 0.0 for c in DROP_COLUMNS}
        drop_rows: Dict[str, int] = {c: 0 for c in DROP_COLUMNS}
        mtimes: List[float] = []

        def run(item: Tuple[str, str]):
            stage, path = item
            try:
                return self._aggregate_part(stage, path), None
            except ParquetReadError as exc:
                return None, (path, str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad part, not a bad run
                return None, (path, f"{type(exc).__name__}: {exc}")

        if parts:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(parts))) as pool:
                results = list(pool.map(run, parts))
        else:
            results = []

        for ok, failure in results:
            if failure is not None:
                snap.parts_failed += 1
                if len(snap.failed_part_samples) < _MAX_FAILED_SAMPLES:
                    snap.failed_part_samples.append(f"{failure[0]}: {failure[1]}")
                continue
            stage, agg, from_cache, mtime = ok
            if from_cache:
                snap.parts_cached += 1
            else:
                snap.parts_read += 1
            if mtime is not None:
                mtimes.append(mtime)

            if stage == "stage1":
                stage1_sources.update(agg.sources)
                stage1_failed.update(agg.failed_sources)
                snap.stage1_valid_segments += agg.valid_segments
                snap.stage1_valid_secs += agg.valid_secs
                for key, count in agg.error_types.items():
                    error_types[key] = error_types.get(key, 0) + count
            else:
                snap.stage2_total_rows += agg.rows
                snap.stage2_total_secs += agg.total_secs
                snap.stage2_kept_secs += agg.kept_secs
                snap.stage2_kept_rows += agg.kept_rows
                snap.stage2_error_rows += agg.error_rows
                snap.stage2_retriable_error_rows += agg.retriable_error_rows
                for column in DROP_COLUMNS:
                    drop_secs[column] += agg.drop_secs.get(column, 0.0)
                    drop_rows[column] += agg.drop_rows.get(column, 0)

        # A file can appear in several parts (its chunks straddle a flush
        # boundary), so dedup is global rather than per part.
        snap.stage1_done_files = len(stage1_sources)
        snap.stage1_failed_files = len(stage1_failed)
        snap.stage1_error_types = dict(
            sorted(error_types.items(), key=lambda kv: -kv[1])[:_MAX_ERROR_TYPES]
        )
        snap.stage2_drop_secs = drop_secs
        snap.stage2_drop_rows = drop_rows

        done_secs, matched = self._manifest_duration_of(stage1_sources)
        snap.stage1_done_secs = done_secs
        snap.stage1_matched_files = matched

        if mtimes:
            snap.min_part_mtime = min(mtimes)
            snap.max_part_mtime = max(mtimes)

        self._prune_cache(parts)
        snap.scan_secs = time.time() - started
        return snap

    def _prune_cache(self, parts: Sequence[Tuple[str, str]]) -> None:
        """Drop cache entries for parts that no longer exist under their old
        identity, so a long-running monitor's cache tracks the tree instead of
        growing forever."""
        live: Set[str] = set()
        for _, path in parts:
            try:
                live.add(_cache_key(path, os.stat(path)))
            except OSError:
                continue
        if live:
            self._cache = {k: v for k, v in self._cache.items() if k in live}

    def persist_cache(self) -> None:
        if self.state is None:
            return
        self.state.save_scan_cache({k: v.to_json() for k, v in self._cache.items()})
