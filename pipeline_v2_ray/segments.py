"""Segment-level parquet output for the pipeline.

The driver accumulates per-segment records returned by actors and flushes a
`segments_part-NNNNN.parquet` into each manifest shard's `segments/` dir once
that shard reaches SEG_SHARD_SIZE segments (the remainder is flushed when the
run ends). One row per segment; the schema mirrors the fields the exporter
already computes, flattened for columnar querying (Superset / DuckDB).
"""
from __future__ import annotations

import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline_v2.state import PIPELINE_VERSION, SegmentRecord

# Flush a segment shard every this many segments (per manifest shard).
SEG_SHARD_SIZE = 100_000


SEGMENT_SCHEMA = pa.schema([
    ("utt_id", pa.string()),
    ("source", pa.string()),
    ("pipeline_version", pa.string()),
    ("chunk_index", pa.int32()),
    ("chunk_audio_path", pa.string()),
    ("sample_rate", pa.int32()),
    ("chunk_duration", pa.float64()),
    ("speaker_id", pa.string()),
    ("speaker_min_similarity", pa.float64()),
    ("start", pa.float64()),
    ("end", pa.float64()),
    ("seg_duration", pa.float64()),
    ("dnsmos", pa.float64()),
    ("c50", pa.float64()),
    ("snr", pa.float64()),
    ("error", pa.string()),          # NULL for a real segment; set on failed-file rows
])


def error_record(source: str, error: str) -> SegmentRecord:
    """A single placeholder row for a failed file: source + error set, all
    segment fields NULL. Keeps failures in the same table (queryable, and
    counted by resume so a failed file isn't retried forever)."""
    rec: SegmentRecord = {k: None for k in SEGMENT_SCHEMA.names}  # type: ignore[assignment]
    rec["source"] = source
    rec["pipeline_version"] = PIPELINE_VERSION
    rec["error"] = error
    return rec


def write_segments_shard(records: list[SegmentRecord], out_dir: str, part_index: int) -> str:
    """Write one segments shard (temp-then-rename, so a reader never sees a
    half-written file). Returns the shard path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"segments_part-{part_index:05d}.parquet")
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pylist(records, schema=SEGMENT_SCHEMA), tmp)
    os.replace(tmp, path)
    return path


def resume_state(shard_dir: str) -> tuple[set[str], int]:
    """Resume support: inspect a shard's already-written segments_part parquets.

    Returns (processed_sources, next_part_index):
      - processed_sources: the `source` (relative_path) values already recorded,
        so the driver can skip those files on a rerun. The parquet IS the
        checkpoint -- no extra bookkeeping. Files whose segments weren't flushed
        before a crash (< SEG_SHARD_SIZE tail) are reprocessed, which is
        idempotent (deterministic id + overwrite).
      - next_part_index: len(existing parts), so new flushes continue numbering
        instead of overwriting existing segments_part files.
    """
    parts = sorted(glob.glob(os.path.join(shard_dir, "segments_part-*.parquet")))
    sources: set[str] = set()
    for p in parts:
        try:
            sources.update(pq.read_table(p, columns=["source"])["source"].to_pylist())
        except Exception:
            # A corrupt/unreadable part shouldn't block resume; just skip it.
            continue
    return sources, len(parts)
