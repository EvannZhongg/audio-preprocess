"""Segment-level parquet output for stage 2 (remote ASR + v1 post-processing).

Mirrors `pipeline_v2_ray/segments.py`'s shard/resume/flush contract exactly
(same function shapes: `write_*_shard`, `resume_state*`, `error_record*`, so
`ClusterDriver` can be handed either pair via dependency injection), but is
keyed by `chunk_audio_path` rather than the stage-1 file-level
`relative_path` -- that's the granularity stage 2 actually processes (one
stage-1 chunk wav -> one ASR + postprocessing call -> one row group here),
and using a distinct `stage2_segments_part-*.parquet` file name lets stage 2
write into the very same shard directory as stage 1 without colliding with
stage 1's own `segments_part-*.parquet` / resume bookkeeping.
"""
from __future__ import annotations

import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

import logger
from pipeline_v2.state import (ASR_ACCESS_FAILED_MARKER, PIPELINE_VERSION,
                               Stage2SegmentRecord)

SEG_SHARD_SIZE = 100_000

STAGE2_SEGMENT_SCHEMA = pa.schema(
    [
        ("utt_id", pa.string()),
        ("source", pa.string()),  # chunk_audio_path; resume/dedup key for stage 2
        ("origin_source", pa.string()),  # stage-1 file-level relative_path
        ("shard", pa.string()),
        ("pipeline_version", pa.string()),
        ("chunk_index", pa.int32()),
        ("chunk_audio_path", pa.string()),
        ("start", pa.float64()),
        ("end", pa.float64()),
        ("seg_duration", pa.float64()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("language", pa.string()),
        ("domain_text", pa.string()),
        ("domain_acoustic", pa.string()),
        ("domain_speaker", pa.string()),
        ("speaking_rate", pa.float64()),
        ("alignment_score", pa.float64()),
        ("ppl", pa.float64()),
        ("llm_text_score", pa.float64()),
        ("dropped_by_silence", pa.bool_()),
        ("dropped_by_alignment", pa.bool_()),
        ("dropped_by_text_quality", pa.bool_()),
        ("dropped_by_speaking_rate", pa.bool_()),
        ("dropped_by_asr_validation", pa.bool_()),
        ("asr_wer", pa.float64()),
        ("asr_val_text", pa.string()),
        ("error", pa.string()),
    ]
)


def error_record2(source: str, shard: str, error: str) -> Stage2SegmentRecord:
    """One placeholder row for a chunk whose stage-2 processing failed
    entirely, so resume can skip it without retrying forever (mirrors
    `segments.error_record`)."""
    rec = {name: None for name in STAGE2_SEGMENT_SCHEMA.names}
    rec["source"] = source
    rec["shard"] = shard
    rec["pipeline_version"] = PIPELINE_VERSION
    rec["dropped_by_silence"] = False
    rec["dropped_by_alignment"] = False
    rec["dropped_by_text_quality"] = False
    rec["dropped_by_speaking_rate"] = False
    rec["dropped_by_asr_validation"] = False
    rec["error"] = error
    return rec  # type: ignore[return-value]


def write_stage2_segments_shard(records: list, out_dir: str, part_index: int) -> str:
    """Write one shard of stage-2 rows. Atomic (tmp + rename) so a crash mid-write
    never leaves a corrupt parquet that resume would trip over."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"stage2_segments_part-{part_index:05d}.parquet")
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pylist(records, schema=STAGE2_SEGMENT_SCHEMA), tmp)
    os.replace(tmp, path)
    return path


def _is_retriable_error(error) -> bool:
    """Whether an `error` cell marks a failure worth retrying on the next run.

    Only remote-ASR access failures qualify: they're transient
    (service drained / timed out / answered garbage), not a property of the
    chunk itself. Every other failure (poison chunk, actor crash, task error)
    keeps the original "record it and never retry" semantics, since retrying
    those just reproduces the crash.
    """
    return isinstance(error, str) and ASR_ACCESS_FAILED_MARKER in error


def resume_state2(shard_dir: str) -> tuple[set, int]:
    """Read every existing `stage2_segments_part-*.parquet` in `shard_dir` and
    return the set of already-processed `source` (chunk_audio_path) values
    plus the next free part index (mirrors `segments.resume_state`).

    Rows whose `error` is a retriable remote-ASR failure are deliberately
    excluded from the "done" set, so those chunks get reprocessed instead of
    being skipped forever with a lost transcript. Membership is a union over
    all parts ("any non-retriable row wins"), which converges correctly when an
    older part holds the ASR failure row and a newer part holds the successful
    retry."""
    parts = sorted(glob.glob(os.path.join(shard_dir, "stage2_segments_part-*.parquet")))
    sources: set = set()
    retriable: set = set()
    for p in parts:
        try:
            table = pq.read_table(p, columns=["source", "error"])
        except Exception:  # noqa: BLE001 - a truncated/corrupt shard shouldn't block resume
            continue
        for source, error in zip(
            table["source"].to_pylist(), table["error"].to_pylist()
        ):
            if _is_retriable_error(error):
                retriable.add(source)
            else:
                sources.add(source)
    # Only chunks with *no* completed row anywhere are handed back for retry.
    pending_retry = retriable - sources
    if pending_retry:
        logger.info(
            f"stage2_resume_retry dir {shard_dir} retry {len(pending_retry)} chunks "
            f"previously failed with {ASR_ACCESS_FAILED_MARKER}"
        )
    return sources, len(parts)
