"""Parquet and audio reading, and -- more importantly -- the validity masks.

The mask helpers here are the single source of truth for what "valid" means at
each stage. Every analyzer goes through them, because the one way to make a QC
report actively harmful is to define "kept segment" slightly differently from
the pipeline. Both masks are transcriptions of existing, verified logic:

  * stage 1: a failed *file* is recorded as one placeholder row with `error`
    set and every other column NULL (pipeline_v2_ray/segments.py:44-53), so a
    real segment is exactly `error IS NULL`.
  * stage 2: `tmp/stat_stage2.py:35-44` -- no error, and none of the five
    `dropped_by_*` flags set (a NULL flag counts as False).

Only pyarrow/soundfile/numpy here; nothing torch-shaped, since these functions
run inside the parquet process pool.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qc.config import ASR_RETRIABLE_MARKER

DROP_COLUMNS = (
    "dropped_by_silence",
    "dropped_by_alignment",
    "dropped_by_text_quality",
    "dropped_by_speaking_rate",
    "dropped_by_asr_validation",
)


class ParquetReadError(Exception):
    """A part file could not be read. Callers count and skip, never abort.

    Matches the resume logic's stance (pipeline_v2_ray/segments.py:84-86): one
    truncated shard -- e.g. a `.parquet` still being renamed into place --
    must not take down a whole pass.
    """


def read_columns(path: str, columns: Iterable[str]) -> pa.Table:
    """Read `columns` from a parquet file, tolerating absent columns.

    Older parts may predate a schema addition, so requested-but-missing
    columns come back as all-NULL rather than raising -- otherwise QC could
    not analyse a tree written across a schema change.
    """
    wanted = list(columns)
    try:
        available = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:  # noqa: BLE001 - unreadable footer == unreadable part
        raise ParquetReadError(f"{path}: {type(exc).__name__}: {exc}") from exc

    present = [c for c in wanted if c in available]
    try:
        table = pq.read_table(path, columns=present)
    except Exception as exc:  # noqa: BLE001
        raise ParquetReadError(f"{path}: {type(exc).__name__}: {exc}") from exc

    missing = [c for c in wanted if c not in available]
    for name in missing:
        table = table.append_column(
            name, pa.nulls(table.num_rows, type=pa.null())
        )
    return table.select(wanted)


# ---------------------------------------------------------------------------
# validity masks
# ---------------------------------------------------------------------------

def _all_false(n: int):
    """A length-n boolean array of False, usable in and_kleene/or_kleene."""
    return pc.fill_null(pa.nulls(n, type=pa.bool_()), False)


def all_true_mask(n: int):
    """A length-n boolean array of True, for "sum this column over all rows"."""
    return pc.fill_null(pa.nulls(n, type=pa.bool_()), True)


def error_mask(table: pa.Table) -> pa.Array:
    """True where `error` is a non-empty string.

    Shared by both stages: stage 1 uses it to spot failed-file placeholder rows,
    stage 2 to spot per-chunk processing failures. Empty-string is treated as
    "no error" to match tmp/stat_stage2.py:35.
    """
    if "error" not in table.schema.names:
        return _all_false(table.num_rows)
    col = table["error"]
    if pa.types.is_null(col.type):
        return _all_false(table.num_rows)
    return pc.fill_null(
        pc.and_kleene(pc.is_valid(col), pc.not_equal(col, "")), False
    )


# Backwards-compatible alias used where the stage-1 intent should be explicit.
_has_error = error_mask


def stage1_valid_mask(table: pa.Table) -> pa.Array:
    """Real stage-1 segments: `error IS NULL`.

    See pipeline_v2_ray/segments.py:44-53 -- failed files contribute exactly
    one row whose only populated fields are source/shard/version/error.
    """
    return pc.invert(_has_error(table))


def stage1_error_mask(table: pa.Table) -> pa.Array:
    """Stage-1 failed-file placeholder rows."""
    return _has_error(table)


def stage2_kept_mask(table: pa.Table) -> pa.Array:
    """Segments that survived stage 2 -- the pipeline's final output.

    Transcribed from tmp/stat_stage2.py:35-44 so the headline "keep rate" in a
    QC report is the same number the ad-hoc script produces.
    """
    keep = pc.invert(_has_error(table))
    for name in DROP_COLUMNS:
        if name not in table.schema.names:
            continue
        col = table[name]
        if pa.types.is_null(col.type):
            continue
        keep = pc.and_kleene(keep, pc.invert(pc.fill_null(col, False)))
    return pc.fill_null(keep, False)


def stage2_drop_mask(table: pa.Table, column: str) -> pa.Array:
    """Rows flagged by one specific `dropped_by_*` reason (not exclusive)."""
    if column not in table.schema.names:
        return _all_false(table.num_rows)
    col = table[column]
    if pa.types.is_null(col.type):
        return _all_false(table.num_rows)
    return pc.fill_null(col, False)


def stage2_retriable_error_mask(table: pa.Table) -> pa.Array:
    """Stage-2 rows whose failure was "couldn't reach the remote ASR".

    These are transient and get reprocessed on the next run
    (pipeline_v2_ray/stage2_segments.py:89-98), so counting them as permanent
    losses would understate the achievable yield.
    """
    if "error" not in table.schema.names:
        return _all_false(table.num_rows)
    col = table["error"]
    if pa.types.is_null(col.type):
        return _all_false(table.num_rows)
    return pc.fill_null(
        pc.and_kleene(pc.is_valid(col), pc.match_substring(col, ASR_RETRIABLE_MARKER)),
        False,
    )


def masked_sum(table: pa.Table, column: str, mask: pa.Array) -> float:
    """Sum `column` where `mask`, treating NULL as 0."""
    if column not in table.schema.names:
        return 0.0
    col = table[column]
    if pa.types.is_null(col.type):
        return 0.0
    return float(pc.sum(pc.if_else(mask, pc.fill_null(col, 0.0), 0.0)).as_py() or 0.0)


def mask_count(mask: pa.Array) -> int:
    return int(pc.sum(pc.cast(mask, "int64")).as_py() or 0)


def classify_error(error: Optional[str]) -> str:
    """Bucket a raw error string into a short category.

    Same heuristic as tmp/stat_parquet.py:23 (prefix before the first colon,
    else a 40-char prefix): pipeline errors are formatted `stage: detail`, so
    the prefix is the stage while the detail is high-cardinality noise.
    """
    if not error:
        return "<none>"
    if ASR_RETRIABLE_MARKER in error:
        return ASR_RETRIABLE_MARKER
    if ":" in error:
        return error.split(":", 1)[0].strip()[:60]
    return error[:40]


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def read_chunk_audio(path: str) -> tuple[np.ndarray, int]:
    """Read a whole chunk wav as mono float32 at its native sample rate.

    Deliberately identical to how stage 2 loads the same files
    (pipeline_v2_ray/actors/v2_stage_2.py:126): `dtype="float32"`,
    `always_2d=False`, and NO resampling or re-normalisation. The wav was
    already decoded to mono at the target rate and RMS-normalised toward
    -20 dBFS by stage 1 (pipeline_v2/steps/standardization.py:221-272), so any
    extra processing here would measure QC's own preprocessing rather than the
    pipeline's output.
    """
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        # Defensive: production writes mono, but a stray multi-channel file
        # must not reach the models with the wrong shape.
        waveform = waveform.mean(axis=1)
    return np.ascontiguousarray(waveform), int(sample_rate)


def slice_segment(
    waveform: np.ndarray, sample_rate: int, start: float, end: float
) -> np.ndarray:
    """Cut `[start, end)` seconds out of a chunk waveform, clamped to bounds.

    Segment boundaries in the parquet already include the grace period
    (pipeline_v2/steps/segment.py:222-235), so `end` can sit a hair past the
    last sample; clamping keeps that from producing an empty slice.
    """
    n = len(waveform)
    s = max(0, int(round(start * sample_rate)))
    e = min(n, int(round(end * sample_rate)))
    if e <= s:
        return np.empty(0, dtype=np.float32)
    return waveform[s:e]


def audio_duration_seconds(path: str) -> Optional[float]:
    """Duration from the wav header, without decoding samples."""
    import soundfile as sf

    try:
        with sf.SoundFile(path) as f:
            return len(f) / float(f.samplerate)
    except Exception:  # noqa: BLE001 - corrupt/unsupported file
        return None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
