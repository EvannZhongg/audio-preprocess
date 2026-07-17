"""Parquet IO for the two manifest stages, plus the generic per-file probe.

Stage 1 (scan_paths.py) writes a paths file: `relative_path`, `format`.
Stage 2 (build_manifest.py) reads it, probes each file for duration/size, and
writes the final manifest: `relative_path`, `duration`, `file_size`, `format`.

`format` (the lowercase extension) is a cheap path-derived column computed in
stage 1, so the extension distribution is inspectable before the expensive
probe stage. Stage 2 carries it through rather than recomputing.

All parquet schema knowledge lives here so the two entrypoints stay thin.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

# ---- schemas -------------------------------------------------------------

TOP_SCHEMA = pa.schema([
    ("name", pa.string()),          # first-level entry name, relative to root (no type -- see stage 1)
])

PATHS_SCHEMA = pa.schema([
    ("relative_path", pa.string()),
    ("format", pa.string()),        # lowercase extension without the dot
])

MANIFEST_SCHEMA = pa.schema([
    ("relative_path", pa.string()),
    ("duration", pa.float64()),     # seconds; 0.0 when it could not be determined
    ("file_size", pa.int64()),      # bytes; 0 on stat failure
    ("format", pa.string()),        # lowercase extension without the dot
])


def _ext(relative_path: str) -> str:
    return os.path.splitext(relative_path)[1].lstrip(".").lower()


# ---- sharded writer / reader (shared by top + paths) ---------------------

def shard_name(prefix: str, index: int) -> str:
    """Canonical shard filename, e.g. shard_name('paths', 3) -> 'paths_part-00003.parquet'."""
    return f"{prefix}_part-{index:05d}.parquet"


def list_shards(shard_dir: str, prefix: str) -> list[str]:
    """Sorted paths of all `<prefix>_part-*.parquet` shards in shard_dir."""
    tag = f"{prefix}_part-"
    names = [n for n in os.listdir(shard_dir)
             if n.startswith(tag) and n.endswith(".parquet")]
    return [os.path.join(shard_dir, n) for n in sorted(names)]


def num_rows(parquet_path: str) -> int:
    """Row count of a parquet file from its footer metadata (no data read)."""
    return pq.ParquetFile(parquet_path).metadata.num_rows


def _write_sharded(record_iter, out_dir: str, prefix: str, schema: pa.Schema,
                   shard_size: int) -> int:
    """Stream dict records into <prefix>_part-NNNNN.parquet shards of shard_size
    rows each. temp-then-rename so a crash never leaves a half-written shard a
    resume would treat as complete. Returns the total row count."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    shard_index = 0
    buf: list[dict] = []

    def flush() -> None:
        nonlocal written, shard_index
        if not buf:
            return
        out_path = os.path.join(out_dir, shard_name(prefix, shard_index))
        tmp = out_path + ".tmp"
        pq.write_table(pa.Table.from_pylist(buf, schema=schema), tmp)
        os.replace(tmp, out_path)
        written += len(buf)
        shard_index += 1
        buf.clear()

    for rec in record_iter:
        buf.append(rec)
        if len(buf) >= shard_size:
            flush()
    flush()
    return written


# ---- stage 0: top-level entries ------------------------------------------

def write_top(name_iter, out_dir: str, shard_size: int = 100_000) -> int:
    """Write first-level entry names into top_part-NNNNN.parquet shards."""
    records = ({"name": name} for name in name_iter)
    return _write_sharded(records, out_dir, "top", TOP_SCHEMA, shard_size)


def read_top(top_parquet: str) -> list[str]:
    """Load one stage-0 shard's first-level names."""
    return pq.read_table(top_parquet, columns=["name"])["name"].to_pylist()


# ---- stage 1: paths file -------------------------------------------------

def write_paths(path_iter, out_dir: str, shard_size: int = 100_000) -> int:
    """Enumerate relative paths into paths_part-NNNNN.parquet shards (with the
    derived `format` column). Sharding keeps memory bounded on huge datasets and
    lets a matching manifest shard be built (and resumed) per paths shard."""
    records = ({"relative_path": rel, "format": _ext(rel)} for rel in path_iter)
    return _write_sharded(records, out_dir, "paths", PATHS_SCHEMA, shard_size)


def list_path_shards(paths_dir: str) -> list[str]:
    """Sorted paths of all stage-1 shard files in `paths_dir`."""
    return list_shards(paths_dir, "paths")


def read_paths(paths_parquet: str) -> list[dict]:
    """Load one stage-1 shard's records as dicts ({relative_path, format})."""
    t = pq.read_table(paths_parquet, columns=["relative_path", "format"])
    return t.to_pylist()


# ---- stage 2: probe + manifest -------------------------------------------

def probe(record: dict, root: str) -> dict:
    """Given a stage-1 record ({relative_path, format}), resolve it under `root`
    and read its duration / size. Duration is best-effort via mutagen (header
    read); 0.0 on failure -- those records are kept for manual analysis rather
    than dropped. `format` is carried through from stage 1."""
    rel = record["relative_path"]
    abs_path = os.path.join(root, rel)
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        size = 0
    return {
        "relative_path": rel,
        "duration": _probe_duration(abs_path),
        "file_size": size,
        "format": record["format"],
    }


def _probe_duration(abs_path: str) -> float:
    try:
        from mutagen import File as MutagenFile

        mf = MutagenFile(abs_path)
        if mf is not None and mf.info is not None and mf.info.length:
            return float(mf.info.length)
    except Exception:
        pass
    return 0.0


def write_manifest_shard(records: list[dict], out_path: str) -> None:
    """Write one shard of probed records to a manifest parquet file."""
    pq.write_table(pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA), out_path)


def read_manifest(manifest: str, columns: list[str] | None = None):
    """Read a manifest parquet (a single file or a directory of shards) into a
    pyarrow Table. `columns` restricts which are loaded."""
    return pq.read_table(manifest, columns=columns)

