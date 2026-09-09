"""Copy the audio files referenced by a manifest into a two-level tree.

Layout, one first-level dir per manifest shard, second-level dirs bucketed by
row order (<=bucket files each):

    out/
      manifest_part-00000/          # first level = parquet filename (no ext)
        0_9999/                     # second level = row-order bucket
          <file> ...
        10000_19999/
          ...
      manifest_part-00001/
        ...

Source files are resolved as os.path.join(--audio-root, relative_path), the
same convention build_manifest.py uses. Files are COPIED (source left intact).
Row order is taken from the manifest shard, counted from 0 within each shard,
so bucket boundaries are stable and reproducible across reruns.

Tuned for NFS (source) -> JuiceFS (destination), i.e. two network filesystems
where latency and per-file metadata ops dominate, not local disk bandwidth:
  * bucket dirs are pre-created once per shard, not per file (a per-file
    os.makedirs would be one JuiceFS metadata round-trip per file);
  * no per-file destination stat: we just copy, so the hot path has zero extra
    metadata round-trips (a rerun re-copies everything, wasting only bandwidth);
  * copyfile (not copy2) avoids extra chmod/utime metadata writes and uses the
    Linux sendfile fast path;
  * high, tunable concurrency to hide network latency.

    python move_dataset.py --manifest manifest_dir/ --audio-root /nfs/audio \
                           --out /juicefs/dataset [--bucket 10000] \
                           [--workers 64]

Copies are published via temp-then-rename so a reader never sees a half-written
file. A rerun is safe but not incremental -- it re-copies every file.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from tqdm import tqdm

from source_scan.manifest import list_shards


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="manifest shard directory (manifest_part-*.parquet)")
    p.add_argument("--audio-root", required=True,
                   help="audio root the manifest relative_paths are relative to")
    p.add_argument("--out", required=True, help="output dataset root")
    p.add_argument("--bucket", type=int, default=10_000,
                   help="max files per second-level directory (default 10000)")
    p.add_argument("--workers", type=int, default=64,
                   help="concurrent copy threads (default 64; raise for more "
                        "network-latency hiding)")
    p.add_argument("--dry-run", action="store_true",
                   help="print planned copies without touching the filesystem")
    return p.parse_args()


def _bucket_lo(index: int, bucket: int) -> int:
    return (index // bucket) * bucket


def _bucket_name(lo: int, bucket: int) -> str:
    """Second-level dir name, e.g. 0_9999, 10000_19999."""
    return f"{lo}_{lo + bucket - 1}"


def _copy_one(job: tuple[str, str], dry_run: bool) -> str | None:
    """Copy src->dst. Returns an error string, or None on success.

    No destination stat: on NFS->JuiceFS every metadata round-trip counts, so we
    just copy. A rerun re-copies everything (idempotent, only wasted bandwidth)."""
    src, dst = job
    if dry_run:
        return None
    try:
        tmp = dst + ".tmp"
        # copyfile uses os.sendfile on Linux (in-kernel, no user-space buffer)
        # and does NOT copy mode/mtime, saving metadata writes on JuiceFS.
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)                         # atomic publish, never a half file
        return None
    except FileNotFoundError:
        return f"missing source: {src}"
    except Exception as e:                           # keep going on per-file errors
        return f"{src} -> {dst}: {e}"


def _prepare_shard(shard: str, audio_root: str, out_root: str, bucket: int):
    """Read a shard's relative_paths, pre-create its bucket dirs once, and return
    (level1, jobs). Pre-creating here removes a per-file os.makedirs (a JuiceFS
    metadata round-trip per file) from the parallel copy path."""
    level1 = os.path.splitext(os.path.basename(shard))[0]
    rels = pq.read_table(shard, columns=["relative_path"])["relative_path"].to_pylist()

    n = len(rels)
    for b in range(math.ceil(n / bucket) if n else 0):
        lo = b * bucket
        os.makedirs(os.path.join(out_root, level1, _bucket_name(lo, bucket)),
                    exist_ok=True)

    jobs = []
    for i, rel in enumerate(rels):
        src = os.path.join(audio_root, rel)
        # flatten nested relative_path to one filename so files from different
        # sub-dirs never collide inside a bucket, while staying readable.
        flat = rel.replace(os.sep, "_")
        dst = os.path.join(out_root, level1,
                           _bucket_name(_bucket_lo(i, bucket), bucket), flat)
        jobs.append((src, dst))
    return level1, jobs


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.audio_root):
        sys.exit(f"--audio-root not a directory: {args.audio_root}")
    shards = list_shards(args.manifest, "manifest")
    if not shards:
        sys.exit(f"no manifest shards found in {args.manifest}")

    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    print(f"move_dataset: {len(shards)} shards, {total} files -> {args.out} "
          f"(bucket={args.bucket}, workers={args.workers}, "
          f"{'DRY-RUN' if args.dry_run else 'copy'})",
          file=sys.stderr)

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            tqdm(total=total, desc="copy", unit="file") as bar:
        for shard in shards:
            _, jobs = _prepare_shard(shard, args.audio_root, args.out, args.bucket)
            for err in pool.map(lambda j: _copy_one(j, args.dry_run), jobs):
                if err:
                    errors.append(err)
                bar.update(1)

    if errors:
        print(f"\n{len(errors)} errors (showing first 20):", file=sys.stderr)
        for e in errors[:20]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print(f"move_dataset done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
