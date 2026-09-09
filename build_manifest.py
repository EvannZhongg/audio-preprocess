"""Stage 2: probe each path from the stage-1 paths shards and write the final
manifest (relative_path, duration, file_size, format), one manifest shard per
paths shard.

    python build_manifest.py --paths paths_dir/ --audio-root /data/xxx \
                             --out manifest_dir/ [--workers 32]

Each paths_part-NNNNN.parquet (~100k rows) is read whole, probed concurrently,
and written to manifest_part-NNNNN.parquet with the same index. On rerun,
manifest shards that already exist are skipped -- a crash only loses the shard
in flight. Point --manifest (in main_v2_ray.py) at --out to consume all shards.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from source_scan.manifest import (list_path_shards, num_rows, probe, read_paths,
                                   shard_name, write_manifest_shard)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", required=True, help="stage-1 paths shard directory")
    p.add_argument("--audio-root", required=True, help="audio root the paths are relative to")
    p.add_argument("--out", required=True, help="output directory for manifest shards")
    p.add_argument("--workers", type=int, default=32, help="concurrent probe threads")
    return p.parse_args()


def _shard_index(paths_shard: str) -> int:
    # paths_part-00003.parquet -> 3
    return int(os.path.basename(paths_shard).split("-")[-1].split(".")[0])


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.audio_root):
        sys.exit(f"--audio-root not a directory: {args.audio_root} (check for typos; "
                 f"every file would otherwise probe as size=0)")
    os.makedirs(args.out, exist_ok=True)

    shards = list_path_shards(args.paths)
    if not shards:
        print(f"no paths shards found in {args.paths}", file=sys.stderr)
        sys.exit(1)
    # Footer-only row counts give a file-level progress total up front (no data read).
    total = sum(num_rows(s) for s in shards)
    print(f"build_manifest: {len(shards)} paths shards, {total} files -> {args.out}",
          file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            tqdm(total=total, desc="probe", unit="file") as bar:
        for paths_shard in shards:
            out_path = os.path.join(args.out, shard_name("manifest", _shard_index(paths_shard)))
            if os.path.exists(out_path):        # resume: skip finished shard
                bar.update(num_rows(paths_shard))
                continue
            records = []
            # pool.map yields results lazily in submission order, so we can tick
            # the bar per file rather than once per whole shard.
            for rec in pool.map(partial(probe, root=args.audio_root), read_paths(paths_shard)):
                records.append(rec)
                bar.update(1)
            # temp-then-rename so a half-written shard is never mistaken for done.
            tmp = out_path + ".tmp"
            write_manifest_shard(records, tmp)
            os.replace(tmp, out_path)

    print(f"build_manifest done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
