"""Stage 0: enumerate only the first level under a root into top_part shards.

    python scan_top.py --root /data/xxx --out top_dir/ [--shard-size 100000]

Splits a huge tree into independent units: each first-level entry is recorded
as {name, is_dir}. Stage 1 (scan_paths.py --scanner top) then knows the total
up front (for progress) and recurses into the directories. Fast -- one scandir
of the root, no recursion.
"""
from __future__ import annotations

import argparse
import os
import sys

from source_scan.manifest import write_top
from source_scan.toplevel import top_entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="source dataset root directory")
    p.add_argument("--out", required=True, help="output directory for top-entry shards")
    p.add_argument("--shard-size", type=int, default=100_000, help="rows per shard")
    return p.parse_args()


def _assert_out_outside_root(root: str, out: str) -> None:
    root_abs = os.path.abspath(root)
    out_abs = os.path.abspath(out)
    if os.path.commonpath([root_abs, out_abs]) == root_abs:
        sys.exit(f"--out ({out_abs}) must not be inside --root ({root_abs})")


def main() -> None:
    args = parse_args()
    _assert_out_outside_root(args.root, args.out)
    n = write_top(top_entries(args.root), args.out, shard_size=args.shard_size)
    print(f"scan_top done: {n} top-level entries -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
