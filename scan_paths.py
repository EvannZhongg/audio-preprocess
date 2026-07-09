"""Stage 1: enumerate a source dataset's audio files into sharded paths parquet.

    # whole tree in one pass
    python scan_paths.py --scanner dir --root /data/xxx --out paths_dir/
    # driven by stage-0 top entries (progress + resumable per top entry)
    python scan_paths.py --scanner top --root /data/xxx --top-dir top_dir/ --out paths_dir/

Records paths relative to --root plus the derived `format` column; it does not
open files -- duration probing is stage 2 (build_manifest.py). Output is sharded
(paths_part-NNNNN.parquet), one file per --shard-size rows, so stage 2 can
build/resume a manifest shard per paths shard.
"""
from __future__ import annotations

import argparse
import os
import sys

from source_scan.directory import DirectoryScanner
from source_scan.manifest import write_paths
from source_scan.top_dir_scanner import TopDirScanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scanner", default="dir", choices=["dir", "top"],
                   help="'dir': walk --root directly; 'top': drive from --top-dir shards")
    p.add_argument("--root", required=True, help="source dataset root directory")
    p.add_argument("--top-dir", help="stage-0 top-entry shard dir (required for --scanner top)")
    p.add_argument("--out", required=True, help="output directory for paths shards")
    p.add_argument("--shard-size", type=int, default=100_000, help="rows per paths shard")
    args = p.parse_args()
    if args.scanner == "top" and not args.top_dir:
        p.error("--scanner top requires --top-dir")
    return args


def _assert_out_outside_root(root: str, out: str) -> None:
    """The scan lazily consumes the walk while writing shards, so an --out
    inside --root would re-scan the parquet files it just wrote (self-pollution).
    Require them to be disjoint."""
    root_abs = os.path.abspath(root)
    out_abs = os.path.abspath(out)
    if os.path.commonpath([root_abs, out_abs]) == root_abs:
        sys.exit(f"--out ({out_abs}) must not be inside --root ({root_abs})")


def _make_scanner(args: argparse.Namespace):
    if args.scanner == "top":
        return TopDirScanner(args.root, args.top_dir)
    return DirectoryScanner(args.root)


def main() -> None:
    args = parse_args()
    _assert_out_outside_root(args.root, args.out)
    scanner = _make_scanner(args)
    n = write_paths(scanner.relative_paths(), args.out, shard_size=args.shard_size)
    print(f"scan_paths done: {n} files -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
