"""Ray cluster entrypoint for PipelineV2.

Usage:
    python main_v2_ray.py --ray-config configs/pipeline_v2_ray.yaml \
                          --input <audio_or_folder> \
                          --output <output_folder> \
                          [--address auto]

One persistent actor per GPU auto-detects its hardware (configs/pipeline_v2_ray.yaml) and
overlaps IO with GPU compute internally, so the GPU stays saturated. The
distributed cache (JuiceFS) is a POSIX mount -- input/output paths are used
directly.
"""
from __future__ import annotations

import argparse
import os

LARGE_TEMP_PATH = f"{os.getcwd()}/TEMP"
os.makedirs(LARGE_TEMP_PATH, exist_ok=True)
os.environ["LARGE_TEMP_DIR"] = LARGE_TEMP_PATH
os.environ["TMPDIR"] = LARGE_TEMP_PATH
os.environ["TEMP"] = LARGE_TEMP_PATH
os.environ["TMP"] = LARGE_TEMP_PATH
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import warnings
from pathlib import Path

import ray

import logger
import pipeline_v2_ray.actors  # noqa: F401 -- importing the package registers all actors
from pipeline_v2_ray.actors.base import ACTOR_REGISTRY
from pipeline_v2_ray.config import load_ray_config
from pipeline_v2_ray.driver import ClusterDriver, FileItem
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-config", default="configs/pipeline_v2_ray.yaml", help="ray hardware map")
    # Input is either an ad-hoc file/folder (--input) or a prebuilt manifest
    # (--manifest, whose relative_paths are resolved under --audio-root).
    p.add_argument("--input", help="audio file or folder")
    p.add_argument("--manifest", help="manifest parquet (file or shard dir) from build_manifest.py")
    p.add_argument("--audio-root", help="audio root to resolve manifest relative_paths against")
    p.add_argument("--actor", default="v2_stage_1", choices=sorted(ACTOR_REGISTRY),
                   help="which processing actor to run")
    p.add_argument("--output", required=True, help="output folder for exported jsons")
    p.add_argument("--address", default="auto", help="ray cluster address")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="skip manifest files shorter than this many seconds "
                        "(uses the manifest's duration column; 0 = no filter). "
                        "Note: duration==0 means unknown/probe-failed, so it is "
                        "also skipped when this is > 0. Only applies to --manifest.")
    args = p.parse_args()
    if bool(args.input) == bool(args.manifest):
        p.error("provide exactly one of --input or --manifest")
    if args.manifest and not args.audio_root:
        p.error("--manifest requires --audio-root to resolve relative paths")
    return args


def collect_audio_paths(input_path: str) -> list[tuple[str, list[FileItem]]]:
    """Ad-hoc file/folder mode. One "input" batch (no manifest shards);
    relative_path is relative to the input dir (or the basename for a single
    file) so the export id is stable."""
    p = Path(input_path)
    if p.is_file():
        return [("input", [FileItem(audio_path=str(p), relative_path=p.name)])]
    if p.is_dir():
        items = [
            FileItem(audio_path=fp, relative_path=os.path.relpath(fp, input_path))
            for fp in get_audio_files(str(p))
        ]
        return [("input", items)]
    print(f"input not found: {input_path}", file=sys.stderr)
    sys.exit(1)


def collect_manifest_paths(manifest: str, audio_root: str,
                           min_duration: float = 0.0) -> list[tuple[str, list[FileItem]]]:
    """Resolve a manifest against the audio root, grouped by shard: returns
    [(shard_name, [FileItem, ...]), ...] in shard order. One group per
    manifest_part-*.parquet so the driver processes shards one at a time. Kept
    out of module import time so source_scan/pyarrow only load in this mode.

    If min_duration > 0, files whose manifest duration is below it are skipped
    (duration==0 means unknown/probe-failed and is skipped too)."""
    from source_scan.manifest import list_shards, read_manifest

    shards = list_shards(manifest, "manifest") if os.path.isdir(manifest) else [manifest]
    groups: list[tuple[str, list[FileItem]]] = []
    kept = skipped = 0
    for shard in shards:
        shard_name = os.path.splitext(os.path.basename(shard))[0]  # e.g. manifest_part-00000
        t = read_manifest(shard, columns=["relative_path", "duration"])
        rels = t["relative_path"].to_pylist()
        durs = t["duration"].to_pylist()
        items = []
        for rel, dur in zip(rels, durs):
            if min_duration > 0 and (dur is None or dur < min_duration):
                skipped += 1
                continue
            items.append(FileItem(
                audio_path=os.path.join(audio_root, rel),
                relative_path=rel,
                duration=dur or 0.0,
            ))
            kept += 1
        if items:
            groups.append((shard_name, items))
    if min_duration > 0:
        logger.info(f"min_duration_filter min {min_duration}s kept {kept} skipped {skipped}")
    return groups


def main() -> None:
    args = parse_args()

    if args.manifest:
        groups = collect_manifest_paths(args.manifest, args.audio_root, args.min_duration)
    else:
        if args.min_duration > 0:
            logger.warning("--min-duration ignored in --input mode (no manifest duration)")
        groups = collect_audio_paths(args.input)
    if not groups:
        logger.warning("no audio files to process")
        sys.exit(0)

    ray.init(address=args.address, ignore_reinit_error=True)
    driver = ClusterDriver(load_ray_config(args.ray_config), args.actor)
    results = []
    try:
        driver.start()
        # Process manifest shards strictly in order: each shard is fully drained
        # (and its segments flushed) before the next begins.
        for shard_name, items in groups:
            results.extend(driver.run_batch(shard_name, items, args.output))
    finally:
        driver.shutdown()
        ray.shutdown()

    n_ok = sum(1 for r in results if r.success)
    logger.info(
        f"ray_driver_done files {len(results)} success {n_ok} "
        f"failed {len(results) - n_ok}"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
