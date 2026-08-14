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
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")    #A100上跑要开启， V100上关闭
os.environ.setdefault("LD_PRELOAD", "/lib64/libcuda.so.1")      #A100上跑要开启， V100上关闭
os.environ.setdefault("NCCL_P2P_DISABLE", "1")       #A100上跑要开启， V100上关闭
os.environ.setdefault("NCCL_SHM_DISABLE", "1")        #A100上跑要开启， V100上关闭

import sys
import warnings
from pathlib import Path

import ray

import logger
import pipeline_v2_ray.actors  # noqa: F401 -- importing the package registers all actors
from pipeline_v2_ray.actors.base import ACTOR_REGISTRY
from pipeline_v2_ray.config import load_ray_config, load_stage2_ray_config
from pipeline_v2_ray.driver import ClusterDriver, FileItem
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-config", default="configs/pipeline_v2_ray.yaml", help="ray hardware map")
    # Input is either an ad-hoc file/folder (--input), a prebuilt manifest
    # (--manifest, whose relative_paths are resolved under --audio-root), or a
    # stage-1 output folder (--stage1-output, for stage-2: remote ASR + v1
    # post-processing on top of stage-1's already-exported chunk wavs).
    p.add_argument("--input", help="audio file or folder")
    p.add_argument("--manifest", help="manifest parquet (file or shard dir) from build_manifest.py")
    p.add_argument("--audio-root", help="audio root to resolve manifest relative_paths against")
    p.add_argument("--stage1-output",
                   help="stage-1 output folder to read segments from and write stage-2 "
                        "results into (ASR + v1 post-processing mode, --actor v2_stage_2)")
    p.add_argument("--actor", default="v2_stage_1", choices=sorted(ACTOR_REGISTRY),
                   help="which processing actor to run")
    p.add_argument("--output", help="output folder for exported jsons; defaults to "
                                "--stage1-output in stage-2 mode")
    p.add_argument("--address", default="auto", help="ray cluster address")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="skip manifest files shorter than this many seconds "
                        "(uses the manifest's duration column; 0 = no filter). "
                        "Note: duration==0 means unknown/probe-failed, so it is "
                        "also skipped when this is > 0. Only applies to --manifest.")
    args = p.parse_args()
    modes = [bool(args.input), bool(args.manifest), bool(args.stage1_output)]
    if sum(modes) != 1:
        p.error("provide exactly one of --input, --manifest, or --stage1-output")
    if args.manifest and not args.audio_root:
        p.error("--manifest requires --audio-root to resolve relative paths")
    if args.stage1_output:
        args.output = args.output or args.stage1_output
    elif not args.output:
        p.error("--output is required for --input/--manifest modes")
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


def collect_stage1_segments(stage1_output: str) -> list[tuple[str, list[FileItem]]]:
    """Stage-2 input mode: read stage-1's segments_part-*.parquet from every
    shard subdir of `stage1_output`, and group rows by `chunk_audio_path` --
    one FileItem per stage-1 chunk wav, with `payload` set to that chunk's
    stage-1 segments (start/end/speaker_id/utt_id/...).

    `FileItem.relative_path` is the `chunk_audio_path` itself (output-root
    relative, e.g. "<shard>/audios/<bucket>/<file>.wav"), since that -- not
    any file-level path -- is the resume/dedup granularity stage 2 works at
    (matches STAGE2_SEGMENT_SCHEMA.source). Rows belonging to a failed
    stage-1 file (error set, no chunk_audio_path) are skipped: there's
    nothing for stage 2 to re-process.
    """
    import pyarrow.parquet as pq

    from source_scan.manifest import list_shards

    groups: list[tuple[str, list[FileItem]]] = []
    shard_names = sorted(
        d for d in os.listdir(stage1_output)
        if os.path.isdir(os.path.join(stage1_output, d))
    )
    cols = ["utt_id", "source", "chunk_index", "chunk_audio_path",
            "speaker_id", "start", "end", "error"]
    for shard_name in shard_names:
        shard_dir = os.path.join(stage1_output, shard_name)
        parts = list_shards(shard_dir, "segments")
        if not parts:
            continue
        chunks: dict[str, list[dict]] = {}
        for part in parts:
            for row in pq.read_table(part, columns=cols).to_pylist():
                if row.get("error") is not None or not row.get("chunk_audio_path"):
                    continue  # failed-file placeholder row; nothing to re-process
                chunks.setdefault(row["chunk_audio_path"], []).append(row)
        items = []
        for chunk_audio_path, rows in chunks.items():
            rows.sort(key=lambda r: r.get("start") or 0.0)
            payload = [
                {
                    "utt_id": r["utt_id"],
                    "origin_source": r["source"],
                    "chunk_index": r["chunk_index"],
                    "speaker_id": r["speaker_id"],
                    "start": r["start"],
                    "end": r["end"],
                }
                for r in rows
            ]
            items.append(FileItem(
                audio_path=os.path.join(stage1_output, chunk_audio_path),
                relative_path=chunk_audio_path,
                duration=sum((r.get("end") or 0.0) - (r.get("start") or 0.0) for r in rows),
                payload=payload,
            ))
        if items:
            groups.append((shard_name, items))
    return groups


def main() -> None:
    args = parse_args()

    if args.stage1_output:
        groups = collect_stage1_segments(args.stage1_output)
    elif args.manifest:
        groups = collect_manifest_paths(args.manifest, args.audio_root, args.min_duration)
    else:
        if args.min_duration > 0:
            logger.warning("--min-duration ignored in --input mode (no manifest duration)")
        groups = collect_audio_paths(args.input)
    if not groups:
        logger.warning("no audio files to process")
        sys.exit(0)

    # Propagate GPU-related env vars to every ray worker process (they were only
    # set in this driver process above; workers inherit the env `ray start` was
    # launched with, not this script's, so they must be injected explicitly).
    worker_env_vars = {
        k: os.environ[k]
        for k in (
            #"PYTORCH_CUDA_ALLOC_CONF",
            "LD_PRELOAD",
            "NCCL_P2P_DISABLE",
            "NCCL_SHM_DISABLE",
        )
        if k in os.environ
    }
    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": worker_env_vars},
    )
    if args.stage1_output:
        # Stage 2 flushes a distinct stage2_segments_part-*.parquet into the
        # same shard dirs stage 1 already wrote to, with its own resume key
        # (chunk_audio_path) -- see pipeline_v2_ray.stage2_segments.
        from pipeline_v2_ray.stage2_segments import (error_record2,
                                                      resume_state2,
                                                      write_stage2_segments_shard)
        driver = ClusterDriver(
            load_stage2_ray_config(args.ray_config), args.actor,
            segment_writer=write_stage2_segments_shard,
            segment_resumer=resume_state2,
            error_record_fn=error_record2,
        )
    else:
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
