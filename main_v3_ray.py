"""Ray cluster entrypoint for pipeline_v3: multi-stage streaming pipeline.

Runs one or more configured stages (stage_1, stage_2, ... see
pipeline_v3/stages.py for the pluggable per-stage registry) in a single
process. Every selected stage gets its own actor pool, gated by its own Ray
custom resource (slot_stage_1, slot_stage_2, ... configurable in
configs/pipeline_v3.yaml); a file finishing stage N is queued straight into
stage N+1 (if it's also selected this run) instead of waiting for the whole
shard to finish stage N first.

Manifest shards are pipelined too: the first stage opens the next shard as soon
as its own queue runs low, so it never idles waiting for a slower downstream
stage, and once its input is exhausted its whole actor pool is released (GPUs
and slots handed back) while the downstream stages keep draining.

Usage:
    # both stages, streamed together:
    python main_v3_ray.py --ray-config configs/pipeline_v3.yaml \
                           --manifest <manifest> --audio-root <root> \
                           --output <output_folder> --stages stage_1,stage_2

    # only stage_2, reading stage_1's already-flushed parquet from --output:
    python main_v3_ray.py --ray-config configs/pipeline_v3.yaml \
                           --output <output_folder> --stages stage_2
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
os.environ.setdefault("LD_PRELOAD", "/lib64/libcuda.so.1")      # A100上跑要开启， V100上关闭
os.environ.setdefault("NCCL_P2P_DISABLE", "1")                  # A100上跑要开启， V100上关闭
os.environ.setdefault("NCCL_SHM_DISABLE", "1")                   # A100上跑要开启， V100上关闭

import sys
import warnings
from pathlib import Path

import ray

import logger
import pipeline_v2_ray.actors  # noqa: F401 -- importing the package registers v2_stage_1/v2_stage_2
from pipeline_v3.config import load_pipeline_v3_config
from pipeline_v3.driver import MultiStagePipelineRunner, StageTotals
from pipeline_v3.stages import STAGE_REGISTRY
from pipeline_v3.types import FileItem
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-config", default="configs/pipeline_v3.yaml",
                    help="pipeline_v3 stage map (stage order, resources, hardware routing)")
    p.add_argument("--input", help="audio file or folder (only used when the FIRST selected "
                                    "stage is also the pipeline's first stage overall)")
    p.add_argument("--manifest", help="manifest parquet (file or shard dir) from build_manifest.py")
    p.add_argument("--audio-root", help="audio root to resolve manifest relative_paths against")
    p.add_argument("--stages", default=None,
                   help="comma-separated stages to run this invocation, e.g. 'stage_1,stage_2' "
                        "or 'stage_2' (order doesn't matter, always run in pipeline order; "
                        "default: every stage declared in --ray-config)")
    p.add_argument("--output", required=True, help="shared output folder for every selected stage "
                                                      "(also where a downstream-only stage reads its "
                                                      "upstream stage's already-flushed parquet from)")
    p.add_argument("--address", default="auto", help="ray cluster address")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="skip manifest files shorter than this many seconds "
                        "(0 = no filter). Only applies to --manifest, and only "
                        "when the first selected stage is the pipeline's first stage.")
    args = p.parse_args()
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
    [(shard_name, [FileItem, ...]), ...] in shard order -- one group per
    manifest_part-*.parquet, so shards are processed one at a time."""
    from source_scan.manifest import list_shards, read_manifest

    shards = list_shards(manifest, "manifest") if os.path.isdir(manifest) else [manifest]
    groups: list[tuple[str, list[FileItem]]] = []
    kept = skipped = 0
    for shard in shards:
        shard_name = os.path.splitext(os.path.basename(shard))[0]
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


def _upstream_of(stage_order: list[str]) -> dict[str, str]:
    """downstream_key -> upstream_key, derived from STAGE_REGISTRY's
    `next_stage` links -- used to find which stage's disk output seeds a
    given stage when it isn't fed live this run."""
    upstream: dict[str, str] = {}
    for key, sdef in STAGE_REGISTRY.items():
        if sdef.next_stage:
            upstream[sdef.next_stage] = key
    return upstream


def main() -> None:
    args = parse_args()

    all_cfgs = load_pipeline_v3_config(
        args.ray_config, {k: v.params_cls for k, v in STAGE_REGISTRY.items()}
    )
    pipeline_order = [c.key for c in all_cfgs]  # full pipeline order, as declared in --ray-config

    if args.stages:
        requested = {s.strip() for s in args.stages.split(",") if s.strip()}
        unknown = requested - set(pipeline_order)
        if unknown:
            raise SystemExit(f"unknown stage(s) {sorted(unknown)}; choices: {pipeline_order}")
        selected = [s for s in pipeline_order if s in requested]
    else:
        selected = pipeline_order

    stage_cfgs = [c for c in all_cfgs if c.key in selected]
    first_stage = selected[0]
    upstream_of = _upstream_of(pipeline_order)

    if first_stage == pipeline_order[0]:
        # The first selected stage is the pipeline's very first stage overall
        # -> fed from raw --input/--manifest.
        if bool(args.input) == bool(args.manifest):
            raise SystemExit("provide exactly one of --input or --manifest")
        if args.manifest:
            groups = collect_manifest_paths(args.manifest, args.audio_root, args.min_duration)
        else:
            if args.min_duration > 0:
                logger.warning("--min-duration ignored in --input mode (no manifest duration)")
            groups = collect_audio_paths(args.input)
    else:
        # The first selected stage is a downstream stage (e.g. running only
        # stage_2) -> its input comes entirely from its upstream stage's
        # already-flushed parquet under --output (matches main_v2_ray's
        # --stage1-output mode).
        if args.input or args.manifest:
            logger.warning(
                f"--input/--manifest ignored: first selected stage '{first_stage}' "
                "reads its upstream stage's disk output under --output instead"
            )
        prev_key = upstream_of.get(first_stage)
        if prev_key is None:
            raise SystemExit(f"stage '{first_stage}' has no upstream stage to read input from")
        groups = STAGE_REGISTRY[prev_key].load_output_from_disk(args.output)

    if not groups:
        logger.warning("no work to process")
        sys.exit(0)

    # Propagate GPU-related env vars to every ray worker process.
    worker_env_vars = {
        k: os.environ[k]
        for k in ("LD_PRELOAD", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
        if k in os.environ
    }
    ray.init(address=args.address, ignore_reinit_error=True,
             runtime_env={"env_vars": worker_env_vars})

    def seed_from_disk(shard_name: str) -> dict[str, list[FileItem]]:
        """Extra seed for the selected NON-first stages of one shard, read from
        their own upstream stage's already-flushed parquet.

        Covers resuming a multi-stage run where an earlier stage got ahead (and
        flushed to disk) before a crash, ahead of this run's live streaming
        hand-off (in MultiStagePipelineRunner.run).
        """
        seed_items: dict[str, list[FileItem]] = {}
        for key in selected[1:]:
            prev_key = upstream_of.get(key)
            if prev_key is None:
                continue
            disk_groups = STAGE_REGISTRY[prev_key].load_output_from_disk(
                args.output, shard_names=[shard_name]
            )
            for g_shard, g_items in disk_groups:
                if g_shard == shard_name and g_items:
                    seed_items[key] = g_items
        return seed_items

    runner = MultiStagePipelineRunner(stage_cfgs)
    totals: dict[str, StageTotals] = {}
    try:
        runner.start()
        # Shards are pipelined: the first stage opens the next manifest shard as
        # soon as its own queue runs low, without waiting for the downstream
        # stages to drain the previous one, and its actor pool is released as
        # soon as its input is exhausted.
        totals = runner.run(groups, args.output, seed_from_disk)
    finally:
        runner.shutdown()
        ray.shutdown()

    for key, tot in totals.items():
        logger.info(
            f"ray_v3_driver_done stage {key} files {tot.files} success {tot.ok} "
            f"failed {tot.failed}"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
