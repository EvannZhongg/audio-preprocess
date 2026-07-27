"""Local PipelineV2 entrypoint for a single-GPU machine (multiprocessing).

The processing implementation and exported ``pipeline_version`` remain V2.
``--num-workers`` controls the number of worker processes; each process owns
its own PipelineV2 instance.  A cross-process GPU lock serializes the GPU
stages so the two processes never hit the embedding-refinement memory peak at
the same time, while CPU decode/standardize and export still overlap across
processes.

Usage:
    python main_v3.py --config <config.json> \
                      --input <audio_or_folder> \
                      --output <output_folder> \
                      [--num-workers N]

    python main_v3.py --config <config.json> \
                      --manifest <manifest.parquet_or_shard_dir> \
                      --audio-root <audio_root> \
                      --output <output_folder> \
                      [--min-duration SECONDS] \
                      [--num-workers N]
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

# 必须在 import torch 之前设置 allocator 配置。早期在单卡上启动多个 Pipeline
# 时容易先触发显存申请失败,再进入依赖 NVML 的 OOM 诊断路径;关闭
# expandable_segments 可避开已观察到的不稳定分配路径。这里会保留用户通过
# 环境变量提供的其他 allocator 参数。
_cuda_alloc_options = [
    option
    for option in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
    if option and not option.strip().startswith("expandable_segments:")
]
_cuda_alloc_options.append("expandable_segments:False")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_cuda_alloc_options)

import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
import tqdm

import logger
from logger import make_extra_tags
from pipeline_v2.exceptions import PipelineError
from pipeline_v2.params import PipelineParams
from pipeline_v2.pipeline import PipelineV2
from pipeline_v2.state import PIPELINE_VERSION, PipelineState
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class FileItem:
    """One source file plus the stable path used by PipelineV2 for export IDs."""

    audio_path: str
    relative_path: str
    duration: float = 0.0


@dataclass(frozen=True)
class ProcessTask:
    audio_path: str
    output_folder: str
    relative_path: str
    shard: str | None


# ---------------------------------------------------------------------------
# 单 GPU 多进程调度状态
#
# 1. 每个 worker 进程各自创建一份 _WORKER_PIPE(进程隔离,天然避免线程版
#    的跨线程原生对象 SIGSEGV);单 GPU 上因此有 num_workers 份模型权重常驻;
# 2. _GPU_LOCK 是跨进程锁,只在 GPU 阶段持有,保证任一时刻只有一个进程在
#    冲 embedding refinement 的显存峰值,峰值约为 1x 而非 num_workers x;
# 3. standardize(CPU decode/normalize)和 export(WAV/JSON 写盘)不加锁,
#    可与另一个进程的 GPU 阶段重叠。
# ---------------------------------------------------------------------------
_WORKER_PIPE: PipelineV2 | None = None
_GPU_LOCK = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the V2 pipeline locally with file/folder or Parquet manifest input."
    )
    p.add_argument("--config", required=True)

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="audio file or folder")
    input_group.add_argument(
        "--manifest",
        help="manifest parquet file or shard directory produced by build_manifest.py",
    )

    p.add_argument(
        "--audio-root",
        help="audio root used to resolve manifest relative_path values",
    )
    p.add_argument("--output", required=True, help="output folder for exported WAV/JSON files")
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="number of worker processes; each process owns its own PipelineV2",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="skip manifest files shorter than this many seconds; 0 disables filtering",
    )

    args = p.parse_args()
    if args.manifest and not args.audio_root:
        p.error("--manifest requires --audio-root")
    if args.num_workers < 1:
        p.error("--num-workers must be at least 1")
    if args.min_duration < 0:
        p.error("--min-duration must be non-negative")
    return args


def collect_audio_groups(input_path: str) -> list[tuple[str, list[FileItem]]]:
    """Collect an ad-hoc file/folder as one non-manifest input group."""

    p = Path(input_path)
    if p.is_file():
        return [("input", [FileItem(audio_path=str(p), relative_path=p.name)])]
    if p.is_dir():
        items = [
            FileItem(
                audio_path=audio_path,
                relative_path=os.path.relpath(audio_path, input_path),
            )
            for audio_path in get_audio_files(str(p))
        ]
        return [("input", items)]
    print(f"input not found: {input_path}", file=sys.stderr)
    sys.exit(1)


def collect_manifest_groups(
    manifest: str,
    audio_root: str,
    min_duration: float = 0.0,
) -> list[tuple[str, list[FileItem]]]:
    """Read manifest shards and resolve their relative paths under audio_root."""

    from source_scan.manifest import list_shards, read_manifest

    if not os.path.exists(manifest):
        print(f"manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(audio_root):
        print(f"audio root not found or not a directory: {audio_root}", file=sys.stderr)
        sys.exit(1)

    shards = list_shards(manifest, "manifest") if os.path.isdir(manifest) else [manifest]
    groups: list[tuple[str, list[FileItem]]] = []
    kept = 0
    skipped = 0

    for shard_path in shards:
        shard_name = os.path.splitext(os.path.basename(shard_path))[0]
        table = read_manifest(shard_path, columns=["relative_path", "duration"])
        relative_paths = table["relative_path"].to_pylist()
        durations = table["duration"].to_pylist()

        items: list[FileItem] = []
        for relative_path, duration in zip(relative_paths, durations):
            if min_duration > 0 and (duration is None or duration < min_duration):
                skipped += 1
                continue
            items.append(
                FileItem(
                    audio_path=os.path.join(audio_root, relative_path),
                    relative_path=relative_path,
                    duration=duration or 0.0,
                )
            )
            kept += 1

        if items:
            groups.append((shard_name, items))

    if min_duration > 0:
        logger.info(
            f"main_v3_min_duration_filter min {min_duration}s "
            f"kept {kept} skipped {skipped}"
        )
    return groups


def _init_worker(config_path: str, gpu_lock) -> None:
    """Build this process's own PipelineV2 and record the shared GPU lock."""

    global _WORKER_PIPE, _GPU_LOCK
    params = PipelineParams.from_config(config_path)
    _WORKER_PIPE = PipelineV2(params)
    _GPU_LOCK = gpu_lock


def _process_one(task: ProcessTask) -> tuple[str, int, str]:
    """Standardize/export in this process; run GPU stages under the shared lock."""

    assert _WORKER_PIPE is not None
    bootstrap = PipelineState(
        audio_path=task.audio_path,
        relative_path=task.relative_path,
        shard=task.shard,
        log_tag=make_extra_tags(
            audio_file=task.relative_path,
            version=PIPELINE_VERSION,
        ),
    )

    try:
        # Decode/normalize is CPU-bound and holds no GPU lock, so it can overlap
        # the other process's GPU inference.
        chunk_states = _WORKER_PIPE.standardize(bootstrap)
        segment_count = 0

        for chunk_index, state in enumerate(chunk_states):
            t0 = time.perf_counter()
            is_last_chunk = chunk_index == len(chunk_states) - 1

            # Serialize the GPU stages across processes: only one process runs
            # separate/diarize/vad/refine/segment/metrics at a time, so the
            # embedding-refinement memory peak stays ~1x rather than num_workers x.
            with _GPU_LOCK:
                state, vad_dur, refine_dur = _WORKER_PIPE.run_gpu_stages(state)
                if torch.cuda.is_available():
                    # Kernels run async; sync before releasing the lock so this
                    # file's GPU work is truly done and its peak memory can be
                    # freed before the other process acquires the lock.
                    torch.cuda.synchronize()
                    if is_last_chunk:
                        # Per-file cleanup mirrors PipelineV2.run's finally; done
                        # inside the lock so the freed memory is available to the
                        # next process before it starts.
                        torch.cuda.empty_cache()

            # Export is CPU/IO-bound; run it outside the lock so it overlaps the
            # next process's GPU work.
            state = _WORKER_PIPE.export(state, chunk_index, task.output_folder)
            PipelineV2.log_chunk_stats(state, t0, vad_dur, refine_dur)
            segment_count += len(state.segment_list or [])

        return (task.relative_path, segment_count, "")
    except Exception as exc:
        detail = (
            f"stage {exc.stage} msg {exc.message}"
            if isinstance(exc, PipelineError)
            else f"unexpected {type(exc).__name__} {exc}"
        )
        logger.error(f"pipeline_failed {detail}", extra=bootstrap.log_tag)
        return (task.relative_path, 0, f"{type(exc).__name__}: {exc}")


def _log_result(result: tuple[str, int, str]) -> None:
    relative_path, segment_count, error = result
    if error:
        logger.error(f"main_v3_failed file {relative_path} err {error}")
    else:
        logger.info(
            f"main_v3_done file {relative_path} segments {segment_count}"
        )


def _tasks_for_group(
    shard_name: str,
    items: list[FileItem],
    output_root: str,
    manifest_mode: bool,
) -> list[ProcessTask]:
    output_folder = (
        os.path.join(output_root, shard_name) if manifest_mode else output_root
    )
    shard = shard_name if manifest_mode else None
    return [
        ProcessTask(
            audio_path=item.audio_path,
            output_folder=output_folder,
            relative_path=item.relative_path,
            shard=shard,
        )
        for item in items
    ]


def main() -> None:
    args = parse_args()
    manifest_mode = bool(args.manifest)

    if manifest_mode:
        groups = collect_manifest_groups(
            args.manifest,
            args.audio_root,
            args.min_duration,
        )
    else:
        if args.min_duration > 0:
            logger.warning(
                "--min-duration ignored in --input mode (no manifest duration)"
            )
        groups = collect_audio_groups(args.input)

    total = sum(len(items) for _, items in groups)
    if total == 0:
        logger.warning("no audio files to process")
        sys.exit(0)

    os.makedirs(args.output, exist_ok=True)
    num_workers = max(1, min(args.num_workers, total))

    if num_workers > 1:
        # 单卡多进程:每个进程各自加载一套完整 PipelineV2,显存占用随进程数
        # 翻倍。GPU 锁只能防止两进程同时冲峰值,防不了 num_workers 份常驻权重
        # 本身超预算。若 OOM,继续下调 refinement_batch_size 或退回 1 个进程。
        logger.warning(
            f"single-GPU multiprocessing: {num_workers} processes each load a "
            f"full PipelineV2; GPU memory scales with workers. GPU stages are "
            f"serialized by a cross-process lock, but if OOM occurs lower "
            f"refinement_batch_size or use --num-workers 1"
        )

    logger.info(
        f"main_v3 files {total} workers {num_workers} gpu_lock serialized "
        f"output {args.output} mode {'manifest' if manifest_mode else 'input'} "
        f"pipeline_version v2 "
        f"allocator_conf {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
    )

    if num_workers == 1:
        # Single process: no cross-process contention, so a no-op lock keeps
        # _process_one's code path identical to the multi-process one.
        _init_worker(args.config, nullcontext())
        for shard_name, items in groups:
            tasks = _tasks_for_group(shard_name, items, args.output, manifest_mode)
            for task in tqdm.tqdm(tasks, desc=f"pipeline_v3:{shard_name}"):
                _log_result(_process_one(task))
        return

    # Manager().Lock() yields a picklable proxy that survives spawn+initargs;
    # a raw mp.Lock() is not reliably passable through Pool initargs under spawn.
    manager = mp.Manager()
    gpu_lock = manager.Lock()
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.config, gpu_lock),
    ) as pool:
        for shard_name, items in groups:
            tasks = _tasks_for_group(shard_name, items, args.output, manifest_mode)
            for result in tqdm.tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=1),
                total=len(tasks),
                desc=f"pipeline_v3:{shard_name}",
            ):
                _log_result(result)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
    sys.exit(0)
