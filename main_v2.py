"""Local PipelineV2 entrypoint for a single-GPU machine.

The processing implementation and exported ``pipeline_version`` remain V2.
``--num-workers`` controls concurrent file preparation/export threads.  One
dedicated GPU-owner thread initializes and runs the only PipelineV2 instance,
so CUDA/native model calls never migrate between file worker threads.

Usage:
    python main_v2.py --config <config.json> \
                      --input <audio_or_folder> \
                      --output <output_folder> \
                      [--num-workers N]

    python main_v2.py --config <config.json> \
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

# 必须在 import torch 之前设置 allocator 配置。早期在单卡上启动多个Pipeline 时容易先触发显存申请失败，再进入依赖 NVML 的 OOM 诊断路径；
# 关闭 expandable_segments 可避开已观察到的不稳定分配路径。这里会保留
# 用户通过环境变量提供的其他 allocator 参数。
_cuda_alloc_options = [
    option
    for option in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").split(",")
    if option and not option.strip().startswith("expandable_segments:")
]
_cuda_alloc_options.append("expandable_segments:False")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_cuda_alloc_options)

import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm

import logger
from logger import make_extra_tags
from pipeline_v2.exceptions import PipelineError
from pipeline_v2.params import PipelineParams
from pipeline_v2.pipeline import PipelineV2
from pipeline_v2.state import PIPELINE_VERSION, PipelineState
from pipeline_v2.steps.standardization import Standardizer
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
# 单 GPU 调度状态
#
# 1. _PIPELINE 只创建一份，并且只允许 GPU owner 线程初始化/调用；
# 2. 文件 worker 负责 CPU decode/standardize 和 export；
# 3. Standardizer 通过 thread-local 按文件线程隔离，避免 Silero 的递归状态
#    在不同线程之间迁移；
# 4. _GPU_EXECUTOR 只有一个 worker，从调度层保证同一时刻只有一个 GPU
#    Pipeline 在运行，并避免原生 CUDA/ONNX 对象跨线程调用导致 core dump。
# ---------------------------------------------------------------------------
_PIPELINE: PipelineV2 | None = None
_STANDARDIZATION_PARAMS = None
_PREP_THREAD_LOCAL = threading.local()
_GPU_EXECUTOR: ThreadPoolExecutor | None = None
_GPU_OWNER_THREAD_ID: int | None = None


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
        help=(
            "number of concurrent in-flight file threads; all threads share "
            "one GPU PipelineV2 instance"
        ),
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
            f"main_v2_min_duration_filter min {min_duration}s "
            f"kept {kept} skipped {skipped}"
        )
    return groups


def _init_gpu_pipeline(params: PipelineParams) -> None:
    """Initialize all CUDA/native models inside their permanent owner thread."""

    global _PIPELINE, _GPU_OWNER_THREAD_ID

    # 该函数作为单线程 GPU executor 的 initializer 执行。模型必须在后续
    # 执行推理的同一个 OS 线程中创建；仅用 Lock 防并发并不能防止模型在
    # 不同文件线程间迁移，之前曾因此出现无 Python traceback 的 SIGSEGV。
    _GPU_OWNER_THREAD_ID = threading.get_ident()
    _PIPELINE = PipelineV2(params)


def _gpu_runtime_info() -> tuple[str, int]:
    """Run on the GPU-owner thread and report its identity for diagnostics."""

    assert _PIPELINE is not None
    assert _GPU_OWNER_THREAD_ID == threading.get_ident()
    return _PIPELINE.params.device_name, _GPU_OWNER_THREAD_ID


def _get_standardizer() -> Standardizer:
    """Return the Standardizer owned by the current file thread."""
    assert _STANDARDIZATION_PARAMS is not None
    standardizer = getattr(_PREP_THREAD_LOCAL, "standardizer", None)
    if standardizer is None:
        # Silero TorchScript 模型内部保存 LSTM h/c 状态。每个文件 worker
        # 懒加载并长期复用自己的 CPU Standardizer，使模型的创建与调用固定
        # 在同一线程；num_workers=2 时会有两套较小的 CPU Silero 模型，但
        # 不会复制完整 GPU Pipeline。
        standardizer = Standardizer(_STANDARDIZATION_PARAMS, "cpu")
        _PREP_THREAD_LOCAL.standardizer = standardizer
    return standardizer


def _run_gpu_stages(
    state: PipelineState,
    cleanup_after: bool,
) -> tuple[PipelineState, float, float]:
    """Run only on the dedicated GPU-owner thread."""

    assert _PIPELINE is not None
    # 这是线程亲和性的运行时防线。如果后续重构误把 GPU 调用放回文件
    # worker，会在 Python 层尽早失败，而不是再次演变为原生层 core dump。
    assert _GPU_OWNER_THREAD_ID == threading.get_ident()
    try:
        result = _PIPELINE.run_gpu_stages(state)
        if torch.cuda.is_available():
            # CUDA kernel 默认异步执行。返回 state 前同步，确保当前文件的
            # GPU 工作已经结束，文件线程才能安全地进行 CPU export，同时
            # 下一项 GPU 任务也不会与当前任务的尾部 kernel 意外交叠。
            torch.cuda.synchronize()
        return result
    except Exception:
        if torch.cuda.is_available():
            # 异常路径释放未使用的 allocator cache，降低后续文件被上一个
            # 失败任务的显存高水位影响的概率。
            torch.cuda.empty_cache()
        raise
    finally:
        if cleanup_after and torch.cuda.is_available():
            # cleanup_after 仅在一个源文件的最后一个 chunk 为 True，保持
            # 原 PipelineV2 的按文件清理语义，避免每个 chunk 都清缓存。
            torch.cuda.empty_cache()


def _empty_cuda_cache() -> None:
    """Run CUDA allocator maintenance on the same thread as model inference."""

    assert _GPU_OWNER_THREAD_ID == threading.get_ident()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_one(task: ProcessTask) -> tuple[str, int, str]:
    """Prepare/export in this file thread; execute CUDA in the owner thread."""

    assert _PIPELINE is not None
    assert _GPU_EXECUTOR is not None
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
        # Decode/normalize stays in file threads. ffmpeg and NumPy work can
        # overlap the previous file's GPU inference without loading another
        # copy of the GPU models.
        chunk_states = _PIPELINE.standardize(
            bootstrap,
            standardizer=_get_standardizer(),
        )
        segment_count = 0

        for chunk_index, state in enumerate(chunk_states):
            t0 = time.perf_counter()

            # The executor has exactly one worker. Model construction and every
            # CUDA/native inference call therefore happen on the same OS thread.
            # 文件 worker 在 future.result() 处等待，但其他文件 worker 仍可
            # 并行 decode/normalize；这保留了 CPU/IO 与 GPU 推理的重叠。
            gpu_future = _GPU_EXECUTOR.submit(
                _run_gpu_stages,
                state,
                chunk_index == len(chunk_states) - 1,
            )
            state, vad_dur, refine_dur = gpu_future.result()

            # Export does not use CUDA and can overlap the next file's GPU work.
            state = _PIPELINE.export(
                state,
                chunk_index,
                task.output_folder,
            )
            PipelineV2.log_chunk_stats(state, t0, vad_dur, refine_dur)
            segment_count += len(state.segment_list or [])

        return (task.relative_path, segment_count, "")
    except PipelineError as exc:
        logger.error(
            f"pipeline_failed stage {exc.stage} msg {exc.message}",
            extra=bootstrap.log_tag,
        )
        return (
            task.relative_path,
            0,
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        logger.error(
            f"pipeline_failed unexpected {type(exc).__name__} {exc}",
            extra=bootstrap.log_tag,
        )
        return (
            task.relative_path,
            0,
            f"{type(exc).__name__}: {exc}",
        )


def _log_result(result: tuple[str, int, str]) -> None:
    relative_path, segment_count, error = result
    if error:
        logger.error(f"main_v2_failed file {relative_path} err {error}")
    else:
        logger.info(
            f"main_v2_done file {relative_path} segments {segment_count}"
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


def _run_group(
    tasks: list[ProcessTask],
    num_workers: int,
    description: str,
) -> None:
    if num_workers == 1:
        for task in tqdm.tqdm(tasks, desc=description):
            _log_result(_process_one(task))
        return

    # Only active threads hold decoded waveforms, so host-memory use is bounded
    # by num_workers rather than by the full manifest size.
    # 注意：num_workers 在当前表示“同时在途的文件线程数”，不再表示multiprocessing 进程数，也不会增加 GPU Pipeline 的副本数。
    with ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="pipeline-v2",
    ) as executor:
        futures = [executor.submit(_process_one, task) for task in tasks]
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc=description,
        ):
            _log_result(future.result())


def main() -> None:
    global _STANDARDIZATION_PARAMS, _GPU_EXECUTOR

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

    params = PipelineParams.from_config(args.config)

    # File threads lazily create their own CPU Standardizer. This keeps the
    # stateful Silero TorchScript model pinned to the thread that owns it.
    # 这里只保存不可变配置，不在主线程预先创建 Standardizer，否则模型仍会
    # 在主线程创建、文件线程调用，失去 thread-local 隔离的意义。
    _STANDARDIZATION_PARAMS = params.standardization

    # The initializer runs inside the executor's sole worker, so all model
    # construction and all later CUDA calls have stable thread affinity.
    _GPU_EXECUTOR = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pipeline-v2-gpu",
        initializer=_init_gpu_pipeline,
        initargs=(params,),
    )
    device_name, gpu_thread_id = _GPU_EXECUTOR.submit(_gpu_runtime_info).result()
    assert _PIPELINE is not None

    logger.info(
        f"main_v2 files {total} io_workers {num_workers} gpu_pipelines 1 "
        f"gpu_owner_threads 1 gpu_thread_id {gpu_thread_id} "
        f"device {device_name} output {args.output} "
        f"mode {'manifest' if manifest_mode else 'input'} pipeline_version v2 "
        f"allocator_conf {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}"
    )

    try:
        for shard_name, items in groups:
            tasks = _tasks_for_group(
                shard_name,
                items,
                args.output,
                manifest_mode,
            )
            _run_group(
                tasks,
                num_workers,
                description=f"pipeline_v2:{shard_name}",
            )
    finally:
        # Submit cleanup before shutdown so it runs on the CUDA-owner thread.
        # 不要在主线程直接调用 torch.cuda.empty_cache()：本轮改造要求所有
        # CUDA API（包括 allocator maintenance）都由同一个 owner 线程执行。
        if _GPU_EXECUTOR is not None:
            _GPU_EXECUTOR.submit(_empty_cuda_cache).result()
            _GPU_EXECUTOR.shutdown(wait=True)
            _GPU_EXECUTOR = None


if __name__ == "__main__":
    main()
    sys.exit(0)
