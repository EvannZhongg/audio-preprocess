"""Minimal entrypoint for PipelineV2.

Usage:
    python main_v2.py --config <config.json> --input <audio_or_folder> [--num-workers N]
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

import torch.multiprocessing as mp
import tqdm

import logger
from pipeline_v2.params import PipelineParams
from pipeline_v2.pipeline import PipelineV2
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


_WORKER_PIPE: PipelineV2 | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True, help="audio file or folder")
    p.add_argument("--output", required=True, help="output folder for exported jsons")
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args()


def collect_audio_paths(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return get_audio_files(str(p))
    print(f"input not found: {input_path}", file=sys.stderr)
    sys.exit(1)


def _init_worker(config_path: str) -> None:
    global _WORKER_PIPE
    params = PipelineParams.from_config(config_path)
    _WORKER_PIPE = PipelineV2(params)


def _process_one(task: tuple[str, str]) -> tuple[str, int, str]:
    assert _WORKER_PIPE is not None
    audio_path, output_folder = task
    name = os.path.basename(audio_path)
    try:
        states = _WORKER_PIPE.run(audio_path, output_folder)
        n = sum(len(s.segment_list) if s.segment_list else 0 for s in states)
        return (name, n, "")
    except Exception as e:
        return (name, 0, f"{type(e).__name__}: {e}")


def main() -> None:
    args = parse_args()

    audio_paths = collect_audio_paths(args.input)
    if not audio_paths:
        logger.warning("no audio files to process")
        sys.exit(0)

    total = len(audio_paths)
    num_workers = max(1, min(args.num_workers, total))
    logger.info(f"main_v2 files {total} workers {num_workers} output {args.output}")

    tasks = [(p, args.output) for p in audio_paths]

    if num_workers == 1:
        _init_worker(args.config)
        results = [
            _process_one(t)
            for t in tqdm.tqdm(tasks, desc="pipeline_v2")
        ]
    else:
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(args.config,),
        ) as pool:
            results = list(tqdm.tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=1),
                total=total,
                desc="pipeline_v2",
            ))

    for name, n, err in results:
        if err:
            logger.error(f"main_v2_failed file {name} err {err}")
        else:
            logger.info(f"main_v2_done file {name} segments {n}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
    sys.exit(0)
