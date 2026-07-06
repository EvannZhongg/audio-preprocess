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
from pipeline_v2_ray.config import load_ray_config
from pipeline_v2_ray.driver import ClusterDriver
from utils.tool import get_audio_files

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-config", default="configs/pipeline_v2_ray.yaml", help="ray hardware map")
    p.add_argument("--input", required=True, help="audio file or folder")
    p.add_argument("--output", required=True, help="output folder for exported jsons")
    p.add_argument("--address", default="auto", help="ray cluster address")
    return p.parse_args()


def collect_audio_paths(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return get_audio_files(str(p))
    print(f"input not found: {input_path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = parse_args()

    audio_paths = collect_audio_paths(args.input)
    if not audio_paths:
        logger.warning("no audio files to process")
        sys.exit(0)

    ray.init(address=args.address, ignore_reinit_error=True)
    try:
        ray_config = load_ray_config(args.ray_config)
        driver = ClusterDriver(ray_config)
        driver.run(audio_paths, args.output)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
    sys.exit(0)
