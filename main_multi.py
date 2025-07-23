# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from functools import partial
import pathlib

import tqdm
import torch
from utils.logger import Logger, time_logger
from utils.tool import get_audio_files, load_cfg, detect_gpu
from main import init_worker, main_process_wrapper


def main():
    """
    Main function to orchestrate the multi-GPU processing of audio files.
    """
    # Use 'spawn' for CUDA safety in multiprocessing
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-GPU Audio Processing Pipeline")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Path to a CSV manifest file listing audio files to process (priority)."
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default=None,
        help="Path to a folder with audio files (used if manifest is not provided)."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="processed_data_multi",
        help="The root folder where all processed data will be saved."
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="Config file path"
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=2,
        help="Number of worker processes to spawn per GPU.",
    )
    parser.add_argument(
        "--disabled_gpu_ids",
        type=str,
        default="",
        help="Comma-separated list of disabled GPU IDs (e.g., '0,2').",
    )
    # Add other relevant arguments from main.py
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for ASR")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type for Whisper")
    parser.add_argument("--whisper_arch", type=str, default="medium", help="Whisper model architecture")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per worker")

    args = parser.parse_args()
    
    main_logger = Logger.get_logger("main_multi")
    main_cfg = load_cfg(args.config_path)

    # --- GPU Availability Check ---
    if not detect_gpu() or torch.cuda.device_count() == 0:
        main_logger.error("No GPUs detected. main_multi.py requires at least one GPU. Exiting.")
        sys.exit(1)

    total_gpus = torch.cuda.device_count()
    disabled_ids = [int(i.strip()) for i in args.disabled_gpu_ids.split(',') if i]
    available_gpus = [i for i in range(total_gpus) if i not in disabled_ids]

    if not available_gpus:
        main_logger.error("All GPUs are disabled or unavailable. Exiting.")
        sys.exit(1)

    main_logger.info(f"Total GPUs: {total_gpus}, Available GPUs for this run: {available_gpus}")

    # --- Input File Discovery ---
    manifest_entries = []
    if args.manifest_path:
        main_logger.info(f"Reading audio manifest from: {args.manifest_path}")
        try:
            with open(args.manifest_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                manifest_entries = [row for row in reader]
        except (FileNotFoundError, IOError) as e:
            main_logger.error(f"Error reading manifest file: {e}")
            sys.exit(1)
    elif args.input_folder_path:
        main_logger.info(f"Scanning for audio files in: {args.input_folder_path}")
        if not os.path.isdir(args.input_folder_path):
            main_logger.error(f"Input folder not found: {args.input_folder_path}")
            sys.exit(1)
        
        audio_paths = get_audio_files(args.input_folder_path)
        for audio_path in audio_paths:
            path_parts = pathlib.Path(audio_path).parts
            podcast_name = path_parts[-2] if len(path_parts) > 1 else "UnknownPodcast"
            episode_name = os.path.splitext(os.path.basename(audio_path))[0]
            manifest_entries.append({
                "PodcastName": podcast_name,
                "EpisodeName": episode_name,
                "FilePath": audio_path
            })
    else:
        main_logger.error("Error: You must provide either --manifest_path or --input_folder_path.")
        sys.exit(1)
        
    if not manifest_entries:
        main_logger.warning("No audio files found to process. Exiting.")
        sys.exit(0)

    # --- Task Distribution ---
    num_files = len(manifest_entries)
    num_gpus = len(available_gpus)
    files_per_gpu = [[] for _ in range(num_gpus)]
    for i, file_entry in enumerate(manifest_entries):
        files_per_gpu[i % num_gpus].append(file_entry)

    main_logger.info(f"Distributing {num_files} files among {num_gpus} GPUs.")
    for i, gpu_id in enumerate(available_gpus):
        main_logger.info(f"  - GPU {gpu_id} will process {len(files_per_gpu[i])} files.")

    # Create the main output directory
    os.makedirs(args.output_folder, exist_ok=True)
    main_logger.info(f"Processed data will be saved in: {args.output_folder}")

    # --- Process Pool Execution for each GPU ---
    total_workers = args.num_workers_per_gpu * num_gpus
    init_args = (main_cfg, args)
    
    process_func = partial(main_process_wrapper, output_folder=args.output_folder)

    # We can use a single pool and let the init_worker handle GPU assignment
    # The worker_id is assigned by the pool, and we use worker_id % num_gpus to assign a GPU
    # This is exactly how main.py does it.
    main_logger.info(f"Creating a single process pool with {total_workers} workers for {num_gpus} GPUs.")
    
    with mp.Pool(processes=total_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm.tqdm(pool.imap(process_func, manifest_entries), total=num_files))

    main_logger.info("--- All files have been processed by all GPUs. ---")


if __name__ == "__main__":
    main()
