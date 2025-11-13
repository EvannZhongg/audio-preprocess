import csv
import os
import pathlib
import sys
import warnings
from functools import partial

import torch.multiprocessing as mp
import tqdm

from pipeline.arg_parser import get_cmd_args
from pipeline.global_var import init_pipeline_global
from utils.logger import Logger
from utils.tool import filter_manifest_by_report, get_audio_files, load_cfg

warnings.filterwarnings("ignore")


def get_audio_manifest(audio_path, base_dir):   
    relative_path = os.path.relpath(os.path.dirname(audio_path), base_dir)
    return {
        "RelativePath": relative_path,
        "FilePath": audio_path
    }

    
def main():
    # Use 'spawn' for CUDA safety in multiprocessing
    mp.set_start_method("spawn", force=True)

    args = get_cmd_args()

    # --- Main Process Setup ---
    main_logger = Logger.get_logger("main")
    main_cfg = load_cfg(args.config_path)

    # --- Override config with CLI arguments ---
    if args.separation_provider:
        main_cfg["separate"]["provider"] = args.separation_provider
        main_logger.info(f"Overriding separation provider with: {args.separation_provider}")

    if args.prepare_env:
        init_pipeline_global(main_cfg, args)
        sys.exit(0)

    # --- Determine Input Source ---
    manifest_entries = []
    if args.input_audio_path:
        manifest_entries.append(get_audio_manifest(args.input_audio_path, os.path.dirname(args.input_audio_path))) 
    elif args.manifest_path:
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
            manifest_entries.append(get_audio_manifest(audio_path, args.input_folder_path))
    else:
        main_logger.error("Error: You must provide either --manifest_path or --input_folder_path.")
        sys.exit(1)
        
    if not manifest_entries:
        main_logger.warning(f"No audio files found to process. Exiting.")
        sys.exit(1)

    # --- Resume from previous run ---
    manifest_entries = filter_manifest_by_report(manifest_entries, args.report_path)
    if not manifest_entries:
        main_logger.info("All files in the manifest have already been processed. Exiting.")
        sys.exit(1)

    # Create the main output directory
    os.makedirs(args.output_folder, exist_ok=True)
    main_logger.info(f"Processed data will be saved in: {args.output_folder}")

    num_workers = min(args.num_workers, len(manifest_entries))
    main_logger.info(f"Found {len(manifest_entries)} audio files. Processing with {num_workers} worker(s).")

    # --- Process Pool Execution ---
    init_args = (main_cfg, args)
    
    with mp.Pool(processes=num_workers, initializer=init_pipeline_global, initargs=init_args) as pool:
        from pipeline.main_process import main_process
        process_func = partial(main_process, output_folder=args.output_folder, report_path=args.report_path)
        results = list(tqdm.tqdm(pool.imap(process_func, manifest_entries), total=len(manifest_entries)))

    main_logger.info("--- All files have been processed. ---")
    
    if args.exit_pipeline:
        main_logger.info("exit_pipeline is True, exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
    sys.exit(0)
