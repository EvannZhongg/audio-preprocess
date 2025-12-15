import os

LARGE_TEMP_PATH = f"{os.getcwd()}/TEMP" 

try:
    os.makedirs(LARGE_TEMP_PATH, exist_ok=True)
    os.environ["LARGE_TEMP_DIR"] = LARGE_TEMP_PATH
    os.environ["TMPDIR"] = LARGE_TEMP_PATH
    os.environ["TEMP"] = LARGE_TEMP_PATH
    os.environ["TMP"] = LARGE_TEMP_PATH
except Exception as e:
    print(f"Failed to set large temp dir: {e}")

import csv
import sys
import warnings
from functools import partial
from pathlib import Path

import torch.multiprocessing as mp
import tqdm

from pipeline.arg_parser import get_cmd_args
from pipeline.global_var import init_pipeline_global
from pipeline.main_process import main_process
from utils.logger import Logger
from utils.tool import filter_manifest_by_report, get_audio_files, load_cfg

warnings.filterwarnings("ignore")

def get_audio_manifest(audio_path: Path, base_dir: Path):
    """构建音频文件的 manifest 字典"""
    audio_path = Path(audio_path)
    base_dir = Path(base_dir)
    
    try:
        relative_path = audio_path.parent.relative_to(base_dir)
    except ValueError:
        relative_path = Path(".")

    return {
        "RelativePath": str(relative_path),
        "FilePath": str(audio_path)
    }

def collect_manifest_entries(args, logger):
    """根据命令行参数收集需要处理的音频文件列表"""
    manifest_entries = []
    
    if args.input_audio_path:
        p = Path(args.input_audio_path)
        manifest_entries.append(get_audio_manifest(p, p.parent))
        
    elif args.manifest_path:
        logger.info(f"Reading audio manifest from: {args.manifest_path}")
        try:
            with open(args.manifest_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                manifest_entries = [row for row in reader]
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Error reading manifest file: {e}")
            sys.exit(1)
            
    elif args.input_folder_path:
        folder_path = Path(args.input_folder_path)
        logger.info(f"Scanning for audio files in: {folder_path}")
        if not folder_path.is_dir():
            logger.error(f"Input folder not found: {folder_path}")
            sys.exit(1)
        
        audio_paths = get_audio_files(str(folder_path))
        for audio_path in audio_paths:
            manifest_entries.append(get_audio_manifest(audio_path, folder_path))
            
    else:
        logger.error("Error: You must provide either --manifest_path, --input_folder_path or --input_audio_path.")
        sys.exit(1)

    return manifest_entries

def safe_process_wrapper(entry, output_folder, report_path):
    try:
        main_process(entry, output_folder, report_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to process {entry.get('FilePath', 'unknown')}: {e}")
        # traceback.print_exc()

def main():
    args = get_cmd_args()

    # --- Main Process Setup ---
    main_logger = Logger.get_logger("main")
    main_cfg = load_cfg(args.config_path)


    if args.separation_provider:
        main_cfg["separate"]["provider"] = args.separation_provider
        main_logger.info(f"Overriding separation provider with: {args.separation_provider}")

    if args.prepare_env:
        init_pipeline_global(main_cfg, args)
        sys.exit(0)

    # --- Determine Input Source ---
    manifest_entries = collect_manifest_entries(args, main_logger)
        
    if not manifest_entries:
        main_logger.warning("No audio files found to process. Exiting.")
        sys.exit(1)

    manifest_entries = filter_manifest_by_report(manifest_entries, args.report_path)
    if not manifest_entries:
        main_logger.info("All files in the manifest have already been processed. Exiting.")
        sys.exit(0) 

    # Create the main output directory
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    main_logger.info(f"Processed data will be saved in: {output_folder}")

    # --- Execution Strategy ---
    total_files = len(manifest_entries)
    num_workers = min(args.num_workers, total_files)
    
    process_func = partial(main_process, output_folder=str(output_folder), report_path=args.report_path)

    main_logger.info(f"Found {total_files} audio files to process.")

    if num_workers > 1:
        main_logger.info(f"Starting multiprocessing pool with {num_workers} workers.")
        
        init_args = (main_cfg, args)
        
        with mp.Pool(processes=num_workers, initializer=init_pipeline_global, initargs=init_args) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_func, manifest_entries, chunksize=1), 
                total=total_files,
                desc="Processing"
            ))
    else:
        main_logger.info("Running in single-process mode (Sequentially).")
        init_pipeline_global(main_cfg, args)
        
        for entry in tqdm.tqdm(manifest_entries, desc="Processing"):
            process_func(entry)

    main_logger.info("--- All files have been processed. ---")
    
    if args.exit_pipeline:
        main_logger.info("exit_pipeline is True, exiting...")
        sys.exit(0)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass 
        
    main()
    sys.exit(0)
