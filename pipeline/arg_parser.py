import argparse

def get_cmd_args():
    parser = argparse.ArgumentParser()
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
        "--input_audio_path",
        type=str,
        default=None,
        help="Path to a audio file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="processed_data",
        help="The root folder where all processed data will be saved."
    )
    parser.add_argument(
        "--separation_provider",
        type=str,
        default=None,
        help="Override the separation provider from config (e.g., 'smru' or 'uvr')."
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="processing_report.csv",
        help="Path to the processing report CSV file for resuming progress.",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="medium",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    # 打docker的时候用来下载模型
    parser.add_argument(
        "--prepare_env",
        type=bool,
        default=False
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(get_cmd_args())