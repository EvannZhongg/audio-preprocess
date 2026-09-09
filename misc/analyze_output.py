import os
import argparse
import concurrent.futures
from collections import defaultdict
import soundfile as sf
import numpy as np

# Define common audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a'}

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.
    Returns duration or None if it fails.
    """
    try:
        with sf.SoundFile(file_path) as f:
            # For RAW files, duration might not be directly available,
            # but for most formats, this is the most reliable way.
            duration = len(f) / f.samplerate
        return duration
    except Exception as e:
        # This can fail for corrupted files or unsupported formats
        # that might have slipped through the extension check.
        print(f"Warning: Could not read duration for {file_path}. Error: {e}")
        return None

def analyze_directory(root_folder):
    """
    Analyzes audio files in a directory to get total count, duration,
    and distribution across different length intervals using multithreading.
    """
    print(f"[*] Starting analysis of folder: {root_folder}")
    
    # 1. Find all audio files recursively
    audio_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print("[!] No audio files found in the specified directory.")
        return

    total_files = len(audio_files)
    print(f"[*] Found {total_files} audio files. Analyzing durations with multiple threads...")

    # 2. Use a ThreadPoolExecutor to get durations in parallel
    total_duration = 0
    durations = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(get_audio_duration, file_path): file_path for file_path in audio_files}
        for future in concurrent.futures.as_completed(future_to_file):
            duration = future.result()
            if duration is not None:
                total_duration += duration
                durations.append(duration)

    # 3. Define duration intervals and calculate distribution
    intervals = [
        (0, 3), (3, 5), (5, 7), (7, 9), (9, 12), (12, 15), (15, np.inf)
    ]
    distribution = defaultdict(int)
    
    for d in durations:
        for low, high in intervals:
            if low <= d < high:
                distribution[(low, high)] += 1
                break

    # 4. Print the summary report
    print("\n--- Audio Analysis Report ---")
    print(f"Scanned Folder: {root_folder}")
    print("-" * 30)
    
    # Format total duration into H:M:S
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total Audio Files: {total_files}")
    print(f"Total Duration: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
    print("-" * 30)
    
    print("Duration Distribution:")
    print(f"{'Interval (s)':<15} | {'Count':<10} | {'Percentage':<12}")
    print("-" * 45)
    
    for low, high in intervals:
        count = distribution.get((low, high), 0)
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        
        if high == np.inf:
            interval_str = f"{low}+"
        else:
            interval_str = f"{low}-{high}"
            
        print(f"{interval_str:<15} | {count:<10} | {percentage:11.2f}%")
        
    print("-" * 45)
    print("--- End of Report ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast, multi-threaded analysis of audio files in a directory."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="The root folder containing processed audio files to analyze."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: The specified folder does not exist: {args.folder}")
    else:
        analyze_directory(args.folder) 