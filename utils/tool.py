# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import tqdm
from pydub import AudioSegment

from utils.logger import Logger, time_logger


def load_cfg(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"{config_path} not found. Please copy, configure, and rename `config.json.example` to `{config_path}`."
        )
    with open(config_path, "r") as f:
        try:
            cfg = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise TypeError(
                "Please finish the `// TODO:` in the `config.json` file before running the script. Check README.md for details."
            )
    return cfg


def write_wav(path, sr, x):
    """Write numpy array to WAV file."""
    sf.write(path, x, sr)


def write_mp3(path, sr, x):
    """Convert numpy array to MP3."""
    try:
        # Ensure x is in the correct format and normalize if necessary
        if x.dtype != np.int16:
            # Normalize the array to fit in int16 range if it's not already int16
            x = np.int16(x / np.max(np.abs(x)) * 32767)

        # Create audio segment from numpy array
        audio = AudioSegment(
            x.tobytes(), frame_rate=sr, sample_width=x.dtype.itemsize, channels=1
        )
        # Export as MP3 file
        audio.export(path, format="mp3")
    except Exception as e:
        print(e)
        print("Error: Failed to write MP3 file.")


def get_short_hash(text, length=6):
    """
    Generates a short, deterministic hash from a string to create a unique file-specific prefix.
    """
    if isinstance(text, str):
        text = text.encode('utf-8')
    
    hasher = hashlib.sha1(text)
    return hasher.hexdigest()[:length]


def get_audio_files(folder_path):
    """Get all audio files in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_specific_files(folder_path, ext):
    """Get specific files with a given extension in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith(ext):
                audio_files.append(os.path.join(root, file))
    return audio_files


def export_to_srt(asr_result, file_path):
    """Export ASR result to SRT file."""
    with open(file_path, "w") as f:

        def format_time(seconds):
            return (
                time.strftime("%H:%M:%S", time.gmtime(seconds))
                + f",{int(seconds * 1000 % 1000):03d}"
            )

        for idx, segment in enumerate(asr_result):
            f.write(f"{idx + 1}\n")
            f.write(
                f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n"
            )
            f.write(f"{segment['speaker']}: {segment['text']}\n\n")


def detect_gpu():
    """Detect if GPU is available and print related information."""
    logger = Logger.get_logger()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        logger.info("ENV: CUDA_VISIBLE_DEVICES not set, use default setting")
    else:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"ENV: CUDA_VISIBLE_DEVICES = {gpu_id}")

    if not torch.cuda.is_available():
        logger.error("Torch CUDA: No GPU detected. torch.cuda.is_available() = False.")
        return False

    num_gpus = torch.cuda.device_count()
    logger.debug(f"Torch CUDA: Detected {num_gpus} GPUs.")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        logger.debug(f" * GPU {i}: {gpu_name}")

    logger.debug("Torch: CUDNN version = " + str(torch.backends.cudnn.version()))
    if not torch.backends.cudnn.is_available():
        logger.error("Torch: CUDNN is not available.")
        return False
    logger.debug("Torch: CUDNN is available.")

    ort_providers = ort.get_available_providers()
    logger.debug(f"ORT: Available providers: {ort_providers}")
    if "CUDAExecutionProvider" not in ort_providers:
        logger.warning(
            "ORT: CUDAExecutionProvider is not available. "
            "Please install a compatible version of ONNX Runtime. "
            "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
        )

    return True


def get_gpu_nums():
    """Get GPU nums by nvidia-smi."""
    logger = Logger.get_logger()
    try:
        result = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
        gpus_count = int(result.decode().strip())
    except Exception as e:
        logger.error("Error occurred while getting GPU count: " + str(e))
        gpus_count = 8  # Default to 8 if GPU count retrieval fails
    return gpus_count


def check_env(logger):
    """Check environment variables."""
    if "http_proxy" in os.environ:
        logger.info(f"ENV: http_proxy = {os.environ['http_proxy']}")
    else:
        logger.info("ENV: http_proxy not set")

    if "https_proxy" in os.environ:
        logger.info(f"ENV: https_proxy = {os.environ['https_proxy']}")
    else:
        logger.info("ENV: https_proxy not set")

    if "HF_ENDPOINT" in os.environ:
        logger.info(
            f"ENV: HF_ENDPOINT = {os.environ['HF_ENDPOINT']}, if downloading slow, try `unset HF_ENDPOINT`"
        )
    else:
        logger.info("ENV: HF_ENDPOINT not set")

    hostname = os.popen("hostname").read().strip()
    logger.debug(f"HOSTNAME: {hostname}")

    environ_path = os.environ["PATH"]
    environ_ld_library = os.environ.get("LD_LIBRARY_PATH", "")
    logger.debug(f"ENV: PATH = {environ_path}, LD_LIBRARY_PATH = {environ_ld_library}")


@time_logger
def export_to_mp3(audio, asr_result, folder_path, file_name):
    """Export segmented audio to MP3 files."""
    sr = audio["sample_rate"]
    audio = audio["waveform"]

    os.makedirs(folder_path, exist_ok=True)

    # Function to process each segment in a separate thread
    def process_segment(idx, segment):
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = audio[start:end]
        split_audio = librosa.to_mono(split_audio)
        out_file = f"{file_name}_{idx}.mp3"
        out_path = os.path.join(folder_path, out_file)
        write_mp3(out_path, sr, split_audio)

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=72) as executor:
        # Submit each segment processing as a separate thread
        futures = [
            executor.submit(process_segment, idx, segment)
            for idx, segment in enumerate(asr_result)
        ]

        # Wait for all threads to complete
        for future in tqdm.tqdm(
            futures, total=len(asr_result), desc="Exporting to MP3"
        ):
            future.result()


@time_logger
def export_to_libritts(audio, asr_result, folder_path, file_name):
    """Export segmented audio and text to LibriTTS format."""
    sr = audio["sample_rate"]
    waveform = audio["waveform"]

    # Keep track of utterance count for each speaker
    speaker_counts = {}

    for segment in tqdm.tqdm(asr_result, desc="Exporting to LibriTTS format"):
        speaker_id = segment.get("speaker", "UNKNOWN_SPEAKER")

        # Create speaker-specific directory if it doesn't exist
        speaker_folder = os.path.join(folder_path, speaker_id)
        os.makedirs(speaker_folder, exist_ok=True)

        # Update and get the utterance count for the current speaker
        count = speaker_counts.get(speaker_id, 0) + 1
        speaker_counts[speaker_id] = count

        # Define file basenames like SPEAKER_00-001
        base_filename = f"{speaker_id}-{str(count).zfill(5)}"

        # 1. Save the audio segment as WAV
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = waveform[start:end]

        # Peak normalize and convert to 16-bit PCM, same as in MP3 export
        max_abs_val = np.max(np.abs(split_audio))
        if max_abs_val > 0:
            split_audio = (split_audio / max_abs_val) * 32767
        split_audio = split_audio.astype(np.int16)

        wav_path = os.path.join(speaker_folder, f"{base_filename}.wav")
        write_wav(wav_path, sr, split_audio)

        # 2. Save the transcription to a .normalized.txt file
        text_path = os.path.join(speaker_folder, f"{base_filename}.normalized.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(segment["text"])


@time_logger
def export_to_wav(audio, asr_result, folder_path, file_name):
    """Export segmented audio to WAV files."""
    sr = audio["sample_rate"]
    audio = audio["waveform"]

    os.makedirs(folder_path, exist_ok=True)

    for idx, segment in enumerate(tqdm.tqdm(asr_result, desc="Exporting to WAV")):
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = audio[start:end]
        split_audio = librosa.to_mono(split_audio)
        out_file = f"{file_name}_{idx}.wav"
        out_path = os.path.join(folder_path, out_file)
        write_wav(out_path, sr, split_audio)


@time_logger
def export_to_default(audio, asr_result, folder_path, file_name):
    """Export segmented audio and metadata to default format (LibriTTS structure with JSON metadata)."""
    sr = audio["sample_rate"]
    waveform = audio["waveform"]

    # Keep track of utterance count for each speaker
    speaker_counts = {}

    for segment in tqdm.tqdm(asr_result, desc="Exporting to default format"):
        speaker_id = segment.get("speaker", "UNKNOWN_SPEAKER")

        # Create speaker-specific directory if it doesn't exist
        speaker_folder = os.path.join(folder_path, speaker_id)
        os.makedirs(speaker_folder, exist_ok=True)

        # Update and get the utterance count for the current speaker
        count = speaker_counts.get(speaker_id, 0) + 1
        speaker_counts[speaker_id] = count

        # Define file basenames like SPEAKER_00-001
        base_filename = f"{speaker_id}-{str(count).zfill(5)}"

        # 1. Save the audio segment as MP3
        start, end = int(segment["start"] * sr), int(segment["end"] * sr)
        split_audio = waveform[start:end]

        # Convert to mono and normalize for MP3
        split_audio = librosa.to_mono(split_audio)
        
        mp3_path = os.path.join(speaker_folder, f"{base_filename}.mp3")
        write_mp3(mp3_path, sr, split_audio)

        # 2. Create JSON metadata with DNSMOS, duration, ASR_SenseVoice, etc.
        duration = segment["end"] - segment["start"]
        metadata = {
            "dnsmos": segment.get("dnsmos", 0.0),
            "duration": duration,
            "asr_sensevoice": segment.get("text", ""),
            "speaker": speaker_id
        }
        
        # Add any additional metadata fields that exist in the segment
        for key, value in segment.items():
            if key not in ["start", "end", "text", "speaker", "dnsmos"]:
                metadata[key] = value

        # 3. Save the metadata to a JSON file
        json_path = os.path.join(speaker_folder, f"{base_filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


@time_logger
def export_to_metadata(audio, asr_result, folder_path, meta_info, file_name):
    """Export segmented audio and metadata to default format (LibriTTS structure with JSON metadata)."""
    sr = audio["sample_rate"]
    waveform = audio["waveform"]

    # save_audio
    wav_path = os.path.join(folder_path, f"{file_name}.wav")
    write_wav(wav_path, sr, waveform)

    # update json
    meta_info.clear_sentences()
    for segment in tqdm.tqdm(asr_result, desc="Exporting to default format"):
        speaker_id = segment.get("speaker", "UNKNOWN_SPEAKER")

        setence_metadata = {
            "utt_id": file_name,
            "speaker_id": speaker_id,
            "speaker_min_similarity": f'{segment.get("min_similarity", 0.61):.4f}',
            "language": segment.get('language', 'zh'),
            "time_range": {
                "duration": segment.get("duration", 0.0),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
            },
            "transcription_info": {
                "text": segment.get("text", ""),
                "val_text": segment.get("val_text", ""),
                "norm_text": segment.get("norm_text", ""),
                "wer": f'{segment.get("wer", 0.):.4f}',
                "avg_char_duration": f'{segment.get("avg_char_duration", 0.2):.4f}',
            },
            "metrics_info":{
                "dnsmos": f'{segment.get("dnsmos", 0.0):.4f}',
                "c50": f'{segment.get("c50", 0.0):.4f}',
                "snr": f'{segment.get("snr", 0.0):.4f}',
            }
        }
        meta_info.add_sentence(setence_metadata)

    save_json_path = os.path.join(folder_path, f"{file_name}.json")
    meta_info.save_to_file(save_json_path)


def get_char_count(text):
    """
    Get the character count of a given text, excluding punctuation and spaces.
    """
    # Using regular expression to remove punctuation and spaces
    cleaned_text = re.sub(r"[,.!?\"'，。！？“”‘’ ]", "", text)
    char_count = len(cleaned_text)
    return char_count


def calculate_audio_stats(data, metrics_filter_cfg):
    """"
    Reading the proviced json, calculate and return the audio ID and their duration that meet the given filtering criteria.

    Args:
        data: JSON.
        metrics_filter_cfg: Configuration dictionary containing filtering criteria.
    Returns:
        valid_audio_stats: A list containing tuples of audio ID and their duration.
    """

    min_duration = metrics_filter_cfg.get("min_duration", 3)
    max_duration = metrics_filter_cfg.get("max_duration", 30)
    min_dnsmos = metrics_filter_cfg.get("fixed_dnsmos_threshold", 3.0)
    min_char_count = metrics_filter_cfg.get("min_char_count", 2)

    all_audio_stats = []
    valid_audio_stats = []
    avg_durations = []
    avg_char_durations = []


    # iterate over each entry in the JSON to collect the average duration of the phonemes
    for entry in data:
        # remove punctuation and spaces
        char_count = get_char_count(entry["text"])
        duration = entry["end"] - entry["start"]
        if char_count > 0:
            avg_durations.append(duration / char_count)

    # calculate the bounds for the average character duration
    if len(avg_durations) > 0:
        q1 = np.percentile(avg_durations, 25)
        q3 = np.percentile(avg_durations, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    else:
        # if no valid character data, use default values
        lower_bound, upper_bound = 0, np.inf

    # iterate over each entry in the JSON to apply all filtering criteria
    for idx, entry in enumerate(data):
        duration = entry["end"] - entry["start"]
        dnsmos = entry["dnsmos"]
        # remove punctuation and spaces
        char_count = get_char_count(entry["text"])
        if char_count > 0:
            avg_char_duration = duration / char_count
        else:
            avg_char_duration = 0

        # collect the duration of all audios
        all_audio_stats.append((idx, duration))

        # apply filtering criteria
        if (
            (min_duration <= duration <= max_duration)  # withing duration range
            and (dnsmos >= min_dnsmos)
            and (char_count >= min_char_count)
            and (
                lower_bound <= avg_char_duration <= upper_bound
            )  # average character duration within bounds
        ):
            valid_audio_stats.append((idx, duration))
            avg_char_durations.append(avg_char_duration)

    return valid_audio_stats, all_audio_stats, avg_char_durations


def filter_manifest_by_report(manifest_entries, report_path):
    """
    Filters a list of manifest entries by removing those already processed,
    based on a processing report CSV file.

    Args:
        manifest_entries (list): A list of dictionaries, where each dictionary
                                 represents a file to be processed and must
                                 contain 'RelativePath'.
        report_path (str): The path to the processing report CSV file.

    Returns:
        list: A new list of manifest entries containing only the items
              that have not yet been processed.
    """
    logger = Logger.get_logger()
    total_count_initial = len(manifest_entries)

    if not os.path.exists(report_path):
        logger.info(
            f"Processing report '{report_path}' not found. "
            "Assuming no files have been processed yet."
        )
        return manifest_entries

    try:
        import pandas as pd

        report_df = pd.read_csv(report_path)
        # Create a set of tuples for quick lookup
        processed_set = set(report_df["RelativePath"])
        logger.info(
            f"Found {len(processed_set)} entries in the processing report."
        )
    except (Exception, ImportError) as e:
        logger.error(
            f"Failed to read or parse processing report with pandas: {e}. "
            "Processing all files as a fallback. Please consider `pip install pandas`."
        )
        # Fallback to manual CSV reading if pandas is not available or fails
        processed_set = set()
        try:
            import csv
            with open(report_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'RelativePath' in row:
                        processed_set.add((row['RelativePath']))
            logger.info(f"Fallback reader found {len(processed_set)} entries.")
        except Exception as csv_e:
            logger.error(f"Fallback CSV reader also failed: {csv_e}. Processing all files.")
            return manifest_entries

    # Filter the manifest entries
    unprocessed_entries = [
        entry
        for entry in manifest_entries
        if entry["RelativePath"] not in processed_set
    ]

    processed_count = total_count_initial - len(unprocessed_entries)

    if processed_count > 0:
        logger.info(
            f"Resuming from report: Total files in manifest = {total_count_initial}, "
            f"Completed = {processed_count}, Remaining = {len(unprocessed_entries)}. "
            "Skipping completed files."
        )

    return unprocessed_entries
