import argparse
import hashlib
import json
import os
import sys
import warnings
from functools import partial

# import google.generativeai as genai
# import jiwer
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
import tqdm
from models import (dnsmos, funasr_asr, separate_fast, silero_vad,
                    )
from models.eres2net.ERes2NetV2 import ERes2NetV2
from models.eres2net.features import FBank
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import \
    SpeakerDiarization as PyannoteSpeakerDiarization
from pydub import AudioSegment
from utils.logger import Logger, time_logger
from utils.tool import (calculate_audio_stats, check_env, detect_gpu,
                        export_to_libritts, export_to_mp3, get_audio_files,
                        load_cfg)
import traceback
import csv
import pathlib

warnings.filterwarnings("ignore")
audio_count = 0


# --- Globals for Multiprocessing ---
cfg = None
g_logger = None
# Models
dia_pipeline = None
asr_model = None
whisper_asr_model = None
funasr_asr_model = None
vad = None
separate_predictor1 = None
dnsmos_compute_score = None
refinement_model = None
refinement_feature_extractor = None
# ASR options
supported_languages = None
multilingual_flag = None
# Args
batch_size = 8
device = None
g_args = None


def get_short_hash(text, length=6):
    """
    Generates a short, deterministic hash from a string to create a unique file-specific prefix.
    """
    if isinstance(text, str):
        text = text.encode('utf-8')
    
    hasher = hashlib.sha1(text)
    return hasher.hexdigest()[:length]


def init_worker(config, cli_args):
    """
    Initializes a worker process.
    - Sets up logger and device (GPU).
    - Loads all models into global variables for this process.
    """
    global logger, cfg, g_args, device, batch_size, supported_languages, multilingual_flag
    global dia_pipeline, asr_model, whisper_asr_model, funasr_asr_model, vad
    global separate_predictor1, dnsmos_compute_score, refinement_model, refinement_feature_extractor

    from multiprocessing.process import current_process
    worker_id_str = current_process().name
    worker_id = int(worker_id_str.split('-')[-1]) - 1

    # 1. Setup globals
    g_args = cli_args
    cfg = config
    batch_size = g_args.batch_size
    logger = Logger.get_logger(f"worker_{worker_id}")

    # 2. Setup device
    num_gpus = torch.cuda.device_count() if detect_gpu() else 0
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        logger.info(f"Worker {worker_id} using GPU {gpu_id}")
        device_name = f"cuda:{gpu_id}"
        simple_device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info(f"Worker {worker_id} using CPU")
        device_name = "cpu"
        simple_device_name = "cpu"
        device = torch.device(device_name)
        # whisperX expects compute type: int8 on CPU
        logger.info(f"Worker {worker_id} overriding compute type to int8 for CPU.")
    
    logger.debug(f"Worker {worker_id} loading models...")

    # 3. Load all models
    # Diarization Provider Loading
    logger.debug(" * Loading Speaker Diarization Model (pyannote)")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("huggingface_token must start with 'hf', check the config file.")
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)

    # ASR Model Loading
    logger.debug(" * Loading ASR Model(s)")
    asr_provider = cfg.get("asr_provider", "whisper")
    asr_validation_enabled = cfg.get("asr_validation", {}).get("enable", False)

    if asr_validation_enabled:
        logger.info("Loading both Whisper and FunASR for cross-validation.")
        whisper_asr_model = whisper_asr.load_asr_model(g_args.whisper_arch, device_name, compute_type=g_args.compute_type)
        funasr_cfg = cfg.get("funasr", {})
        funasr_asr_model = funasr_asr.load_asr_model(
            model_dir=funasr_cfg.get("model_dir", "iic/SenseVoiceSmall"), device=device_name
        )
    else:
        if asr_provider == "gemini":
            if "gemini" not in cfg:
                raise ValueError("Gemini configuration not found in config.json")
            asr_model = gemini_asr.load_asr_model(
                api_key=cfg["gemini"].get("api_key"),
                model_name=cfg["gemini"].get("model", "genimi-2.5-flash"),
            )
        elif asr_provider == "funasr":
            if "funasr" not in cfg:
                raise ValueError("FunASR configuration not found in config.json")
            asr_model = funasr_asr.load_asr_model(
                model_dir=cfg["funasr"].get("model_dir", "iic/SenseVoiceSmall"), device=device_name
            )
        elif asr_provider == "paraformer":
            if "paraformer" not in cfg:
                raise ValueError("Paraformer configuration not found in config.json")
            paraformer_cfg = cfg["paraformer"]
            asr_model = funasr_asr.load_asr_model(
                model_dir=paraformer_cfg.get("model_dir", "paraformer-zh"), device=device_name
            )
        else: # Default to Whisper
            asr_model = whisper_asr.load_asr_model(
                g_args.whisper_arch, device_name, compute_type=g_args.compute_type, threads=g_args.threads,
                asr_options={"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."}
            )

    # VAD
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)

    # Background Noise Separation
    logger.debug(" * Loading Background Noise Model")
    separate_predictor1 = separate_fast.Predictor(args=cfg["separate"]["step1"], device=simple_device_name)

    # DNSMOS Scoring
    logger.debug(" * Loading DNSMOS Model")
    primary_model_path = cfg["mos_model"]["primary_model_path"]
    dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, simple_device_name)

    # Refinement Model
    refinement_cfg = cfg.get("embedding_refinement", {})
    if refinement_cfg.get("enable", True):
        logger.debug(" * Loading ERes2Net Model for Refinement")
        eres2net_path = refinement_cfg.get("eres2net_model_path")
        if eres2net_path and os.path.exists(eres2net_path):
            refinement_model = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
            pretrained_state = torch.load(eres2net_path, map_location=device)
            refinement_model.load_state_dict(pretrained_state)
            refinement_model.to(device)
            refinement_model.eval()
            refinement_feature_extractor = FBank()
        else:
            logger.warning("ERes2Net model path not found or specified, skipping refinement.")
            refinement_model = None

    # Language flags
    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    
    logger.debug(f"Worker {worker_id} finished loading models.")


def main_process_wrapper(manifest_entry, output_folder):
    """
    A wrapper for main_process to be used with pool.map.
    It constructs the save path and passes all necessary info.
    """
    podcast_name = manifest_entry["PodcastName"]
    episode_name = manifest_entry["EpisodeName"]
    audio_path = manifest_entry["FilePath"]

    # --- Skip large files ---
    try:
        file_size = os.path.getsize(audio_path)
        MAX_FILE_SIZE_BYTES = 900 * 1024 * 1024
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"Skipping file '{os.path.basename(audio_path)}' because its size "
                f"({file_size / 1024 / 1024:.2f} MB) exceeds the limit of {MAX_FILE_SIZE_BYTES / 1024 / 1024:.2f} MB."
            )
            return None
    except Exception as e:
        logger.error(f"Could not get file size for {audio_path}: {e}")
        return None

    # Construct the structured output path
    # e.g., output_folder/PodcastName/EpisodeName
    save_path = os.path.join(output_folder, podcast_name, episode_name)

    return main_process(audio_path, podcast_name, episode_name, save_path=save_path)


def append_to_report(report_path, podcast_name, episode_name, file_path, initial_duration, final_duration):
    """
    Appends a new row to the processing report CSV file.
    Creates the file and writes the header if it doesn't exist.
    """
    file_exists = os.path.isfile(report_path)
    retention_rate = (final_duration / initial_duration) * 100 if initial_duration > 0 else 0
    retention_rate = min(100.0, retention_rate)

    with open(report_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "PodcastName", 
                "EpisodeName", 
                "FilePath",
                "InitialDuration(s)", 
                "FinalDuration(s)", 
                "RetentionRate(%)"
            ])
        
        writer.writerow([
            podcast_name, 
            episode_name, 
            file_path,
            f"{initial_duration:.2f}", 
            f"{final_duration:.2f}", 
            f"{retention_rate:.2f}"
        ])


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }


@time_logger
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    logger.debug(f"vocals shape before resample: {vocals.shape}")
    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    logger.debug(f"vocals shape after resample: {vocals.shape}")
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, only use one channel

    return audio


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(dia_pipeline, audio, provider="pyannote"):
    """
    Perform speaker diarization on the given audio.

    Args:
        dia_pipeline: The loaded diarization pipeline object.
        audio (dict): A dictionary containing the audio waveform and sample rate.
        provider (str): The name of the provider ('pyannote').

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A dataframe containing segments with speaker labels.
            - dict: A dictionary mapping speaker labels to their embedding centroids.
    """
    logger.debug(f"Start speaker diarization with provider: {provider}")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    speaker_centroids = {}

    if provider == "pyannote":
        waveform = torch.tensor(audio["waveform"]).to(dia_pipeline.device)
        waveform = torch.unsqueeze(waveform, 0)
        # Pass return_embeddings=True to get speaker centroids
        segments, embeddings = dia_pipeline(
            {"waveform": waveform, "sample_rate": audio["sample_rate"]},
            return_embeddings=True,
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        # Create a mapping from speaker labels to their centroid embeddings
        for i, speaker in enumerate(segments.labels()):
            speaker_centroids[speaker] = embeddings[i]

    else:
        raise ValueError(f"Unsupported diarization provider: {provider}")

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df, speaker_centroids


@time_logger
def cut_by_speaker_label(vad_list, audio_duration, stats, step_name="post_process_vad"):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.
    Also adds a grace period to the end of segments to reduce cut-offs.
    This function now internally tracks and updates statistics.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.
        audio_duration (float): Total duration of the audio in seconds.
        stats (dict): The main statistics dictionary to be updated.
        step_name (str): The name of the step for statistics tracking.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 20  # max segment length in seconds
    GRACE_PERIOD_START_S = 0.00
    GRACE_PERIOD_END_S = 0.03
    updated_list = []

    # --- Internal Statistics ---
    discarded_long_count = 0
    discarded_long_duration = 0.0

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            duration = vad["end"] - vad["start"]
            logger.warning(
                f"cut_by_speaker_label > Discarding segment for speaker {vad['speaker']} "
                f"because its duration ({duration:.2f}s) is longer than "
                f"MAX_SEGMENT_LENGTH ({MAX_SEGMENT_LENGTH}s)."
            )
            # Track discard due to max length
            discarded_long_count += 1
            discarded_long_duration += duration
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    # Calculate discards from the final length filtering
    count_before_min_len_filter = len(updated_list)
    duration_before_min_len_filter = sum(s["end"] - s["start"] for s in updated_list)

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]
    
    count_after_min_len_filter = len(filter_list)
    duration_after_min_len_filter = sum(s["end"] - s["start"] for s in filter_list)

    discarded_short_count = count_before_min_len_filter - count_after_min_len_filter
    discarded_short_duration = duration_before_min_len_filter - duration_after_min_len_filter

    logger.debug(
        f"cut_by_speaker_label > removed: {discarded_short_count} segments by length"
    )

    # Update the main statistics dictionary
    stats['steps'][step_name]['discarded_count'] = discarded_long_count + discarded_short_count
    stats['steps'][step_name]['discarded_duration'] = discarded_long_duration + discarded_short_duration

    # --- Add Grace Period Logic ---
    if not filter_list:
        return filter_list

    logger.debug(
        f"cut_by_speaker_label > Applying {GRACE_PERIOD_START_S}s grace period to segment starts and {GRACE_PERIOD_END_S}s to ends."
    )

    # First, handle the end times to avoid overlap with the *next* segment
    # Iterate up to the second to last segment
    for i in range(len(filter_list) - 1):
        current_segment = filter_list[i]
        next_segment_start = filter_list[i + 1]["start"]

        # Add grace period, ensuring it doesn't extend into the next segment
        new_end = current_segment["end"] + GRACE_PERIOD_END_S
        current_segment["end"] = min(new_end, next_segment_start)

    # Handle the last segment's end, ensuring it doesn't extend beyond the total audio duration
    last_segment = filter_list[-1]
    new_end = last_segment["end"] + GRACE_PERIOD_END_S
    last_segment["end"] = min(new_end, audio_duration)

    # Second, handle the start times to avoid overlap with the *previous* segment
    # Handle the first segment's start time, ensuring it doesn't go below zero
    first_segment = filter_list[0]
    new_start = first_segment["start"] - GRACE_PERIOD_START_S
    first_segment["start"] = max(0.0, new_start)

    # Iterate from the second segment onwards
    for i in range(1, len(filter_list)):
        current_segment = filter_list[i]
        previous_segment_end = filter_list[i - 1]["end"]

        # Subtract grace period, ensuring it doesn't overlap with the previous segment
        new_start = current_segment["start"] - GRACE_PERIOD_START_S
        current_segment["start"] = max(new_start, previous_segment_end)

    return filter_list


@time_logger
def refine_vad_list_by_embedding(
    vad_list, audio, refinement_model, feature_extractor, device
):
    """
    Refines the raw VAD list by removing segments that are not internally consistent
    in their speaker embedding. This is based on the new, more reliable logic.

    Args:
        vad_list (list): The raw list of VAD segments from vad.vad().
        audio (dict): The audio data.
        refinement_model: The loaded ERes2NetV2 model for refinement.
        feature_extractor: The FBank feature extractor for the model.
        device: The torch device to run the model on.

    Returns:
        list: A new list of VAD segments after filtering out inconsistent ones.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    refined_vad_list = []
    MIN_SEGMENT_DURATION_S = 1.0  # Segments shorter than this are not processed
    WINDOW_SIZE_S = 1.1  # Window size for consistency check
    WINDOW_STEP_S = 0.4  # Step for the sliding window
    # Threshold for cosine similarity. If a window's similarity to the segment's
    # average embedding is below this, the segment is considered inconsistent.
    SIMILARITY_THRESHOLD = 0.6

    def _get_embedding(waveform_segment):
        """Helper to get embedding from a waveform segment."""
        if len(waveform_segment) / audio["sample_rate"] < 0.1:
            return None
        
        # Resample to 16k for eres2net
        waveform_16k = librosa.resample(
            waveform_segment, orig_sr=audio["sample_rate"], target_sr=16000
        )
        
        features = feature_extractor(torch.tensor(waveform_16k, dtype=torch.float32).to(device))
        with torch.no_grad():
            # The model expects a batch dimension, so we add one with .unsqueeze(0)
            embedding = refinement_model(features.unsqueeze(0)).cpu().numpy()
        return embedding

    for segment in vad_list:
        duration = segment["end"] - segment["start"]
        if duration < MIN_SEGMENT_DURATION_S:
            refined_vad_list.append(segment)
            continue

        # Extract waveform for the whole VAD segment
        start_frame_main = int(segment["start"] * audio["sample_rate"])
        end_frame_main = int(segment["end"] * audio["sample_rate"])
        segment_waveform = audio["waveform"][start_frame_main:end_frame_main]

        # 1. Get the reference (average) embedding for the whole segment
        reference_embedding = _get_embedding(segment_waveform)
        if reference_embedding is None:
            refined_vad_list.append(segment)
            continue
        
        # 2. Use a sliding window to check for internal consistency
        is_consistent = True
        window_start_s = 0
        while window_start_s + WINDOW_SIZE_S <= duration:
            window_start_frame = int(window_start_s * audio["sample_rate"])
            window_end_frame = int((window_start_s + WINDOW_SIZE_S) * audio["sample_rate"])
            window_waveform = segment_waveform[window_start_frame:window_end_frame]
            
            window_embedding = _get_embedding(window_waveform)
            if window_embedding is None:
                window_start_s += WINDOW_STEP_S
                continue

            # 3. Compare window embedding to the reference using Cosine Similarity
            similarity = cosine_similarity(reference_embedding, window_embedding)[0, 0]

            # 4. If similarity is too low, mark as inconsistent and discard
            if similarity < SIMILARITY_THRESHOLD:
                is_consistent = False
                logger.debug(
                    f"Discarding VAD segment from {segment['start']:.2f}s to {segment['end']:.2f}s "
                    f"due to internal inconsistency. Similarity: {similarity:.2f}"
                )
                

                # --- SAVE DISCARDED SEGMENT FOR DEBUGGING ---
                discarded_save_dir = os.path.join('/data/workspace/vad_segs', "discarded_for_refinement_debug")
                os.makedirs(discarded_save_dir, exist_ok=True)
                
                start_s = segment["start"]
                end_s = segment["end"]
                
                filename = f"inconsistent_dist_{similarity:.2f}_{start_s:.2f}s_to_{end_s:.2f}s.wav"
                filepath = os.path.join(discarded_save_dir, filename)

                # sf.write(filepath, segment_waveform, audio["sample_rate"])

                break
            window_start_s += WINDOW_STEP_S
        
        # 5. Only keep segments that are internally consistent
        if is_consistent:
            refined_vad_list.append(segment)

    return refined_vad_list


@time_logger
def asr(vad_segments, audio):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update vad_segments start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # resample to 16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    # --- ASR Cross-Validation Logic ---
    if cfg.get("asr_validation", {}).get("enable", False):
        logger.info("Running ASR cross-validation with Whisper and FunASR.")
        whisper_result = whisper_asr_model.transcribe(
            temp_audio, vad_segments, batch_size=batch_size, print_progress=False
        )["segments"]
        funasr_result = funasr_asr_model.transcribe(
            temp_audio, vad_segments, print_progress=False
        )["segments"]

        if len(whisper_result) != len(funasr_result):
            logger.warning("ASR models produced different number of segments. Validation failed.")
            return []

        validated_segments = []
        wer_threshold = cfg["asr_validation"].get("wer_threshold", 0.15)

        for w_seg, f_seg in zip(whisper_result, funasr_result):
            error_rate = jiwer.wer(w_seg["text"], f_seg["text"])
            if error_rate < wer_threshold:
                # Keep the result from the primary provider
                primary_seg = w_seg if cfg.get("asr_provider") == "whisper" else f_seg
                primary_seg["start"] += start_time
                primary_seg["end"] += start_time
                primary_seg["language"] = "validated"
                validated_segments.append(primary_seg)
            else:
                logger.debug(f"Segment dropped due to high WER: {error_rate:.2f}")
        return validated_segments

    # --- Standard ASR Logic ---
    if multilingual_flag and cfg.get("asr_provider", "whisper") == "whisper":
        logger.debug("Multilingual flag is on for Whisper")
        valid_vad_segments, valid_vad_segments_language = [], []
        # get valid segments to be transcripted
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            # 1. if language is in supported list, 2. if prob > 0.8
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        # if no valid segment, return empty
        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        logger.debug(f"valid_vad_segments_language: {valid_vad_segments_language}")
        unique_languages = list(set(valid_vad_segments_language))
        logger.debug(f"unique_languages: {unique_languages}")
        # process each language one by one
        for language_token in unique_languages:
            language = language_token
            # filter out segments with different language
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            # bacthed trascription
            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            result = transcribe_result_temp["segments"]
            # restore the segment annotation
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        logger.debug(
            f"Running single-language ASR for provider: {cfg.get('asr_provider', 'whisper')}"
        )
        language = None
        # For whisper, we can optionally detect language first
        if cfg.get("asr_provider", "whisper") == "whisper":
            language, prob = asr_model.detect_language(temp_audio)
            if not (language in supported_languages and prob > 0.8):
                logger.warning(
                    f"Detected language '{language}' with prob {prob:.2f} is not supported or confidence is too low."
                )
                return []

        transcribe_result = asr_model.transcribe(
            temp_audio,
            vad_segments,
            batch_size=batch_size,
            language=language,  # For Gemini/FunASR, this will be None and ignored
            print_progress=False,
        )
        result = transcribe_result["segments"]
        for idx, segment in enumerate(result):
            result[idx]["start"] += start_time
            result[idx]["end"] += start_time
            result[idx]["language"] = transcribe_result["language"]
        return result


@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        tuple: A tuple containing the average MOS and the updated VAD segments with MOS scores.
    """
    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")

    return predict_dnsmos, vad_list


def filter(mos_list, mos_filter_cfg):
    """
    Filter out segments based on a configurable MOS strategy, followed by other quality checks.

    Args:
        mos_list (list): List of VAD segments with MOS scores.
        mos_filter_cfg (dict): Configuration for MOS filtering.

    Returns:
        list: A list of VAD segments that passed all filtering stages.
    """
    # 检查输入是否为空
    if not mos_list:
        logger.warning("No segments to filter - mos_list is empty")
        return []

    # --- Step 1: Filter by MOS score based on the chosen strategy ---
    strategy = mos_filter_cfg.get("strategy", "average")
    list_after_mos_filter = []

    if strategy == "fixed":
        threshold = mos_filter_cfg.get("fixed_threshold", 3.0)
        logger.info(f"Filtering with fixed MOS threshold: >={threshold}")
        list_after_mos_filter = [seg for seg in mos_list if seg.get('dnsmos', 0) >= threshold]
    else:  # "average" strategy (default)
        if not mos_list:
            return []
        threshold = np.mean([vad["dnsmos"] for vad in mos_list])
        logger.info(f"Filtering with average MOS threshold: >={threshold:.2f}")
        list_after_mos_filter = [seg for seg in mos_list if seg.get('dnsmos', 0) >= threshold]
    
    logger.info(f"MOS Filter: {len(mos_list) - len(list_after_mos_filter)} segments removed.")

    # 如果没有任何段通过MOS过滤，提前返回
    if not list_after_mos_filter:
        logger.warning("No segments passed the MOS filtering stage.")
        return []

    # --- Step 2: Perform other quality checks (e.g., char duration) ---
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(list_after_mos_filter)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    
    if all_segment == 0:
        logger.warning("No valid segments found after secondary quality checks (calculate_audio_stats)")
        return []
    
    filter_percentage = (all_segment - filtered_segment) / all_segment
    logger.debug(
        f"> Secondary filters (char rate, etc.) removed: {all_segment - filtered_segment}/{all_segment} ({filter_percentage:.2%}) segments."
    )
    
    final_filtered_list = [list_after_mos_filter[idx] for idx, _ in filtered_audio_stats]
    
    if not final_filtered_list:
        logger.warning("All segments were filtered out by secondary quality checks.")
    
    return final_filtered_list


def main_process(audio_path, podcast_name, episode_name, save_path=None):
    """
    Process the audio file. The save_path is now the root for this specific episode.
    """
    processing_stats = {
        'initial': {'count': 0, 'duration': 0.0},
        'steps': {
            'embedding_refinement': {'discarded_count': 0, 'discarded_duration': 0.0},
            'post_process_vad': {'discarded_count': 0, 'discarded_duration': 0.0},
            'asr': {'discarded_count': 0, 'discarded_duration': 0.0},
            'mos_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
        },
        'final': {'count': 0, 'duration': 0.0}
    }

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4")):
        logger.warning(f"Unsupported file type: {audio_path}")

    # If save_path is not provided, create a default one next to the audio file.
    # Otherwise, use the provided path.
    if not save_path:
        save_path = os.path.join(os.path.dirname(audio_path) + "_processed", episode_name)

    os.makedirs(save_path, exist_ok=True)
    logger.debug(
        f"Processing audio: {episode_name}, from {audio_path}, save to: {save_path}"
    )

    logger.info(
        "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    audio = standardization(audio_path)

    logger.info("Step 1: Source Separation")
    audio = source_separation(separate_predictor1, audio)

    logger.info("Step 2: Speaker Diarization")
    diarize_df, speaker_centroids = speaker_diarization(
        dia_pipeline, audio, provider=cfg.get("diarization_provider", "pyannote")
    )

    # Rename speaker labels to be unique for the batch run
    file_hash = get_short_hash(audio_path) # Use full path for uniqueness
    speaker_mapping = {
        old_speaker: f"SPK_{file_hash}_{old_speaker.split('_')[-1]}"
        for old_speaker in diarize_df["speaker"].unique()
    }
    diarize_df["speaker"] = diarize_df["speaker"].map(speaker_mapping)
    logger.info(
        f"Renamed speaker labels for '{episode_name}' using hash '{file_hash}'. New format: SPK_{file_hash}_ID"
    )

    logger.info("Step 3: Fine-grained Segmentation by VAD")
    vad_list_initial = vad.vad(diarize_df, audio)
    processing_stats['initial']['count'] = len(vad_list_initial)
    processing_stats['initial']['duration'] = sum(s["end"] - s["start"] for s in vad_list_initial)
    
    # --- New Step 3.5: Refine VAD list by Embedding ---
    if cfg.get("embedding_refinement", {}).get("enable", True) and refinement_model:
        logger.info(
            "Step 3.5: Refining VAD list by speaker embedding for internal consistency."
        )
        vad_list_refined = refine_vad_list_by_embedding(
            vad_list_initial, audio, refinement_model, refinement_feature_extractor, device
        )
        update_stats(processing_stats, 'embedding_refinement', vad_list_initial, vad_list_refined)
    else:
        vad_list_refined = vad_list_initial

    logger.info("Step 4: Post-process VAD segments")
    audio_duration = len(audio["waveform"]) / audio["sample_rate"]
    segment_list = cut_by_speaker_label(
        vad_list_refined, audio_duration, processing_stats
    )  # post process after vad

    logger.info("Step 5: ASR")
    asr_result = asr(segment_list, audio)
    update_stats(processing_stats, 'asr', segment_list, asr_result)

    # 检查ASR结果是否为空
    if not asr_result:
        logger.warning(f"No valid speech segments found in {episode_name} - skipping MOS prediction and filtering")
        final_path = os.path.join(save_path, episode_name + ".json")
        # 创建空的结果文件
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Empty result saved to: {final_path}")
        processing_stats['final']['count'] = 0
        processing_stats['final']['duration'] = 0.0
        print_processing_summary(processing_stats, episode_name)
        return final_path, []

    logger.info("Step 6: Filter")
    logger.info("Step 6.1: calculate mos_prediction")
    avg_mos, mos_list = mos_prediction(audio, asr_result)

    logger.info(f"Step 6.1: done, average MOS: {avg_mos}")

    logger.info("Step 6.2: Filter out files with less than average MOS")
    filtered_list = filter(mos_list, cfg.get("mos_filter", {}))
    update_stats(processing_stats, 'mos_filter', mos_list, filtered_list)

    # 检查过滤结果是否为空
    if not filtered_list:
        logger.warning(f"No segments passed quality filtering for {episode_name}")
        final_path = os.path.join(save_path, episode_name + ".json")
        # 仍然创建结果文件，但内容为空
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Empty filtered result saved to: {final_path}")
        processing_stats['final']['count'] = 0
        processing_stats['final']['duration'] = 0.0
        print_processing_summary(processing_stats, episode_name)
        return final_path, []

    logger.info("Step 7: write result to file")

    output_format = cfg.get("output_format", "default")
    if output_format == "libritts":
        export_to_libritts(audio, filtered_list, save_path, episode_name)
        final_path = save_path
    else:
        export_to_mp3(audio, filtered_list, save_path, episode_name)
        final_path = os.path.join(save_path, episode_name + ".json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(filtered_list, f, ensure_ascii=False, indent=2)

    logger.info(f"All done, Saved to: {final_path}")
    
    # Final statistics update and summary print
    processing_stats['final']['count'] = len(filtered_list)
    processing_stats['final']['duration'] = sum(s["end"] - s["start"] for s in filtered_list)
    print_processing_summary(processing_stats, episode_name)

    # --- Append to CSV Report ---
    try:
        append_to_report(
            report_path="processing_report.csv",
            podcast_name=podcast_name,
            episode_name=episode_name,
            file_path=audio_path,
            initial_duration=processing_stats['initial']['duration'],
            final_duration=processing_stats['final']['duration']
        )
        logger.info(f"Appended results for '{episode_name}' to processing_report.csv")
    except Exception as e:
        logger.error(f"Failed to append to report for {episode_name}: {e}")

    return final_path, filtered_list


def update_stats(stats, step_name, list_before, list_after):
    """A helper function to calculate and update processing statistics for a given step."""
    count_before = len(list_before)
    # The list can be empty
    duration_before = sum(s["end"] - s["start"] for s in list_before) if list_before else 0
    
    count_after = len(list_after)
    duration_after = sum(s["end"] - s["start"] for s in list_after) if list_after else 0
    
    stats['steps'][step_name]['discarded_count'] = count_before - count_after
    stats['steps'][step_name]['discarded_duration'] = duration_before - duration_after


def print_processing_summary(stats, audio_name):
    """Prints a formatted summary of the audio processing statistics."""
    logger.info(f"--- Processing Summary for: {audio_name} ---")

    initial_count = stats['initial']['count']
    initial_duration = stats['initial']['duration']

    if initial_count == 0:
        logger.info("No initial segments found. Final Retention: 0 segments, 0.00s (0.00%)")
        logger.info("-------------------------------------------------")
        return

    logger.info(f"Initial: {initial_count} segments, {initial_duration:.2f}s total duration.")
    
    for step_name, data in stats['steps'].items():
        discarded_count = data['discarded_count']
        if discarded_count > 0:
            percentage_dropped = (data['discarded_duration'] / initial_duration) * 100
            logger.info(
                f" > Dropped by {step_name}: {discarded_count} segments "
                f"({data['discarded_duration']:.2f}s) - {percentage_dropped:.2f}% of initial."
            )

    final_count = stats['final']['count']
    final_duration = stats['final']['duration']
    retention_rate_count = (final_count / initial_count) * 100
    retention_rate_duration = (final_duration / initial_duration) * 100 if initial_duration > 0 else 0
    retention_rate_duration = min(100.0, retention_rate_duration)


    logger.info("-" * 20)
    logger.info(
        f"Final Retention: {final_count} / {initial_count} segments ({retention_rate_count:.2f}%)"
    )
    logger.info(
        f"Final Duration: {final_duration:.2f}s / {initial_duration:.2f}s ({retention_rate_duration:.2f}%)"
    )
    logger.info("--- End of Summary ---")

if __name__ == "__main__":
    # Use 'spawn' for CUDA safety in multiprocessing
    mp.set_start_method("spawn", force=True)

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
        "--output_folder",
        type=str,
        default="processed_data",
        help="The root folder where all processed data will be saved."
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
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    args = parser.parse_args()
    
    # --- Main Process Setup ---
    main_logger = Logger.get_logger("main")
    main_cfg = load_cfg(args.config_path)

    # --- Determine Input Source ---
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
            # Create manifest-like entries from file paths
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
        main_logger.warning(f"No audio files found to process. Exiting.")
        sys.exit(0)

    # Create the main output directory
    os.makedirs(args.output_folder, exist_ok=True)
    main_logger.info(f"Processed data will be saved in: {args.output_folder}")

    num_workers = min(args.num_workers, len(manifest_entries))
    main_logger.info(f"Found {len(manifest_entries)} audio files. Processing with {num_workers} worker(s).")

    # --- Process Pool Execution ---
    init_args = (main_cfg, args)
    
    # Use partial to pass the fixed output_folder argument to the wrapper
    process_func = partial(main_process_wrapper, output_folder=args.output_folder)

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm.tqdm(pool.imap(process_func, manifest_entries), total=len(manifest_entries)))

    main_logger.info("--- All files have been processed. ---")
    
    if args.exit_pipeline:
        main_logger.info("exit_pipeline is True, exiting...")
        sys.exit(0)
