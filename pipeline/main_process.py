import json
import os
import re
import threading
from pathlib import Path

from pipeline.asr_process import asr
from pipeline.metrics_prediction import filter_by_metrics, metrics_prediction
from pipeline.pipeline_report import (append_to_report,
                                      print_processing_summary, update_stats)
from pipeline.source_separation import source_separation
from pipeline.speaker_diarization import speaker_diarization
from pipeline.standardization import standardization
from pipeline.vad_process import (cut_by_speaker_label,
                                  refine_vad_list_by_embedding)
from utils.meta_info_config import MetaConfig
from utils.tool import export_to_metadata, get_short_hash

_vad_lock = threading.Lock()
_dia_lock = threading.Lock()
_sep_lock = threading.Lock()

def file_is_large(audio_path):
    try:
        file_size = os.path.getsize(audio_path)
        MAX_FILE_SIZE_BYTES = 900 * 1024 * 1024
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"Skipping file '{os.path.basename(audio_path)}' because its size "
                f"({file_size / 1024 / 1024:.2f} MB) exceeds the limit of {MAX_FILE_SIZE_BYTES / 1024 / 1024:.2f} MB."
            )
            return True
    except Exception as e:
        logger.error(f"Could not get file size for {audio_path}: {e}")
        return True
    
    return False
    

def main_process(manifest_entry, output_folder, report_path):
    """
    Process the audio file..
    """
    
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger
    
    cfg = PipelineParam.cfg
    device = PipelineParam.device
    separate_predictor1 = PipelineParam.separate_predictor1
    dia_pipeline = PipelineParam.dia_pipeline
    vad_model = PipelineParam.vad_model
    refinement_model = PipelineParam.refinement_model
    refinement_feature_extractor = PipelineParam.refinement_feature_extractor

    processing_stats = {
        'initial': {'count': 0, 'duration': 0.0},
        'steps': {
            'embedding_refinement': {'discarded_count': 0, 'discarded_duration': 0.0},
            'post_process_vad': {'discarded_count': 0, 'discarded_duration': 0.0},
            'asr': {'discarded_count': 0, 'discarded_duration': 0.0},
            'metrics_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
        },
        'final': {'count': 0, 'duration': 0.0}
    }

    meta_info = MetaConfig()

    rel_path = manifest_entry["RelativePath"]
    audio_path = manifest_entry["FilePath"]
    fid = re.sub(r"['\"\s]", "", Path(audio_path).stem)
    save_path = os.path.join(output_folder, rel_path, fid)

    if file_is_large(audio_path):
        return

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4")):
        logger.warning(f"Unsupported file type: {audio_path}")


    os.makedirs(save_path, exist_ok=True)
    logger.debug(f"Processing audio: {audio_path}, save to: {save_path}")

    logger.info("Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16")
    audio = standardization(audio_path)
    if audio is None:
        logger.error(f"Standardization failed (Skipped/Timeout/Error) for: {audio_path}")
        return None, []
    
    meta_info.update_origin(raw_audio_path=audio_path)
    meta_info.update_origin(sample_rate=audio['sample_rate'])
    meta_info.update_origin(duration=round(audio['duration'], 4))

    logger.info("Step 1: Source Separation")
    # Add a check in config to decide whether to run this step
    if cfg["separate"].get("enable", True):
        with _sep_lock:
            audio = source_separation(separate_predictor1, audio)
    else:
        logger.info("Skipping source separation as per config.")

    with _dia_lock:
        logger.info("Step 2: Speaker Diarization")
        diarize_df, speaker_centroids = speaker_diarization(dia_pipeline, audio, provider=cfg.get("diarization_provider", "pyannote"))

    # Rename speaker labels to be unique for the batch run
    file_hash = get_short_hash(audio_path, length=8) # Use full path for uniqueness
    speaker_mapping = {
        old_speaker: f"SPK_{file_hash}_{old_speaker.split('_')[-1]}"
        for old_speaker in diarize_df["speaker"].unique()
    }
    diarize_df["speaker"] = diarize_df["speaker"].map(speaker_mapping)
    logger.info(f"Renamed speaker labels for '{audio_path}' using hash '{file_hash}'. New format: SPK_{file_hash}_ID")
    
    with _vad_lock:
        logger.info("Step 3: Fine-grained Segmentation by VAD")
        vad_list_initial = vad_model.vad(diarize_df, audio)
        
    processing_stats['initial']['count'] = len(vad_list_initial)
    processing_stats['initial']['duration'] = sum(s["end"] - s["start"] for s in vad_list_initial)
    
    # --- New Step 3.5: Refine VAD list by Embedding ---
    if cfg.get("embedding_refinement", {}).get("enable", True) and refinement_model:
        logger.info("Step 3.5: Refining VAD list by speaker embedding for internal consistency.")
        inter_similarity_threshold = cfg.get("strategy_parameters", {}).get("inter_similarity_threshold", 0.7)
        refinement_batch_size = cfg.get("strategy_parameters", {}).get("refinement_batch_size", 64)
        vad_list_refined = refine_vad_list_by_embedding(vad_list_initial, audio, refinement_model, inter_similarity_threshold, refinement_batch_size, refinement_feature_extractor, device)
        update_stats(processing_stats, 'embedding_refinement', vad_list_initial, vad_list_refined)
    else:
        vad_list_refined = vad_list_initial

    logger.info("Step 4: Post-process VAD segments")
    audio_duration = len(audio["waveform"]) / audio["sample_rate"]
    segment_list = cut_by_speaker_label(vad_list_refined, audio_duration, processing_stats, cfg.get("strategy_parameters", {}))

    logger.info("Step 5: ASR")
    asr_result = asr(segment_list, audio)
    update_stats(processing_stats, 'asr', segment_list, asr_result)

    # 检查ASR结果是否为空
    if not asr_result:
        logger.warning(f"No valid speech segments found in {audio_path} - skipping MOS prediction and filtering")
        final_path = os.path.join(save_path, f"{fid}.json")
        # 创建空的结果文件
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Empty result saved to: {final_path}")
        processing_stats['final']['count'] = 0
        processing_stats['final']['duration'] = 0.0
        print_processing_summary(processing_stats, fid)
        return final_path, []

    logger.info("Step 6: Filter")
    logger.info("Step 6.1: calculate metrics_prediction")
    avg_metrics, metrics_list = metrics_prediction(audio, asr_result, cfg.get("strategy_parameters", {}))

    logger.info(f"Step 6.1: done, average dnsmos: {avg_metrics[0]}")

    logger.info("Step 6.2: Filter out files with less than average metrics")
    filtered_list = filter_by_metrics(metrics_list, cfg.get("strategy_parameters", {}))
    update_stats(processing_stats, 'metrics_filter', metrics_list, filtered_list)

    # 检查过滤结果是否为空
    if not filtered_list:
        logger.warning(f"No segments passed quality filtering for {audio_path}")
        final_path = os.path.join(save_path, f"{fid}.json")
        # 仍然创建结果文件，但内容为空
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info(f"Empty filtered result saved to: {final_path}")
        processing_stats['final']['count'] = 0
        processing_stats['final']['duration'] = 0.0
        print_processing_summary(processing_stats, fid)
        return final_path, []

    logger.info("Step 7: write result to file")

    export_to_metadata(audio, filtered_list, save_path, meta_info, fid)
    final_path = save_path 

    logger.info(f"All done, Saved to: {final_path}")
    
    # Final statistics update and summary print
    processing_stats['final']['count'] = len(filtered_list)
    processing_stats['final']['duration'] = sum(s["end"] - s["start"] for s in filtered_list)
    print_processing_summary(processing_stats, fid)

    # --- Append to CSV Report ---
    try:
        # append_to_report(
        #     report_path=report_path,
        #     rel_path=rel_path,
        #     file_path=audio_path,
        #     initial_duration=processing_stats['initial']['duration'],
        #     final_duration=processing_stats['final']['duration']
        # )
        # logger.info(f"Appended results for '{audio_path}'")
        pass
    except Exception as e:
        logger.error(f"Failed to append to report for {audio_path}: {e}")

    return final_path, filtered_list
