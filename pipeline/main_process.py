import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from pipeline.alignment_filter import (compute_alignment_score,
                                       filter_by_alignment)
from pipeline.asr_process import asr
from pipeline.domain_annotation import annotate_domains
from pipeline.metrics_prediction import filter_by_metrics, metrics_prediction
from pipeline.pipeline_report import (append_to_report,
                                      print_processing_summary, update_stats)
from pipeline.source_separation import source_separation
from pipeline.speaker_diarization import speaker_diarization
from pipeline.speaking_rate import (analyze_speaking_rate,
                                    filter_by_speaking_rate)
from pipeline.silence_filter import (detect_abnormal_silence,
                                     filter_by_abnormal_silence)
from pipeline.standardization import standardization
from pipeline.text_quality_filtering import (filter_by_text_quality,
                                             text_quality_prediction)
from pipeline.vad_process import (cut_by_speaker_label,
                                  refine_vad_list_by_embedding)
from utils.meta_info_config import MetaConfig
from utils.tool import export_to_metadata, get_short_hash


# ==============================================================================
# 显存安全清理与诊断
# ==============================================================================
def safe_empty_cache(step_name, logger):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "CUDA error" in str(e):
            logger.critical(f"GPU CRITICAL FAILURE during cleanup after [{step_name}]! "
                            f"The previous step caused Illegal Memory Access.")
            raise RuntimeError(f"GPU_CRASH_AT_{step_name}") from e
        else:
            logger.warning(f"Non-critical error during cleanup after {step_name}: {e}")

# ==============================================================================
# 大文件/长文件预检查
# ==============================================================================
def file_is_too_large(audio_path, logger):
    try:
        file_size = os.path.getsize(audio_path)
        # 限制：900MB 
        MAX_FILE_SIZE_BYTES = 900 * 1024 * 1024 
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"Skipping HUGE file '{os.path.basename(audio_path)}': "
                f"{file_size / 1024 / 1024:.2f} MB > Limit."
            )
            return True
    except Exception as e:
        logger.error(f"Could not get file size for {audio_path}: {e}")
        return True 
    return False

# ==============================================================================
# 主流程
# ==============================================================================
def main_process(manifest_entry, output_folder, report_path):
    
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
            'speaking_rate_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
            'silence_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
            'alignment_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
            'metrics_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
            'text_quality_filter': {'discarded_count': 0, 'discarded_duration': 0.0},
        },
        'final': {'count': 0, 'duration': 0.0}
    }
    meta_info = MetaConfig()

    rel_path = manifest_entry["RelativePath"]
    audio_path = manifest_entry["FilePath"]
    fid = re.sub(r"['\"\s]", "", Path(audio_path).stem)
    save_path = os.path.join(output_folder, rel_path, fid)

    if file_is_too_large(audio_path, logger):
        return None, []

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4", ".ogg", ".webm")):
        logger.warning(f"Unsupported file type: {audio_path}")
        return None, []

    os.makedirs(save_path, exist_ok=True)
    logger.debug(f"Processing audio: {audio_path}, save to: {save_path}")

    # ----------------------------------------------------------------------
    # Step 0: Standardization
    # ----------------------------------------------------------------------
    logger.info("Step 0: Preprocess (Standardization)")
    audio = standardization(audio_path)
    if audio is None:
        logger.error(f"Standardization failed for: {audio_path}")
        return None, []
    
    meta_info.update_origin(raw_audio_path=audio_path)
    meta_info.update_origin(sample_rate=audio['sample_rate'])
    meta_info.update_origin(duration=round(audio['duration'], 4))

    # ----------------------------------------------------------------------
    # Step 1: Source Separation (SMRU)
    # ----------------------------------------------------------------------
    logger.info("Step 1: Source Separation")
    if cfg["separate"].get("enable", True):
        audio = source_separation(separate_predictor1, audio)
        
        if audio is not None and audio.get("waveform") is not None:
            wf = audio["waveform"]
            if np.isnan(wf).any():
                logger.error(f"💀 TOXIC DATA DETECTED: SMRU generated NaNs for {fid}. Skipping to protect GPU.")
                return None, [] # 直接跳过
            if np.isinf(wf).any():
                logger.error(f"💀 TOXIC DATA DETECTED: SMRU generated Infs for {fid}. Skipping.")
                return None, []

        safe_empty_cache("Step_1_SMRU", logger)
        logger.debug("SMRU memory cleared.")
    else:
        logger.info("Skipping source separation.")

    # ----------------------------------------------------------------------
    # Step 2: Speaker Diarization
    # ----------------------------------------------------------------------
    logger.info("Step 2: Speaker Diarization")
    diarize_df, speaker_centroids = speaker_diarization(dia_pipeline, audio, provider=cfg.get("diarization_provider", "pyannote"))
    safe_empty_cache("Step_2_Diarization", logger)
    logger.debug("Diarization memory cleared.")

    file_hash = get_short_hash(audio_path, length=8) 
    speaker_mapping = {
        old_speaker: f"SPK_{file_hash}_{old_speaker.split('_')[-1]}"
        for old_speaker in diarize_df["speaker"].unique()
    }
    diarize_df["speaker"] = diarize_df["speaker"].map(speaker_mapping)
    
    # ----------------------------------------------------------------------
    # Step 3: VAD
    # ----------------------------------------------------------------------
    logger.info("Step 3: Fine-grained Segmentation by VAD")
    vad_list_initial = vad_model.vad(diarize_df, audio)
    processing_stats['initial']['count'] = len(vad_list_initial)
    processing_stats['initial']['duration'] = sum(s["end"] - s["start"] for s in vad_list_initial)
    
    # Step 3.5: Refinement
    if cfg.get("embedding_refinement", {}).get("enable", True) and refinement_model:
        logger.info("Step 3.5: VAD Refinement")
        inter_thresh = cfg.get("strategy_parameters", {}).get("inter_similarity_threshold", 0.7)
        refine_batch = cfg.get("strategy_parameters", {}).get("refinement_batch_size", 64)
        vad_list_refined = refine_vad_list_by_embedding(vad_list_initial, audio, refinement_model, inter_thresh, refine_batch, refinement_feature_extractor, device)
        update_stats(processing_stats, 'embedding_refinement', vad_list_initial, vad_list_refined)
    else:
        vad_list_refined = vad_list_initial

    logger.info("Step 4: Post-process VAD")
    audio_dur = len(audio["waveform"]) / audio["sample_rate"]
    segment_list = cut_by_speaker_label(vad_list_refined, audio_dur, processing_stats, cfg.get("strategy_parameters", {}))

    # ----------------------------------------------------------------------
    # Step 5: ASR
    # ----------------------------------------------------------------------
    logger.info("Step 5: ASR")
    asr_result = asr(segment_list, audio)
    update_stats(processing_stats, 'asr', segment_list, asr_result)

    if not asr_result:
        logger.warning(f"No valid ASR result for {fid}")
        final_path = os.path.join(save_path, f"{fid}.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return final_path, []

    # ----------------------------------------------------------------------
    # Step 5.5: Domain Annotation (text / acoustic / speaker)
    # ----------------------------------------------------------------------
    if cfg.get("domain_annotation", {}).get("enable", False):
        logger.info("Step 5.5: Domain Annotation")
        try:
            asr_result = annotate_domains(audio, asr_result, cfg["domain_annotation"])
        except Exception as e:
            logger.warning(f"Domain annotation failed: {e}")

    # ----------------------------------------------------------------------
    # Step 5.7: Speaking Rate (per-segment) Scoring & Filter
    # ----------------------------------------------------------------------
    if cfg.get("speaking_rate", {}).get("enable", False):
        logger.info("Step 5.7: Speaking Rate Scoring & Filter")
        try:
            asr_result = analyze_speaking_rate(audio, asr_result, cfg["speaking_rate"])
            before_sr = list(asr_result)
            asr_result = filter_by_speaking_rate(asr_result, cfg["speaking_rate"])
            update_stats(processing_stats, 'speaking_rate_filter', before_sr, asr_result)
        except Exception as e:
            logger.warning(f"Speaking rate analysis failed: {e}")

        if not asr_result:
            logger.warning(f"All segments filtered out by speaking_rate for {fid}")
            final_path = os.path.join(save_path, f"{fid}.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return final_path, []

    # ----------------------------------------------------------------------
    # Step 5.75: Abnormal Silence Detection & Filter
    # ----------------------------------------------------------------------
    if cfg.get("silence_filter", {}).get("enable", False):
        logger.info("Step 5.75: Abnormal Silence Detection & Filter")
        try:
            asr_result = detect_abnormal_silence(audio, asr_result, cfg["silence_filter"])
            before_silence = list(asr_result)
            asr_result = filter_by_abnormal_silence(asr_result, cfg["silence_filter"])
            update_stats(processing_stats, 'silence_filter', before_silence, asr_result)
        except Exception as e:
            logger.warning(f"Abnormal silence detection failed: {e}")

        if not asr_result:
            logger.warning(f"All segments filtered out by abnormal_silence for {fid}")
            final_path = os.path.join(save_path, f"{fid}.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return final_path, []

    # ----------------------------------------------------------------------
    # Step 5.8: Audio-Text Alignment (forced alignment via WhisperX)
    # ----------------------------------------------------------------------
    if cfg.get("alignment", {}).get("enable", False):
        logger.info("Step 5.8: Audio-Text Alignment Scoring & Filter")
        try:
            asr_result = compute_alignment_score(audio, asr_result, cfg["alignment"])
            before_align = list(asr_result)
            asr_result = filter_by_alignment(asr_result, cfg["alignment"])
            update_stats(processing_stats, 'alignment_filter', before_align, asr_result)
        except Exception as e:
            logger.warning(f"Alignment scoring failed: {e}")

        if not asr_result:
            logger.warning(f"All segments filtered out by alignment for {fid}")
            final_path = os.path.join(save_path, f"{fid}.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return final_path, []

    # ----------------------------------------------------------------------
    # Step 6: Filter
    # ----------------------------------------------------------------------
    logger.info("Step 6: Metrics Prediction & Filter")
    avg_metrics, metrics_list = metrics_prediction(audio, asr_result, cfg.get("strategy_parameters", {}))
    logger.info(f"Avg DNSMOS: {avg_metrics[0]}")

    filtered_list = filter_by_metrics(metrics_list, cfg.get("strategy_parameters", {}))
    update_stats(processing_stats, 'metrics_filter', metrics_list, filtered_list)

    if not filtered_list:
        logger.warning(f"All segments filtered out for {fid}")
        final_path = os.path.join(save_path, f"{fid}.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return final_path, []

    # ----------------------------------------------------------------------
    # Step 6.5: Text Quality Scoring & Filter
    # ----------------------------------------------------------------------
    if cfg.get("text_quality", {}).get("enable", False):
        logger.info("Step 6.5: Text Quality Scoring & Filter")
        before_text_quality = list(filtered_list)
        filtered_list = text_quality_prediction(filtered_list, cfg["text_quality"])
        filtered_list = filter_by_text_quality(filtered_list, cfg["text_quality"])
        update_stats(processing_stats, 'text_quality_filter', before_text_quality, filtered_list)

        if not filtered_list:
            logger.warning(f"All segments filtered out by text quality for {fid}")
            final_path = os.path.join(save_path, f"{fid}.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return final_path, []

    # ----------------------------------------------------------------------
    # Step 7: Export
    # ----------------------------------------------------------------------
    logger.info("Step 7: Export")
    export_to_metadata(audio, filtered_list, save_path, meta_info, fid)
    
    # 最终统计
    processing_stats['final']['count'] = len(filtered_list)
    processing_stats['final']['duration'] = sum(s["end"] - s["start"] for s in filtered_list)
    print_processing_summary(processing_stats, fid)

    return save_path, filtered_list
