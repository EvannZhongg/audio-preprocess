import re

import jiwer
import librosa
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from utils.logger import time_logger


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)     
    return text


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
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    cfg = PipelineParam.cfg
    asr_model = PipelineParam.asr_model
    multilingual_flag = PipelineParam.multilingual_flag
    supported_languages = PipelineParam.supported_languages
    batch_size = PipelineParam.batch_size
    validation_asr_model = PipelineParam.validation_asr_model


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
        logger.info("Running ASR cross-validation.")
        language=cfg.get("asr_validation", {}).get("language", "zh")
        asr_result = asr_model.transcribe(
            temp_audio, vad_segments, batch_size=batch_size, language=language, print_progress=False
        )["segments"]
        validation_result = validation_asr_model.transcribe(
            temp_audio, vad_segments, batch_size=batch_size, language=language, print_progress=False
        )["segments"]

        if len(asr_result) != len(validation_result):
            logger.warning("ASR models produced different number of segments. Validation failed.")
            return []

        validated_segments = []
        wer_threshold = cfg["asr_validation"].get("wer_threshold", 0.15)

        for asr_seg, val_seg, vad_seg in zip(asr_result, validation_result, vad_segments):
            error_rate = jiwer.cer(normalize_text(asr_seg["text"]),  normalize_text(val_seg["text"]))
            if error_rate <= wer_threshold:
                primary_seg = asr_seg
                primary_seg['val_text'] = val_seg['text']
                primary_seg["start"] += start_time
                primary_seg["end"] += start_time
                primary_seg["language"] = language
                primary_seg["wer"]  = error_rate
                primary_seg["norm_text"] = rich_transcription_postprocess(asr_seg["text"])
                primary_seg['min_similarity'] = vad_seg['min_similarity']
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
                result[idx]["wer"] = 0.
                result[idx]["norm_text"] = rich_transcription_postprocess(result[idx]["text"])
                result[idx]['min_similarity'] = vad_segments[idx]['min_similarity']
                
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
            result[idx]["wer"] = 0.
            result[idx]["norm_text"] = rich_transcription_postprocess(result[idx]["text"])
            result[idx]['min_similarity'] = vad_segments[idx]['min_similarity']
        
        return result
