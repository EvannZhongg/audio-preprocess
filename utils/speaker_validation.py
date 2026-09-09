import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import google.generativeai as genai
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def worker_validate_speaker(segment, audio_waveform, sample_rate, gemini_model):
    """Worker function to validate a single speaker in a segment using Gemini."""
    prompt = (
        "分析提供的音频片段。此音频中是否包含来自一个以上的人的语音，例如重叠的讲话或来自另一个人的背景应和？"
        "请只回答“是”或“否”。"
    )
    start_frame = int(segment["start"] * sample_rate)
    end_frame = int(segment["end"] * sample_rate)
    segment_audio = audio_waveform[start_frame:end_frame]

    if len(segment_audio) == 0:
        return None  # Invalid segment

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            sf.write(tmpfile.name, segment_audio, sample_rate)
            audio_file = genai.upload_file(path=tmpfile.name)

        response = gemini_model.generate_content(
            [prompt, audio_file], request_options={"timeout": 60}
        )
        genai.delete_file(audio_file.name)

        # If model says 'yes' (multiple speakers), we discard it by returning None.
        if response.text and "是" in response.text:
            return None
        # Otherwise (response is 'no' or something else), we keep it.
        return segment

    except Exception as e:
        logger.error(f"Error during single speaker validation: {e}")
        return segment  # Keep segment in case of API error to be safe


def validate_single_speaker(
    segments: List[Dict],
    audio_waveform,
    sample_rate: int,
    gemini_model,
    max_workers: int,
) -> List[Dict]:
    """
    Uses Gemini to validate that each segment contains only one speaker.
    """
    validated_segments = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                worker_validate_speaker,
                segment,
                audio_waveform,
                sample_rate,
                gemini_model,
            ): segment
            for segment in segments
        }
        for future in tqdm(
            as_completed(future_to_segment),
            total=len(segments),
            desc="Validating Single Speaker",
        ):
            result = future.result()
            if result:
                validated_segments.append(result)

    return validated_segments 
