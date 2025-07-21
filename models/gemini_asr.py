import logging
import os
import re
import tempfile
import time
from typing import List

import google.generativeai as genai
import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GeminiASR:
    """
    ASR class using Google's Gemini API.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        if not api_key or not api_key.startswith("AI"):
            raise ValueError(
                "Invalid Gemini API key. Please check your config.json and make sure it's correct."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.prompt = (
            "请将音频内容转录成文字。请注意，你输出的应该是纯净的、逐字稿的文本，"
            "并使用符合中文书写习惯的标点符号。不要包含任何介绍性文字、时间戳、说话人标签或声音事件标签（例如 [音乐] 或 [嘶嘶声]）。"
            "最终输出应为不含换行符的单行文本。"
        )

    def detect_language(self, audio: np.ndarray):
        """
        Gemini API for audio does not have a separate language detection endpoint.
        It's auto-detected during transcription. This function is for API compatibility.
        """
        logger.debug("Gemini ASR does not support separate language detection.")
        return None, 0.0

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        batch_size=None,  # For compatibility, not used
        language=None,  # For compatibility, not used
        print_progress=False,
        **kwargs,
    ):
        """
        Transcribe audio segments using Gemini API.

        Args:
            audio (np.ndarray): The audio waveform.
            vad_segments (List[dict]): A list of dictionaries with 'start' and 'end' times for segments.
            print_progress (bool): Whether to show a progress bar.

        Returns:
            dict: A dictionary containing the list of transcribed segments and the language.
        """
        sample_rate = 16000  # ASR models in this pipeline expect 16k sample rate
        segments = []
        
        progress = tqdm(total=len(vad_segments), desc="Gemini Transcribing", disable=not print_progress)

        for segment_info in vad_segments:
            start_frame = int(segment_info["start"] * sample_rate)
            end_frame = int(segment_info["end"] * sample_rate)
            segment_audio = audio[start_frame:end_frame]

            if len(segment_audio) == 0:
                progress.update(1)
                continue

            # Use a temporary file to upload to Gemini API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, segment_audio, sample_rate)
                temp_filename = tmpfile.name

            try:
                uploaded_file = genai.upload_file(path=temp_filename)

                # The new API for gemini-1.5-flash can take file directly.
                response = self.model.generate_content(
                    [self.prompt, uploaded_file], request_options={"timeout": 120}
                )
                
                genai.delete_file(uploaded_file.name)

                text = response.text if hasattr(response, "text") else ""

                # Clean up the transcribed text
                # 1. Remove sound event tags like [hiss] or [music]
                text = re.sub(r'\[.*?\]', '', text)
                # 2. Remove any kind of timestamps that might be included
                text = re.sub(r'\d{2,}:\d{2,}\s?', '', text)
                # 3. Remove newlines and carriage returns, replacing them with a space
                text = text.replace('\n', ' ').replace('\r', ' ')
                # 4. Collapse multiple spaces into a single space and strip
                text = re.sub(r'\s+', ' ', text).strip()

                segments.append(
                    {
                        "text": text,
                        "start": round(segment_info["start"], 3),
                        "end": round(segment_info["end"], 3),
                        "speaker": segment_info.get("speaker", None),
                    }
                )
            except Exception as e:
                logger.error(f"Error transcribing segment with Gemini: {e}")
                # In case of error, we add an empty text segment to avoid breaking the pipeline
                segments.append(
                    {
                        "text": "",
                        "start": round(segment_info["start"], 3),
                        "end": round(segment_info["end"], 3),
                        "speaker": segment_info.get("speaker", None),
                    }
                )
            finally:
                os.remove(temp_filename)
                progress.update(1)
        
        progress.close()

        return {"segments": segments, "language": "unknown"}


def load_asr_model(api_key: str, model_name: str, **kwargs):
    """
    Load the Gemini ASR model.
    This function acts as a factory for GeminiASR class.
    """
    logger.info(f"Loading Gemini ASR model: {model_name}")
    return GeminiASR(api_key=api_key, model_name=model_name) 
