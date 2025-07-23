import logging
import re
from typing import List
import os
from filelock import FileLock

import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FunASR:
    """
    ASR class using FunASR models.
    """

    def __init__(self, model_dir: str, device: str, **kwargs):
        logger.info(f"Loading FunASR model from: {model_dir}")

        # Use a file lock to prevent race conditions during model download
        # in a multiprocessing environment.
        # The lock file is placed in the parent of the model directory to avoid
        # being part of the model files themselves.
        lock_dir = os.path.dirname(model_dir)
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{os.path.basename(model_dir)}.lock")

        with FileLock(lock_file):
            logger.debug(f"Acquired lock for FunASR model: {model_dir}")
            self.model = AutoModel(
                model=model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=device,
                **kwargs,
            )
        logger.debug(f"Released lock for FunASR model: {model_dir}")

        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\u2600-\u26FF"  # miscellaneous symbols
            "\u2700-\u27BF"  # dingbats
            "]+",
            flags=re.UNICODE,
        )

    def detect_language(self, audio: np.ndarray):
        """
        FunASR can auto-detect language during transcription.
        This function is for API compatibility with the pipeline.
        """
        logger.debug("FunASR does not require separate language detection.")
        return None, 0.0

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        print_progress=False,
        **kwargs,
    ):
        """
        Transcribe audio segments using a FunASR model.

        Args:
            audio (np.ndarray): The audio waveform (should be 16kHz).
            vad_segments (List[dict]): A list of dictionaries with 'start' and 'end' times.
            print_progress (bool): Whether to show a progress bar.

        Returns:
            dict: A dictionary containing the list of transcribed segments.
        """
        segments = []
        
        progress = tqdm(total=len(vad_segments), desc="FunASR Transcribing", disable=not print_progress)

        for segment_info in vad_segments:
            start_frame = int(segment_info["start"] * 16000)
            end_frame = int(segment_info["end"] * 16000)
            segment_audio = audio[start_frame:end_frame]

            if len(segment_audio) == 0:
                progress.update(1)
                continue

            try:
                # FunASR generate can handle numpy array directly
                res = self.model.generate(
                    input=segment_audio,
                    language="auto",
                    use_itn=True,
                )
                raw_text = res[0]["text"] if res and "text" in res[0] else ""
                text = rich_transcription_postprocess(raw_text)
                text = self.emoji_pattern.sub(r"", text).strip()

                segments.append(
                    {
                        "text": text,
                        "start": round(segment_info["start"], 3),
                        "end": round(segment_info["end"], 3),
                        "speaker": segment_info.get("speaker", None),
                    }
                )
            except Exception as e:
                logger.error(f"Error transcribing segment with FunASR: {e}")
                segments.append(
                    {
                        "text": "",
                        "start": round(segment_info["start"], 3),
                        "end": round(segment_info["end"], 3),
                        "speaker": segment_info.get("speaker", None),
                    }
                )
            finally:
                progress.update(1)
        
        progress.close()

        # FunASR's generate method can auto-detect language for each segment,
        # but the pipeline expects a single language for the whole result.
        return {"segments": segments, "language": "unknown"}


def load_asr_model(model_dir: str, device: str, **kwargs):
    """
    Load the FunASR model.
    This function acts as a factory for the FunASR class.
    """
    return FunASR(model_dir=model_dir, device=device, **kwargs) 
