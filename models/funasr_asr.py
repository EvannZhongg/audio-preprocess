import logging
import os
import re
import shutil
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from filelock import FileLock
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FunASR:
    """
    ASR class using FunASR models.
    Supports SenseVoice (direct numpy input) and ParaFormer (file-based or list input).
    """

    def __init__(
        self,
        asr_model: str,
        model_dir: str,
        vad_model_dir: str,
        device: str,
        punc_model_dir: str = 'ct-punc-c',
        **kwargs
    ):
        logger.info(f"Loading FunASR model from: {model_dir}")

        lock_dir = os.path.dirname(model_dir)
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{os.path.basename(model_dir)}.lock")

        with FileLock(lock_file):
            logger.debug(f"Acquired lock for FunASR model: {model_dir}")
            if asr_model == "SenseVoice":
                self.model = AutoModel(
                    model=model_dir,
                    vad_model=vad_model_dir,
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=device,
                    disable_update=True,
                    **kwargs,
                )
            elif asr_model == "ParaFormer":
                self.model = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model=model_dir,
                    punc_model=punc_model_dir,
                    device=device,
                    disable_update=True,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported FunASR model: {asr_model}")
        self.asr_model = asr_model
        logger.debug(f"Released lock for FunASR model: {model_dir}")

        # Precompile emoji regex for efficiency
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\u2600-\u26FF"          # miscellaneous symbols
            "\u2700-\u27BF"          # dingbats
            "]+",
            flags=re.UNICODE,
        )

    def detect_language(self, audio: np.ndarray):
        """Placeholder for compatibility; FunASR handles language internally."""
        logger.debug("FunASR does not require separate language detection.")
        return None, 0.0

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        sample_rate: int = 16000,
        print_progress: bool = False,
        **kwargs,
    ) -> dict:
        """
        Transcribe audio segments using a FunASR model.

        Args:
            audio (np.ndarray): The audio waveform.
            vad_segments (List[dict]): List of dicts with 'start' and 'end' in seconds.
            sample_rate (int): Audio sample rate (default: 16000).
            print_progress (bool): Show progress bar.

        Returns:
            dict: {"segments": [...], "language": "unknown"}
        """
        if len(vad_segments) == 0:
            return {"segments": [], "language": "unknown"}

        segments = []
        progress = tqdm(total=len(vad_segments), desc="FunASR Transcribing", disable=not print_progress)

        try:
            if self.asr_model == "ParaFormer":
                # Batch mode: write all segments to temp files and transcribe as list
                with tempfile.TemporaryDirectory() as temp_dir:
                    audio_paths = []
                    for i, seg in enumerate(vad_segments):
                        start_frame = int(seg["start"] * sample_rate)
                        end_frame = int(seg["end"] * sample_rate)
                        segment_audio = audio[start_frame:end_frame]

                        if len(segment_audio) == 0:
                            audio_paths.append(None)  # placeholder
                            continue

                        temp_path = os.path.join(temp_dir, f"seg_{i}.wav")
                        sf.write(temp_path, segment_audio, sample_rate)
                        audio_paths.append(temp_path)

                    # Filter out empty segments for batch inference
                    valid_paths = [p for p in audio_paths if p is not None]
                    if valid_paths:
                        try:
                            batch_results = self.model(valid_paths)
                            # Ensure consistent format: wrap single result in list
                            if isinstance(batch_results, dict):
                                batch_results = [batch_results]
                        except Exception as e:
                            logger.error(f"ParaFormer batch transcription failed: {e}")
                            batch_results = [{"text": ""} for _ in valid_paths]
                    else:
                        batch_results = []

                    # Map results back to original segments
                    result_iter = iter(batch_results)
                    for i, seg in enumerate(vad_segments):
                        if audio_paths[i] is None:
                            text = ""
                        else:
                            res = next(result_iter)
                            raw_text = res.get("text", "") if isinstance(res, dict) else str(res)
                            text = rich_transcription_postprocess(raw_text)
                            text = self.emoji_pattern.sub("", text).strip()

                        segments.append({
                            "text": text,
                            "start": round(seg["start"], 3),
                            "end": round(seg["end"], 3),
                            "speaker": seg.get("speaker"),
                        })
                        progress.update(1)

            else:  # SenseVoice
                for seg in vad_segments:
                    start_frame = int(seg["start"] * sample_rate)
                    end_frame = int(seg["end"] * sample_rate)
                    segment_audio = audio[start_frame:end_frame]

                    if len(segment_audio) == 0:
                        text = ""
                    else:
                        try:
                            res = self.model.generate(
                                input=segment_audio,
                                language="auto",
                                use_itn=True,
                            )
                            raw_text = res[0]["text"] if res and isinstance(res, list) and "text" in res[0] else ""
                            text = rich_transcription_postprocess(raw_text)
                            text = self.emoji_pattern.sub("", text).strip()
                        except Exception as e:
                            logger.error(f"Error transcribing segment with SenseVoice: {e}")
                            text = ""

                    segments.append({
                        "text": text,
                        "start": round(seg["start"], 3),
                        "end": round(seg["end"], 3),
                        "speaker": seg.get("speaker"),
                    })
                    progress.update(1)

        finally:
            progress.close()

        return {"segments": segments, "language": "unknown"}


def load_asr_model(
    asr_model: str,
    model_dir: str,
    vad_model_dir: str,
    device: str,
    punc_model_dir: str = 'ct-punc-c',
    **kwargs
):
    """Factory function to load FunASR model."""
    return FunASR(
        asr_model=asr_model,
        model_dir=model_dir,
        vad_model_dir=vad_model_dir,
        punc_model_dir=punc_model_dir,
        device=device,
        **kwargs
    )
