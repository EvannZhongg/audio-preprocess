import logging
import os
import re
import shutil
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch
from filelock import FileLock
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


logger = logging.getLogger(__name__)

class FunASR:
    """
    ASR class using FunASR models (optimized version).
    Fixed critical issues: punctuation model support, redundant methods, empty segment handling.
    Now supports unified batch recognition for both SenseVoice and ParaFormer.
    """

    def __init__(self, asr_model: str, model_dir: str, vad_model_dir: str, device: str, punc_model_dir: str = 'ct-punc-c', **kwargs):
        logger.info(f"Loading FunASR model from: {model_dir}")
        lock_dir = os.path.dirname(model_dir)
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{os.path.basename(model_dir)}.lock")

        self.device = device
        self.asr_model = asr_model

        with FileLock(lock_file):
            logger.debug(f"Acquired lock for FunASR model: {model_dir}")
            if asr_model == "SenseVoice":
                self.model = AutoModel(
                    model=model_dir,
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=device,
                    disable_update=True,
                    **kwargs,
                )
            elif asr_model == "ParaFormer":
                self.model = pipeline(
                    task=Tasks.auto_speech_recognition,
                    batch_size=kwargs.get("batch_size", 64),
                    model=model_dir,
                    punc_model=punc_model_dir, 
                    device=device,
                    disable_update=True
                )
            else:
                raise ValueError(f"Unsupported FunASR model: {asr_model}")
        logger.debug(f"Released lock for FunASR model: {model_dir}")

        # Emoji removal regex (kept for text cleanup)
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
        """API compatibility - no separate language detection needed."""
        logger.debug("FunASR auto-detects language during transcription.")
        return None, 0.0

    def batch_recognize_via_tempfile(self, audio: np.ndarray, vad_segments: List[dict], sample_rate: int = 16000):
        """
        Batch transcribe audio segments using temporary files.
        """
        if not vad_segments:
            return [""] * len(vad_segments)
        
        # 记录非空片段的原始索引，用于结果对齐
        non_empty_segment_indices = [] 
        input_file_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. 创建所有音频片段的临时文件
                for idx, segment_info in enumerate(vad_segments):
                    start_frame = int(segment_info["start"] * sample_rate)
                    end_frame = int(segment_info["end"] * sample_rate)
                    segment_audio = audio[start_frame:end_frame]
                    
                    if len(segment_audio) == 0:
                        continue
                            
                    # 使用索引作为文件名
                    temp_audio_file_path = os.path.join(temp_dir, f"{idx}.wav")
                    sf.write(temp_audio_file_path, segment_audio, sample_rate)
                    
                    non_empty_segment_indices.append(idx)
                    input_file_paths.append(temp_audio_file_path)

                if not input_file_paths:
                    return [""] * len(vad_segments)

                # 2. 根据模型类型执行批量识别
                if self.asr_model == "ParaFormer":
                    # ParaFormer (ModelScope pipeline) 需要 filelist.scp 文件
                    temp_file_path = os.path.join(temp_dir, "filelist.scp")
                    with open(temp_file_path, 'w') as f:
                        for path in input_file_paths:
                            f.write(f"{path}\n")
                    result = self.model(input=temp_file_path)
                
                elif self.asr_model == "SenseVoice":
                    # SenseVoice (AutoModel) 可以直接传入文件路径列表
                    result = self.model.generate(
                        input=input_file_paths,
                        language="auto",
                        use_itn=True,
                    )
                else:
                    raise ValueError(f"Unsupported FunASR model: {self.asr_model}")

                # 3. 后处理结果并与原始片段列表对齐
                if isinstance(result, dict):
                    result = [result] 
                
                full_text_results = [""] * len(vad_segments)
                
                for res, original_idx in zip(result, non_empty_segment_indices):
                    # 获取原始文本
                    if isinstance(res, dict) and "text" in res:
                        raw_text = res["text"]
                    elif isinstance(res, list) and len(res) > 0 and "text" in res[0]:
                         raw_text = res[0]["text"]
                    else:
                        raw_text = ""

                    # 应用 FunASR 的后处理
                    processed_text = rich_transcription_postprocess(raw_text).strip()
                    full_text_results[original_idx] = processed_text

                return full_text_results

            finally:
                pass 

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        print_progress: bool = False,
        **kwargs
    ) -> dict:
        """
        Transcribe audio segments with optimized handling using unified batch processing.
        Returns: {"segments": [{"text": "...", "start": 0.0, "end": 1.5, ...}]}
        """
        if not vad_segments: 
            return {"segments": [], "language": "unknown"}
        
        batch_result = self.batch_recognize_via_tempfile(audio, vad_segments)
        
        segments = []
        for res, segment_info in zip(batch_result, vad_segments):
            text = self.emoji_pattern.sub(r"", res).strip()
            
            # 仅记录非空文本的片段，或者保留所有片段 (取决于具体需求，此处保留所有片段信息)
            segments.append({
                "text": text,
                "start": round(segment_info["start"], 3),
                "end": round(segment_info["end"], 3),
                "speaker": segment_info.get("speaker", None),
            })
        
        return {"segments": segments, "language": "unknown"}


def load_asr_model(
    asr_model: str, 
    model_dir: str, 
    vad_model_dir: str, 
    device: str, 
    punc_model_dir: str = 'ct-punc-c', 
    **kwargs
):
    """Factory function for FunASR model loading."""
    return FunASR(
        asr_model=asr_model, 
        model_dir=model_dir, 
        vad_model_dir=vad_model_dir,
        punc_model_dir=punc_model_dir,
        device=device,
        **kwargs
    )


def __repr__(self):
    return f"FunASR(model={self.asr_model}, device={self.device})"
