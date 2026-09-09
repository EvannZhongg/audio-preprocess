import gc
import logging
import os
import re
import shutil
import tempfile
from typing import List, Union

import numpy as np
import soundfile as sf
import torch
from filelock import FileLock
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logger = logging.getLogger(__name__)

# Fun-ASR-MLT-Nano-2512's `generate(language=...)` does NOT take an ISO code
# or "auto" (its official usage example only ever shows a literal Chinese
# language name, e.g. language="中文"). Passing "auto" is undocumented and,
# for lower-resource languages in its 31-language set (e.g. Vietnamese), it
# has been observed to silently collapse to Chinese/Cantonese decoding
# instead of actually auto-detecting. So for this model we must map our ISO
# code to the name it expects; "auto" is kept only as a last-resort fallback
# for languages not in this table (behavior unchanged from before).
FUNASR_NANO_LANGUAGE_MAP = {
    "zh": "中文",
    "en": "英文",
    "yue": "粤语",
    "ja": "日文",
    "ko": "韩文",
    "vi": "越南语",
    "id": "印尼语",
    "th": "泰语",
    "ms": "马来语",
    "tl": "菲律宾语",
    "fil": "菲律宾语",
    "ar": "阿拉伯语",
    "hi": "印地语",
    "bg": "保加利亚语",
    "hr": "克罗地亚语",
    "cs": "捷克语",
    "da": "丹麦语",
    "nl": "荷兰语",
    "et": "爱沙尼亚语",
    "fi": "芬兰语",
    "el": "希腊语",
    "hu": "匈牙利语",
    "ga": "爱尔兰语",
    "lv": "拉脱维亚语",
    "lt": "立陶宛语",
    "mt": "马耳他语",
    "pl": "波兰语",
    "pt": "葡萄牙语",
    "ro": "罗马尼亚语",
    "sk": "斯洛伐克语",
    "sl": "斯洛文尼亚语",
    "sv": "瑞典语",
}


class FunASR:
    """
    ASR class using FunASR models (Optimized for Memory Stability).
    Fixed: GPU Memory Leak via manual GC and Mini-batching.
    Fixed: System Error due to /tmp full by allowing custom temp directory.
    """

    def __init__(self, asr_model: str, model_dir: str, vad_model_dir: str, device: str, punc_model_dir: str = 'ct-punc-c', **kwargs):
        logger.info(f"Loading FunASR model from: {model_dir}")
        # Lock key must NOT depend on which form of `model_dir` the caller
        # resolved (absolute cache path vs. bare hub id like "iic/xxx"),
        # otherwise concurrent actors on the same machine can pick different
        # lock files for the *same* underlying model: one process may be
        # mid-download (cache dir just created but incomplete) while another
        # process's os.path.exists() check flips to True and it loads the
        # half-written directory directly -> "not registered" errors on
        # fresh machines. Using a fixed, machine-local lock dir + the model's
        # basename ensures all callers contend for the same lock regardless
        # of which branch (cache-hit vs. hub-id) they took.
        lock_dir = os.path.join(tempfile.gettempdir(), "funasr_model_locks")
        os.makedirs(lock_dir, exist_ok=True)
        lock_key = os.path.basename(model_dir.rstrip("/")) or model_dir.replace("/", "_")
        lock_file = os.path.join(lock_dir, f"{lock_key}.lock")

        self.device = device
        self.asr_model = asr_model
        self.inference_batch_size = kwargs.get("batch_size", 16) 

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
            elif asr_model == "FunASRNano":
                self.model = AutoModel(
                    model=model_dir,
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=device,
                    trust_remote_code=True,
                    disable_update=True,
                    **kwargs,
                )
            elif asr_model == "ParaFormer":
                self.model = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model=model_dir,
                    punc_model=punc_model_dir,
                    device=device,
                    disable_update=True
                )
            else:
                raise ValueError(f"Unsupported FunASR model: {asr_model}")
        logger.debug(f"Released lock for FunASR model: {model_dir}")

        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]+",
            flags=re.UNICODE,
        )

    def _clear_memory(self):
        """强制清理显存和内存碎片"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def detect_language(self, audio: np.ndarray):
        return None, 0.0

    def batch_recognize_via_tempfile(self, audio: np.ndarray, vad_segments: List[dict], sample_rate: int = 16000, language: str = "auto"):
        """
        Batch transcribe with explicit chunking and memory cleanup.
        Uses LARGE_TEMP_DIR env var if set to avoid /tmp overflow.
        """
        if not vad_segments:
            return [""] * len(vad_segments)
        
        non_empty_segment_indices = [] 
        input_file_paths = []

        # 优先使用环境变量指定的临时目录
        custom_temp_dir = os.environ.get("LARGE_TEMP_DIR", None)
        if custom_temp_dir:
            try:
                os.makedirs(custom_temp_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create LARGE_TEMP_DIR '{custom_temp_dir}', falling back to system default: {e}")
                custom_temp_dir = None

        with tempfile.TemporaryDirectory(dir=custom_temp_dir) as temp_dir:
            for idx, segment_info in enumerate(vad_segments):
                start_frame = int(segment_info["start"] * sample_rate)
                end_frame = int(segment_info["end"] * sample_rate)
                segment_audio = audio[start_frame:end_frame]
                
                if len(segment_audio) == 0:
                    continue
                        
                temp_audio_file_path = os.path.join(temp_dir, f"{idx}.wav")
                try:
                    sf.write(temp_audio_file_path, segment_audio, sample_rate)
                except Exception as e:
                    logger.critical(f"Failed to write temp audio file to {temp_audio_file_path}. Disk full or Inode exhausted? Error: {e}")
                    raise e
                
                non_empty_segment_indices.append(idx)
                input_file_paths.append(temp_audio_file_path)

            if not input_file_paths:
                return [""] * len(vad_segments)

            all_inference_results = []
            batch_size = self.inference_batch_size
            total_files = len(input_file_paths)
            
            for i in range(0, total_files, batch_size):
                batch_paths = input_file_paths[i : i + batch_size]
                batch_results_chunk = []
                
                try:
                    with torch.no_grad():
                        if self.asr_model == "ParaFormer":
                            batch_scp_path = os.path.join(temp_dir, f"filelist_batch_{i}.scp")
                            with open(batch_scp_path, 'w') as f:
                                for path in batch_paths:
                                    f.write(f"{path}\n")
                            
                            res = self.model(input=batch_scp_path, batch_size=batch_size)
                            batch_results_chunk = res if isinstance(res, list) else [res]
                            
                        elif self.asr_model == "SenseVoice":
                            # SenseVoice's official API documents "auto" as a
                            # valid value (it truly auto-detects among its
                            # zh/en/ja/ko/yue support set), so no mapping
                            # needed here.
                            res = self.model.generate(
                                input=batch_paths,
                                language="auto",
                                use_itn=True,
                                batch_size_s=0, # Disable internal dynamic batching to control strictly
                                batch_size=len(batch_paths)
                            )
                            batch_results_chunk = res if isinstance(res, list) else [res]

                        elif self.asr_model == "FunASRNano":
                            # Map our ISO code to the Chinese language name
                            # this model's `generate()` actually expects; see
                            # FUNASR_NANO_LANGUAGE_MAP for why "auto" is unsafe.
                            nano_language = FUNASR_NANO_LANGUAGE_MAP.get(language, "auto")
                            res = self.model.generate(
                                input=batch_paths,
                                language=nano_language,
                                itn=True,
                                batch_size_s=0,
                                batch_size=len(batch_paths)
                            )
                            batch_results_chunk = res if isinstance(res, list) else [res]
                    
                    all_inference_results.extend(batch_results_chunk)

                except Exception as e:
                    logger.error(f"Error during batch inference at index {i}: {e}")
                    all_inference_results.extend([{"text": ""}] * len(batch_paths))
                
                finally:
                    del batch_results_chunk
                    if 'res' in locals(): del res
                    self._clear_memory()

            full_text_results = [""] * len(vad_segments)
            limit = min(len(all_inference_results), len(non_empty_segment_indices))
            
            for i in range(limit):
                res = all_inference_results[i]
                original_idx = non_empty_segment_indices[i]
                
                raw_text = ""
                if isinstance(res, dict) and "text" in res:
                    raw_text = res["text"]
                elif isinstance(res, list) and len(res) > 0 and "text" in res[0]:
                    raw_text = res[0]["text"]
                
                processed_text = rich_transcription_postprocess(raw_text).strip()
                full_text_results[original_idx] = processed_text

            return full_text_results

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        print_progress: bool = False,
        language: str = "auto",
        **kwargs
    ) -> dict:
        if not vad_segments: 
            return {"segments": [], "language": "unknown"}
        
        self._clear_memory()
        batch_result = self.batch_recognize_via_tempfile(audio, vad_segments, language=language)
        
        segments = []
        for res, segment_info in zip(batch_result, vad_segments):
            text = self.emoji_pattern.sub(r"", res).strip()
            segments.append({
                "text": text,
                "start": round(segment_info["start"], 3),
                "end": round(segment_info["end"], 3),
                "speaker": segment_info.get("speaker", None),
            })
        
        self._clear_memory()
        
        return {"segments": segments, "language": "unknown"}

def load_asr_model(asr_model: str, model_dir: str, vad_model_dir: str, device: str, **kwargs):
    return FunASR(asr_model, model_dir, vad_model_dir, device, **kwargs)

def __repr__(self):
    return f"FunASR(model={self.asr_model}, device={self.device})"
