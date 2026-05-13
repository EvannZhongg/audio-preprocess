"""Qwen3-ASR model wrapper for the audio preprocessing pipeline.

This module provides a Qwen3-ASR model wrapper that uses Polaris service discovery
to connect to a vLLM-served Qwen3-ASR instance for speech recognition.
"""

import base64
import gc
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class Qwen3ASR:
    """
    ASR class using Qwen3-ASR model via Polaris service discovery.

    The model is served by vLLM and accessed via HTTP API. It supports
    automatic language detection and hot words for improved recognition.
    """

    # Qwen3-ASR language name to ISO code mapping
    LANGUAGE_MAPPING = {
        'Chinese': 'zh',
        'English': 'en',
        'Cantonese': 'yue',
        'Japanese': 'ja',
        'Korean': 'ko',
        'French': 'fr',
        'German': 'de',
        'Spanish': 'es',
        'Portuguese': 'pt',
        'Russian': 'ru',
        'Arabic': 'ar',
        'Thai': 'th',
        'Vietnamese': 'vi',
        'Indonesian': 'id',
        'Malay': 'ms',
        'Hindi': 'hi',
        'Italian': 'it',
        'Dutch': 'nl',
        'Polish': 'pl',
        'Turkish': 'tr',
    }

    def __init__(
        self,
        namespace: str = "Production",
        service: str = "trpc.Serving.QwenASR17ServerVllmQwenASR.ChatService",
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        hot_words: Optional[str] = None,
    ):
        """Initialize Qwen3-ASR model.

        Args:
            namespace: Polaris namespace for service discovery.
            service: Polaris service name for the Qwen3-ASR vLLM instance.
            model_name: Model name to pass in API requests.
            device: Device string (not used for API calls, kept for compatibility).
            hot_words: Optional hot words string to improve recognition accuracy.
        """
        self.namespace = namespace
        self.service = service
        self.model_name = model_name
        self.device = device
        self.hot_words = hot_words or ""
        self.timeout = 60
        self.consumer_api = None
        self.instances = []
        self.current_instance_idx = 0

        self._init_polaris()

    def _init_polaris(self):
        """Initialize Polaris service discovery."""
        try:
            from polaris.api.consumer import create_consumer_by_config

            self.consumer_api = create_consumer_by_config("")
            logger.info(f"Initialized Qwen3-ASR with Polaris service discovery")
            self._discover_instances()
        except ImportError as e:
            raise ImportError(
                f"polaris package not found: {e}\n"
                "Please ensure the polaris-cpp-py package is installed in the environment."
            )

    def _discover_instances(self):
        """Discover service instances via Polaris."""
        from polaris.pkg.model.service import GetInstancesRequest
        from polaris.pkg.model.error import SDKError

        request = GetInstancesRequest(namespace=self.namespace, service=self.service)
        try:
            # prefer get_all_instances (new API), fallback to get_instances (deprecated)
            if hasattr(self.consumer_api, 'get_all_instances'):
                response = self.consumer_api.get_all_instances(request)
            else:
                response = self.consumer_api.get_instances(request)
            self.instances = []
            for inst in response:
                host = inst.get_host()
                port = inst.get_port()
                self.instances.append((host, port))

            if not self.instances:
                raise RuntimeError(f"No instances found for {self.namespace}/{self.service}")

            logger.info(f"Discovered {len(self.instances)} Qwen3-ASR instances")
        except SDKError as e:
            raise RuntimeError(f"Polaris service discovery failed: {repr(e)}")

    def _get_next_instance(self) -> tuple:
        """Get next instance using round-robin load balancing."""
        if not self.instances:
            self._discover_instances()

        instance = self.instances[self.current_instance_idx]
        self.current_instance_idx = (self.current_instance_idx + 1) % len(self.instances)
        return instance

    def _transcribe_single(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        hot_words: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a single audio segment via Qwen3-ASR API.

        Args:
            audio_segment: Audio numpy array (mono, 16kHz).
            sample_rate: Sample rate of the audio (should be 16000).
            language: Language code (not used by Qwen3-ASR, kept for compatibility).
            hot_words: Hot words string to improve recognition.

        Returns:
            Dictionary with 'text' and 'language' keys.
        """
        import requests

        # Save audio to temporary WAV file and encode as base64
        custom_temp_dir = os.environ.get("LARGE_TEMP_DIR", None)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=custom_temp_dir) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio_segment, sample_rate)

            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_source = f"data:audio/wav;base64,{audio_base64}"

            # Prepare request content
            content = [
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_source}
                }
            ]

            # Add hot words if provided
            effective_hot_words = hot_words or self.hot_words
            if effective_hot_words and effective_hot_words.strip():
                content.append({
                    "type": "text",
                    "text": effective_hot_words.strip()
                })

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }

            # Get instance and make request
            host, port = self._get_next_instance()
            url = f"http://{host}:{port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()

            # Extract transcription text
            content_text = result["choices"][0]["message"]["content"]

            # Parse Qwen3-ASR output format: language <language_name><asr_text>text
            # Example: "language Chinese<asr_text>你好世界"
            lang_match = re.match(r'language\s+(\w+)<asr_text>(.*)', content_text, re.DOTALL)

            if lang_match:
                detected_language = lang_match.group(1).strip()
                text = lang_match.group(2).strip()
            else:
                # Fallback: try to extract text after <asr_text> tag
                asr_text_match = re.search(r'<asr_text>(.*)', content_text, re.DOTALL)
                if asr_text_match:
                    text = asr_text_match.group(1).strip()
                    detected_language = "unknown"
                else:
                    detected_language = "unknown"
                    text = content_text.strip()

            # Map language name to ISO code
            lang_code = self.LANGUAGE_MAPPING.get(detected_language, 'unknown')

            if not text:
                logger.warning(f"Qwen3-ASR returned empty text. Raw response: {content_text[:200]}")

            if lang_code == 'unknown' and detected_language != 'unknown':
                logger.warning(f"Qwen3-ASR detected unknown language mapping: '{detected_language}'")

            return {
                'text': text,
                'language': lang_code,
                'language_full': detected_language,
            }

        except requests.exceptions.Timeout:
            logger.warning(f"Qwen3-ASR request timed out after {self.timeout}s")
            return {'text': '', 'language': 'unknown', 'language_full': 'unknown'}
        except requests.exceptions.RequestException as e:
            logger.error(f"Qwen3-ASR HTTP error: {e}")
            return {'text': '', 'language': 'unknown', 'language_full': 'unknown'}
        except Exception as e:
            logger.error(f"Qwen3-ASR transcription error: {e}")
            return {'text': '', 'language': 'unknown', 'language_full': 'unknown'}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def detect_language(self, audio: np.ndarray) -> tuple:
        """Detect language from audio using Qwen3-ASR.

        Args:
            audio: Input audio numpy array (16kHz mono).

        Returns:
            Tuple of (language_code, probability).
        """
        result = self._transcribe_single(audio, sample_rate=16000)
        lang_code = result.get('language', 'unknown')
        # Qwen3-ASR generally has high confidence in language detection
        confidence = 0.9 if lang_code != 'unknown' else 0.0
        return lang_code, confidence

    def transcribe(
        self,
        audio: np.ndarray,
        vad_segments: List[dict],
        batch_size: int = 1,
        language: Optional[str] = None,
        print_progress: bool = False,
        **kwargs
    ) -> dict:
        """Transcribe audio segments using Qwen3-ASR.

        This method follows the same interface as whisper_asr and funasr_asr,
        making it a drop-in replacement in the pipeline.

        Args:
            audio: The full audio numpy array (16kHz mono).
            vad_segments: List of VAD segments with 'start' and 'end' keys (in seconds).
            batch_size: Not used (Qwen3-ASR processes one segment at a time via API).
            language: Language code (passed for compatibility, Qwen3-ASR auto-detects).
            print_progress: Whether to print progress.

        Returns:
            Dictionary with 'segments' (list of transcription results) and 'language'.
        """
        if not vad_segments:
            return {"segments": [], "language": "unknown"}

        segments = []
        detected_language = "unknown"
        sample_rate = 16000  # Audio should already be resampled to 16kHz
        empty_count = 0

        for idx, segment_info in enumerate(vad_segments):
            start_frame = int(segment_info["start"] * sample_rate)
            end_frame = int(segment_info["end"] * sample_rate)
            segment_audio = audio[start_frame:end_frame]

            if len(segment_audio) == 0:
                segments.append({
                    "text": "",
                    "start": round(segment_info["start"], 3),
                    "end": round(segment_info["end"], 3),
                    "speaker": segment_info.get("speaker", None),
                })
                empty_count += 1
                continue

            result = self._transcribe_single(
                segment_audio,
                sample_rate=sample_rate,
                language=language,
                hot_words=kwargs.get("hot_words", None),
            )

            text = result.get('text', '').strip()
            seg_language = result.get('language', 'unknown')

            if not text:
                empty_count += 1

            # Use the first successfully detected language as the overall language
            if detected_language == "unknown" and seg_language != "unknown":
                detected_language = seg_language

            segments.append({
                "text": text,
                "start": round(segment_info["start"], 3),
                "end": round(segment_info["end"], 3),
                "speaker": segment_info.get("speaker", None),
                "detected_language": seg_language,
            })

        logger.info(
            f"Qwen3-ASR transcribe done: {len(segments)} segments, "
            f"{empty_count} empty, language={detected_language}"
        )

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"segments": segments, "language": detected_language}


def load_asr_model(
    namespace: str = "Production",
    service: str = "trpc.Serving.QwenASR17ServerVllmQwenASR.ChatService",
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    device: str = "cuda",
    hot_words: Optional[str] = None,
) -> Qwen3ASR:
    """Load Qwen3-ASR model.

    Args:
        namespace: Polaris namespace for service discovery.
        service: Polaris service name.
        model_name: Model name.
        device: Device string (for compatibility).
        hot_words: Optional hot words for recognition.

    Returns:
        Qwen3ASR instance.
    """
    return Qwen3ASR(
        namespace=namespace,
        service=service,
        model_name=model_name,
        device=device,
        hot_words=hot_words,
    )
