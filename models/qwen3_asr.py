"""Qwen3-ASR model wrapper for the audio preprocessing pipeline.

This module provides a Qwen3-ASR model wrapper that uses Polaris service discovery
to connect to a vLLM-served Qwen3-ASR instance for speech recognition.
"""

import gc
import logging
import os
import tempfile
import time
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

    # Set of ISO codes the new /v1/audio/transcriptions endpoint is expected
    # to return directly (values of LANGUAGE_MAPPING). Used only as a sanity
    # check for the fallback described below.
    _ISO_LANGUAGE_CODES = set(LANGUAGE_MAPPING.values())

    # How often to re-pull the instance list from Polaris (seconds). Without
    # this, instances are discovered once at construction time and newly
    # added machines never receive traffic until the process is restarted.
    INSTANCE_REFRESH_INTERVAL = 60.0

    def __init__(
        self,
        namespace: str = "Test",
        service: str = "audio_process_qwen3_asr_service",
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
        self._last_discover_time = 0.0

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
            instances = []
            for inst in response:
                host = inst.get_host()
                port = inst.get_port()
                instances.append((host, port))

            if not instances:
                raise RuntimeError(f"No instances found for {self.namespace}/{self.service}")

            self.instances = instances
            self._last_discover_time = time.time()
            logger.info(f"Discovered {len(self.instances)} Qwen3-ASR instances")
        except SDKError as e:
            raise RuntimeError(f"Polaris service discovery failed: {repr(e)}")

    def _maybe_refresh_instances(self):
        """Periodically re-pull the instance list from Polaris so newly added
        (or removed) machines join/leave the round-robin rotation without
        requiring the process to restart. A transient Polaris failure here
        must not break in-flight requests, so we keep serving with the
        existing (possibly stale) list on error."""
        if time.time() - self._last_discover_time < self.INSTANCE_REFRESH_INTERVAL:
            return
        # Reset the timer up front so a failure doesn't cause a retry storm
        # (next attempt is still gated by the interval below).
        self._last_discover_time = time.time()
        old_count = len(self.instances)
        try:
            self._discover_instances()
            if len(self.instances) != old_count:
                logger.info(
                    f"Qwen3-ASR instance list refreshed: {old_count} -> {len(self.instances)}"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Qwen3-ASR periodic instance refresh failed, keep old list: {e}")

    def _get_next_instance(self) -> tuple:
        """Get next instance using round-robin load balancing."""
        if not self.instances:
            self._discover_instances()
        else:
            self._maybe_refresh_instances()

        # Guard against the list having shrunk since the index was last set.
        if self.current_instance_idx >= len(self.instances):
            self.current_instance_idx = 0
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

        # Save audio to a temporary WAV file to upload as multipart/form-data
        custom_temp_dir = os.environ.get("LARGE_TEMP_DIR", None)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=custom_temp_dir) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio_segment, sample_rate)

            # Get instance and make request. The real service expects a
            # plain multipart file upload, e.g.:
            #   curl -sS -F file=@chunk.wav http://{host}:{port}/v1/audio/transcriptions
            # and returns a flat JSON body: {"text": "...", "language": "zh"}.
            host, port = self._get_next_instance()
            url = f"http://{host}:{port}/v1/audio/transcriptions"

            effective_hot_words = hot_words or self.hot_words
            data = {}
            if effective_hot_words and effective_hot_words.strip():
                # Tentative: the documented interface only requires "file".
                # Forward hot words as an extra form field on a best-effort
                # basis; if the backend doesn't support it, it will simply
                # be ignored and won't affect the main transcription flow.
                data["hot_words"] = effective_hot_words.strip()

            with open(temp_path, 'rb') as f:
                files = {"file": (os.path.basename(temp_path), f, "audio/wav")}
                response = requests.post(
                    url, files=files, data=data or None, timeout=self.timeout
                )
            response.raise_for_status()

            result = response.json()

            # New endpoint returns a flat JSON: {"text": "...", "language": "zh"}
            text = (result.get("text") or "").strip()
            detected_language = (result.get("language") or "unknown").strip()

            # The endpoint is expected to already return an ISO code (e.g.
            # "zh"). Keep a lightweight fallback in case the backend ever
            # returns a full language name instead.
            if detected_language not in self._ISO_LANGUAGE_CODES and detected_language != "unknown":
                detected_language = self.LANGUAGE_MAPPING.get(detected_language, detected_language)

            lang_code = detected_language or 'unknown'

            if not text:
                logger.warning(f"Qwen3-ASR returned empty text. Raw response: {str(result)[:200]}")

            return {
                'text': text,
                'language': lang_code,
                'language_full': lang_code,
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
    namespace: str = "Test",
    service: str = "audio_process_qwen3_asr_service",
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
