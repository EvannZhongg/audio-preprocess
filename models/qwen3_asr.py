"""Qwen3-ASR model wrapper for the audio preprocessing pipeline.

This module provides a Qwen3-ASR model wrapper that uses Polaris service discovery
to connect to a vLLM-served Qwen3-ASR instance for speech recognition.
"""

import gc
import io
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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

    # Default number of segments packed into a single /v1/audio/transcriptions
    # /batch request, and default number of such batch requests issued
    # concurrently for one chunk. See `transcribe()` for the rationale.
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_MAX_GROUP_WORKERS = 8

    # Extra seconds added to the request timeout for each *additional* file in
    # a batch. A batch request does one inference call over N files, so its
    # wall time grows roughly linearly with N -- reusing the single-file
    # timeout verbatim would make bigger batches spuriously time out.
    BATCH_TIMEOUT_PER_FILE = 10

    def __init__(
        self,
        namespace: str = "Test",
        service: str = "audio_process_qwen3_asr_service",
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        hot_words: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_group_workers: int = DEFAULT_MAX_GROUP_WORKERS,
    ):
        """Initialize Qwen3-ASR model.

        Args:
            namespace: Polaris namespace for service discovery.
            service: Polaris service name for the Qwen3-ASR vLLM instance.
            model_name: Model name to pass in API requests.
            device: Device string (not used for API calls, kept for compatibility).
            hot_words: Optional hot words string to improve recognition accuracy.
            batch_size: How many segments to pack into one batch request.
            max_group_workers: Upper bound on the number of batch requests
                issued concurrently while transcribing a single chunk.
        """
        self.namespace = namespace
        self.service = service
        self.model_name = model_name
        self.device = device
        self.hot_words = hot_words or ""
        self.timeout = 60
        self.batch_size = max(1, int(batch_size))
        self.max_group_workers = max(1, int(max_group_workers))
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

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Uniform "no transcription" payload used for every failure path.

        Failures are intentionally degraded into empty text (instead of raised)
        so that one bad segment / one bad batch never aborts a whole chunk.
        """
        return {'text': '', 'language': 'unknown', 'language_full': 'unknown'}

    def _parse_result_payload(self, result: Any) -> Dict[str, Any]:
        """Normalize one per-file JSON payload into the internal contract.

        Shared by the single-file and batch code paths: both endpoints return
        the same flat shape ({"text": ..., "language": ...}), the batch one
        just nests a list of them under "results".
        """
        if not isinstance(result, dict):
            logger.warning(
                f"Qwen3-ASR unexpected result payload type {type(result).__name__}: "
                f"{str(result)[:200]}"
            )
            return self._empty_result()

        text = (result.get("text") or "").strip()
        detected_language = (result.get("language") or "unknown").strip()

        # The endpoint is expected to already return an ISO code (e.g. "zh").
        # Keep a lightweight fallback in case the backend ever returns a full
        # language name instead.
        if detected_language not in self._ISO_LANGUAGE_CODES and detected_language != "unknown":
            detected_language = self.LANGUAGE_MAPPING.get(detected_language, detected_language)

        lang_code = detected_language or 'unknown'

        return {
            'text': text,
            'language': lang_code,
            'language_full': lang_code,
        }

    def _transcribe_batch(
        self,
        segment_audios: List[np.ndarray],
        sample_rate: int = 16000,
        hot_words: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe several audio segments with ONE batch request.

        Uses the service's extended route:
            POST /v1/audio/transcriptions/batch
        which accepts repeated `audio_files` multipart fields (or `file_paths`)
        and answers with {"results": [...]} ordered exactly like the upload.
        One request == one inference call over all files, which is what lets
        the server-side GPU actually batch instead of idling between tiny
        single-file round trips.

        Segments are encoded to WAV fully in memory (`io.BytesIO`); unlike the
        single-file path this never touches the filesystem, avoiding a
        write/open/unlink cycle per segment on (potentially CFS-backed)
        LARGE_TEMP_DIR.

        Args:
            segment_audios: Non-empty audio arrays, in the order to upload.
            sample_rate: Sample rate shared by all segments.
            hot_words: Hot words string, shared by the whole batch.

        Returns:
            A list of per-segment dicts, always exactly as long as
            `segment_audios` and in the same order. On any failure the
            corresponding entries are empty results rather than exceptions.
        """
        import requests

        if not segment_audios:
            return []

        # Encode every segment to an in-memory WAV buffer.
        files = []
        try:
            for idx, segment_audio in enumerate(segment_audios):
                buf = io.BytesIO()
                sf.write(buf, segment_audio, sample_rate, format="WAV")
                buf.seek(0)
                # Repeated field name -> the server sees audio_files[].
                files.append(
                    ("audio_files", (f"segment_{idx}.wav", buf, "audio/wav"))
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Qwen3-ASR failed to encode batch audio in memory: {e}")
            return [self._empty_result() for _ in segment_audios]

        host, port = self._get_next_instance()
        url = f"http://{host}:{port}/v1/audio/transcriptions/batch"

        effective_hot_words = hot_words or self.hot_words
        data = {}
        if effective_hot_words and effective_hot_words.strip():
            # Best-effort, same as the single-file path: the documented batch
            # interface only lists language / response_format /
            # timestamp_granularities[] as shared fields, so a backend that
            # doesn't know `hot_words` will simply ignore it.
            data["hot_words"] = effective_hot_words.strip()

        # A batch does one inference pass over N files, so scale the timeout
        # with N instead of reusing the single-file constant.
        timeout = self.timeout + self.BATCH_TIMEOUT_PER_FILE * (len(segment_audios) - 1)

        try:
            response = requests.post(
                url, files=files, data=data or None, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            logger.warning(
                f"Qwen3-ASR batch request ({len(segment_audios)} files) timed out "
                f"after {timeout}s on {host}:{port}"
            )
            return [self._empty_result() for _ in segment_audios]
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Qwen3-ASR batch HTTP error on {host}:{port} "
                f"({len(segment_audios)} files): {e}"
            )
            return [self._empty_result() for _ in segment_audios]
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Qwen3-ASR batch transcription error on {host}:{port} "
                f"({len(segment_audios)} files): {e}"
            )
            return [self._empty_result() for _ in segment_audios]

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            logger.error(
                f"Qwen3-ASR batch response has no 'results' list "
                f"(sent {len(segment_audios)} files). Raw: {str(payload)[:200]}"
            )
            return [self._empty_result() for _ in segment_audios]

        # Defensive alignment: results are positional, so a length mismatch
        # would silently shift text onto the wrong timestamps. Pad/truncate
        # instead of trusting the response blindly.
        if len(results) != len(segment_audios):
            logger.warning(
                f"Qwen3-ASR batch returned {len(results)} results for "
                f"{len(segment_audios)} uploaded files; padding/truncating"
            )
            if len(results) < len(segment_audios):
                results = results + [None] * (len(segment_audios) - len(results))
            else:
                results = results[: len(segment_audios)]

        return [self._parse_result_payload(item) for item in results]

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

        Segments of one chunk are grouped into batches of `self.batch_size` and
        each group is sent as a single `/v1/audio/transcriptions/batch` request;
        groups are issued concurrently (HTTP is pure I/O wait) and land on
        different service instances via the existing round-robin. This replaces
        the previous one-request-per-segment serial loop, which left the remote
        GPU mostly idle waiting for tiny sequential round trips.

        Args:
            audio: The full audio numpy array (16kHz mono).
            vad_segments: List of VAD segments with 'start' and 'end' keys (in seconds).
            batch_size: Ignored; batching is controlled by `self.batch_size`
                (kept in the signature for interface compatibility with the
                local whisper/funasr backends).
            language: Language code (passed for compatibility, Qwen3-ASR auto-detects).
            print_progress: Whether to print progress.

        Returns:
            Dictionary with 'segments' (list of transcription results) and 'language'.
            `segments` is always the same length and order as `vad_segments`.
        """
        if not vad_segments:
            return {"segments": [], "language": "unknown"}

        sample_rate = 16000  # Audio should already be resampled to 16kHz
        hot_words = kwargs.get("hot_words", None)

        # Pre-allocate so results can be written back positionally: the caller
        # (`run_stage2_asr`) relies on a strict 1:1 order match with its input.
        segments: List[Optional[dict]] = [None] * len(vad_segments)

        # Slice out audio first; empty segments never hit the network (same
        # behaviour as before) and are filled in directly.
        pending_indices: List[int] = []
        pending_audios: List[np.ndarray] = []
        for idx, segment_info in enumerate(vad_segments):
            start_frame = int(segment_info["start"] * sample_rate)
            end_frame = int(segment_info["end"] * sample_rate)
            segment_audio = audio[start_frame:end_frame]

            if len(segment_audio) == 0:
                segments[idx] = {
                    "text": "",
                    "start": round(segment_info["start"], 3),
                    "end": round(segment_info["end"], 3),
                    "speaker": segment_info.get("speaker", None),
                }
                continue

            pending_indices.append(idx)
            pending_audios.append(segment_audio)

        # Split into fixed-size groups; each group == one batch request.
        groups = [
            (pending_indices[i:i + self.batch_size], pending_audios[i:i + self.batch_size])
            for i in range(0, len(pending_indices), self.batch_size)
        ]

        group_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(groups)
        if len(groups) == 1:
            # Single group: skip the thread pool entirely.
            group_results[0] = self._transcribe_batch(
                groups[0][1], sample_rate=sample_rate, hot_words=hot_words
            )
        elif groups:
            workers = min(len(groups), self.max_group_workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._transcribe_batch,
                        group_audios,
                        sample_rate,
                        hot_words,
                    ): g_idx
                    for g_idx, (_, group_audios) in enumerate(groups)
                }
                for future in futures:
                    g_idx = futures[future]
                    try:
                        group_results[g_idx] = future.result()
                    except Exception as e:  # noqa: BLE001
                        # _transcribe_batch already degrades internally, so
                        # this is only for truly unexpected failures; keep
                        # the other groups' results usable.
                        logger.error(f"Qwen3-ASR batch group {g_idx} failed: {e}")
                        group_results[g_idx] = [
                            self._empty_result() for _ in groups[g_idx][1]
                        ]

        detected_language = "unknown"
        empty_count = sum(1 for seg in segments if seg is not None)

        for (group_indices, group_audios), results in zip(groups, group_results):
            if results is None:
                results = [self._empty_result() for _ in group_audios]
            for idx, result in zip(group_indices, results):
                segment_info = vad_segments[idx]
                text = result.get('text', '').strip()
                seg_language = result.get('language', 'unknown')

                if not text:
                    empty_count += 1

                # Use the first successfully detected language as the overall
                # language. Note this is now resolved in segment order (not
                # completion order), so it stays deterministic despite the
                # concurrent requests.
                if detected_language == "unknown" and seg_language != "unknown":
                    detected_language = seg_language

                segments[idx] = {
                    "text": text,
                    "start": round(segment_info["start"], 3),
                    "end": round(segment_info["end"], 3),
                    "speaker": segment_info.get("speaker", None),
                    "detected_language": seg_language,
                }

        # Last-resort guard: every slot must be a dict, since the caller pairs
        # this list 1:1 with its own segments. Normally unreachable (each group
        # returns exactly as many results as it uploaded), but a `None` leaking
        # through would blow up downstream instead of just losing one segment.
        for idx, seg in enumerate(segments):
            if seg is not None:
                continue
            logger.warning(f"Qwen3-ASR segment {idx} got no result; filling empty")
            segment_info = vad_segments[idx]
            segments[idx] = {
                "text": "",
                "start": round(segment_info["start"], 3),
                "end": round(segment_info["end"], 3),
                "speaker": segment_info.get("speaker", None),
                "detected_language": "unknown",
            }
            empty_count += 1

        logger.info(
            f"Qwen3-ASR transcribe done: {len(segments)} segments in "
            f"{len(groups)} batch request(s) (batch_size={self.batch_size}), "
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
    batch_size: int = Qwen3ASR.DEFAULT_BATCH_SIZE,
    max_group_workers: int = Qwen3ASR.DEFAULT_MAX_GROUP_WORKERS,
) -> Qwen3ASR:
    """Load Qwen3-ASR model.

    Args:
        namespace: Polaris namespace for service discovery.
        service: Polaris service name.
        model_name: Model name.
        device: Device string (for compatibility).
        hot_words: Optional hot words for recognition.
        batch_size: Segments packed into one batch request.
        max_group_workers: Max concurrent batch requests per chunk.

    Returns:
        Qwen3ASR instance.
    """
    return Qwen3ASR(
        namespace=namespace,
        service=service,
        model_name=model_name,
        device=device,
        hot_words=hot_words,
        batch_size=batch_size,
        max_group_workers=max_group_workers,
    )
