"""Step 0: standardization.

Decode an arbitrary audio file via ffmpeg into mono float32 PCM at the
target sample rate, then RMS-normalize toward target dBFS (±3 dB clamp)
with a peak guard.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import torch

import logger
from models.vad import SileroVAD
from pipeline_v2.params import StandardizationParams


_SILERO_SR = 16000


@dataclass
class StandardizationResult:
    waveform: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    duration: float


class Standardizer:
    """Step 0 runner. Stateless w.r.t. files; bound to a params object."""

    def __init__(self, params: StandardizationParams, device: str = "cpu") -> None:
        self.params = params
        self.vad_model = SileroVAD(device=torch.device(device))
        self._vad_lock = threading.Lock()

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self, audio_path: str, log_tag: Optional[dict] = None
    ) -> Optional[list[StandardizationResult]]:
        p = self.params
        t_total = time.perf_counter()

        try:
            if os.path.getsize(audio_path) > p.max_file_size_bytes:
                logger.error("std_skip oversized", extra=log_tag)
                return None
        except OSError as e:
            logger.error(f"std_skip stat_failed {e}", extra=log_tag)
            return None

        t0 = time.perf_counter()
        duration_sec, ffmpeg_timeout = self._probe_duration(audio_path, log_tag)
        probe_ms = int((time.perf_counter() - t0) * 1000)
        if duration_sec is not None and duration_sec > p.max_audio_duration_seconds:
            logger.error(f"std_skip long {duration_sec:.1f}s", extra=log_tag)
            return None

        t0 = time.perf_counter()
        pcm = self._ffmpeg_decode(audio_path, ffmpeg_timeout, log_tag)
        if pcm is None:
            return None
        decode_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        waveform = self._normalize(pcm, log_tag)
        normalize_ms = int((time.perf_counter() - t0) * 1000)

        total_dur = len(waveform) / p.target_sample_rate
        if total_dur < p.chunk_min_seconds:
            total_ms = int((time.perf_counter() - t_total) * 1000)
            logger.error(
                f"std_skip too_short {total_dur:.1f}s < {p.chunk_min_seconds:.0f}s "
                f"total_ms {total_ms}",
                extra=log_tag,
            )
            return None

        t0 = time.perf_counter()
        if total_dur <= p.chunk_max_seconds:
            chunks = [(0.0, total_dur)]
            split_ms = int((time.perf_counter() - t0) * 1000)
        else:
            chunks = self._split_by_vad(waveform, p.target_sample_rate, log_tag)
            split_ms = int((time.perf_counter() - t0) * 1000)
            if chunks is None:
                total_ms = int((time.perf_counter() - t_total) * 1000)
                logger.error(
                    f"std_skip no_split duration_s {total_dur:.1f} "
                    f"total_ms {total_ms}",
                    extra=log_tag,
                )
                return None

        results: list[StandardizationResult] = []
        for start_s, end_s in chunks:
            s = int(start_s * p.target_sample_rate)
            e = int(end_s * p.target_sample_rate)
            piece = waveform[s:e]
            results.append(StandardizationResult(
                waveform=piece,
                sample_rate=p.target_sample_rate,
                duration=len(piece) / p.target_sample_rate,
            ))

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"std_time_cost probe_ms {probe_ms} decode_ms {decode_ms} "
            f"normalize_ms {normalize_ms} split_ms {split_ms} total_ms {total_ms} "
            f"duration_s {total_dur:.1f} chunks {len(results)}",
            extra=log_tag,
        )
        return results

    # ------------------------------------------------------------------
    # split
    # ------------------------------------------------------------------
    def _split_by_vad(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict],
    ) -> Optional[list[tuple[float, float]]]:
        """Cut a long waveform into 10–30 min chunks at silero silences.

        Greedy: from the current cut point, find the longest speech segment
        whose end lies in [chunk_min, chunk_max] seconds from the cut and
        place the next cut at the silence right after it. If no candidate
        is found within `chunk_max`, abort (return None). Tail shorter than
        `chunk_min` is discarded.
        """
        p = self.params
        if sample_rate != _SILERO_SR:
            wav16 = librosa.resample(waveform, orig_sr=sample_rate, target_sr=_SILERO_SR)
        else:
            wav16 = waveform

        intervals = None
        for attempt in range(2):
            try:
                with self._vad_lock:
                    reset_states = getattr(
                        self.vad_model.vad_model, "reset_states", None
                    )
                    if callable(reset_states):
                        try:
                            reset_states()
                        except Exception:
                            pass
                    intervals = self.vad_model._get_speech_timestamps_wrapper(
                        wav16, _SILERO_SR
                    )
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        f"std_split_silero_retry {type(e).__name__}: {e}",
                        extra=log_tag,
                    )
                    continue
                logger.error(f"std_split_silero_failed {e}", extra=log_tag)
                return None
        if not intervals:
            return None

        speech: list[tuple[float, float]] = [
            (iv["start"] / _SILERO_SR, iv["end"] / _SILERO_SR) for iv in intervals
        ]
        total_dur = len(waveform) / sample_rate

        chunks: list[tuple[float, float]] = []
        cut = 0.0
        i = 0
        n = len(speech)
        while cut < total_dur:
            best_idx = -1
            best_len = -1.0
            j = i
            while j < n and speech[j][1] - cut <= p.chunk_max_seconds:
                seg_end = speech[j][1]
                seg_len = speech[j][1] - speech[j][0]
                if seg_end - cut >= p.chunk_min_seconds and seg_len > best_len:
                    best_len = seg_len
                    best_idx = j
                j += 1

            remaining = total_dur - cut
            if remaining <= p.chunk_max_seconds:
                if remaining >= p.chunk_min_seconds:
                    chunks.append((cut, total_dur))
                break

            if best_idx < 0:
                return None

            if best_idx + 1 < n:
                next_start = speech[best_idx + 1][0]
                cut_end = (speech[best_idx][1] + next_start) / 2.0
            else:
                cut_end = speech[best_idx][1]
            cut_end = min(cut_end, cut + p.chunk_max_seconds)
            chunks.append((cut, cut_end))
            cut = cut_end
            i = best_idx + 1

        if not chunks:
            return None
        return chunks

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _probe_duration(
        self, path: str, log_tag: Optional[dict]
    ) -> tuple[Optional[float], int]:
        """Return (duration_sec_or_None, ffmpeg_timeout)."""
        base_timeout = self.params.ffmpeg_timeout
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", path],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            dur = float(json.loads(out)["format"]["duration"])
            return dur, max(base_timeout, 60 + int(dur / 24))
        except subprocess.TimeoutExpired:
            logger.error("std_ffprobe_hang", extra=log_tag)
            return None, base_timeout
        except Exception as e:
            logger.error(f"std_ffprobe_failed {e}", extra=log_tag)
            return None, base_timeout

    def _ffmpeg_decode(
        self, path: str, timeout: int, log_tag: Optional[dict]
    ) -> Optional[np.ndarray]:
        """ffmpeg -> raw s16le PCM -> float32 mono numpy. None on failure."""
        cmd = [
            "ffmpeg",
            "-threads", str(self.params.num_threads),
            "-i", path,
            "-ar", str(self.params.target_sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            "-",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7
        )
        try:
            raw_data, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.error(f"std_ffmpeg_timeout {timeout}s", extra=log_tag)
            proc.kill()
            proc.communicate()
            return None

        if proc.returncode != 0:
            logger.error(f"std_ffmpeg_failed {stderr.decode(errors='replace')}", extra=log_tag)
            return None
        if not raw_data:
            logger.error("std_ffmpeg_empty", extra=log_tag)
            return None

        pcm16 = np.frombuffer(raw_data, dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0

    def _normalize(self, waveform: np.ndarray, log_tag: Optional[dict]) -> np.ndarray:
        """In-place RMS gain toward target_dbfs (±3 dB clamp) + peak guard."""
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        if rms > 0:
            current_dbfs = 20.0 * float(np.log10(rms))
            gain_db = self.params.target_dbfs - current_dbfs
            gain_db = max(-3.0, min(3.0, gain_db))
            waveform *= 10 ** (gain_db / 20.0)
        else:
            logger.error("std_zero_rms", extra=log_tag)

        if waveform.size:
            peak = float(np.max(np.abs(waveform)))
            if peak > 1.0:
                waveform /= peak
        return waveform
