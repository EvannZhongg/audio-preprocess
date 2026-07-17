"""Merge / split / filter / grace-period the refined VAD segments.

Inputs are segments produced by `EmbeddingRefiner` (`reference_embedding`
already attached). Same-speaker adjacent segments are merged when their
embeddings are similar enough, their gap is small enough, and the merged
length stays under `max_segment_length`. Overlong segments are split at
silero-detected internal silences, or dropped if no natural cut exists.
Short segments are filtered out; the remainder gets a small grace period
on start/end (clamped to neighbors).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import replace
from typing import Optional

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import logger
from models.vad import SileroVAD
from pipeline_v2.params import SegmenterParams
from pipeline_v2.state import Segment


_SILERO_SR = 16000


class Segmenter:
    """Produces the final `segment_list` consumed by downstream ASR.

    Reuses the SileroVAD instance owned by step 3 for splitting overlong
    spans, so this class doesn't load any model of its own.
    """

    def __init__(self, params: SegmenterParams, vad_model: SileroVAD) -> None:
        self.params = params
        self.vad_model = vad_model

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        vad_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[list[Segment]]:
        if not vad_list:
            logger.error("seg_empty_input_vad_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            merged, n_split, n_drop_long = self._split_and_merge(
                vad_list, waveform, sample_rate, log_tag
            )
            merge_ms = int((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            filtered = [
                s for s in merged
                if s.end - s.start >= self.params.min_segment_length
            ]
            n_drop_short = len(merged) - len(filtered)
            filter_ms = int((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            audio_duration = len(waveform) / sample_rate
            self._apply_grace_period(filtered, audio_duration)
            grace_ms = int((time.perf_counter() - t0) * 1000)
        except Exception:
            logger.error(f"seg_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"seg_time_cost in {len(vad_list)} out {len(filtered)} "
            f"split {n_split} drop_long {n_drop_long} drop_short {n_drop_short} "
            f"merge_ms {merge_ms} filter_ms {filter_ms} grace_ms {grace_ms} "
            f"total_ms {total_ms}",
            extra=log_tag,
        )
        return filtered

    # ------------------------------------------------------------------
    # merge / split
    # ------------------------------------------------------------------
    def _split_and_merge(
        self,
        vad_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict],
    ) -> tuple[list[Segment], int, int]:
        p = self.params
        out: list[Segment] = []
        n_split = 0
        n_drop_long = 0

        for seg in vad_list:
            duration = seg.end - seg.start
            if duration >= p.max_segment_length:
                sub = self._split_long(seg, waveform, sample_rate, log_tag)
                if sub is None:
                    n_drop_long += 1
                    continue
                out.extend(sub)
                n_split += 1
                continue

            if not out:
                out.append(seg)
                continue

            last = out[-1]
            if last.speaker != seg.speaker:
                out.append(seg)
                continue
            if last.reference_embedding is None or seg.reference_embedding is None:
                out.append(seg)
                continue
            if (
                cosine_similarity(last.reference_embedding, seg.reference_embedding)[0, 0]
                < p.intra_similarity_threshold
            ):
                out.append(seg)
                continue

            gap = seg.start - last.end
            merged_dur = seg.end - last.start
            if gap >= p.merge_gap or merged_dur >= p.max_segment_length:
                out.append(seg)
            else:
                last.end = seg.end

        return out, n_split, n_drop_long

    def _split_long(
        self,
        seg: Segment,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict],
    ) -> Optional[list[Segment]]:
        """Split an overlong span at silero-detected internal silences.

        Returns None when there is no usable silence (continuous speech) —
        caller drops the span rather than cutting mid-word.
        """
        max_len = self.params.max_segment_length
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        duration = seg_end - seg_start
        if duration < max_len:
            return [seg]

        s_idx = max(0, int(seg_start * sample_rate))
        e_idx = min(len(waveform), int(seg_end * sample_rate))
        if e_idx <= s_idx:
            return None
        seg_audio = waveform[s_idx:e_idx]
        if seg_audio.ndim > 1:
            seg_audio = seg_audio.mean(axis=0)
        if sample_rate != _SILERO_SR:
            seg_audio = librosa.resample(
                seg_audio, orig_sr=sample_rate, target_sr=_SILERO_SR
            )

        try:
            intervals = self.vad_model._get_speech_timestamps_wrapper(
                seg_audio, _SILERO_SR
            )
        except Exception as e:
            logger.error(f"seg_split_silero_failed {e}", extra=log_tag)
            return None

        if not intervals or len(intervals) < 2:
            return None

        gaps = []
        for i in range(len(intervals) - 1):
            prev_end_s = intervals[i].get("end", 0) / _SILERO_SR
            next_start_s = intervals[i + 1].get("start", 0) / _SILERO_SR
            g = next_start_s - prev_end_s
            if g > 0:
                gaps.append((g, i, prev_end_s, next_start_s))
        if not gaps:
            return None

        target_chunk = max_len * 0.9
        n_chunks_needed = max(2, int(np.ceil(duration / target_chunk)))
        n_cuts_needed = n_chunks_needed - 1
        chosen = sorted(
            sorted(gaps, key=lambda x: -x[0])[:n_cuts_needed],
            key=lambda x: x[1],
        )
        cut_times = [
            seg_start + (prev_end + next_start) / 2.0
            for _, _, prev_end, next_start in chosen
        ]

        sub: list[Segment] = []
        prev_t = seg_start
        for ct in cut_times:
            sub.append(replace(seg, start=prev_t, end=ct))
            prev_t = ct
        sub.append(replace(seg, start=prev_t, end=seg_end))

        for c in sub:
            if c.end - c.start >= max_len:
                return None
        return sub

    # ------------------------------------------------------------------
    # grace period
    # ------------------------------------------------------------------
    def _apply_grace_period(self, segs: list[Segment], audio_duration: float) -> None:
        if not segs:
            return
        end_pad = self.params.grace_period_end
        start_pad = self.params.grace_period_start

        for i in range(len(segs) - 1):
            new_end = segs[i].end + end_pad
            segs[i].end = min(new_end, segs[i + 1].start)
        segs[-1].end = min(segs[-1].end + end_pad, audio_duration)

        segs[0].start = max(0.0, segs[0].start - start_pad)
        for i in range(1, len(segs)):
            segs[i].start = max(segs[i].start - start_pad, segs[i - 1].end)
