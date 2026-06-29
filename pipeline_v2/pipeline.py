"""PipelineV2: explicit DI, no global singleton."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

import logger
from logger import make_extra_tags
from models import funasr_asr
from pipeline_v2.exceptions import PipelineError
from pipeline_v2.params import PipelineParams
from pipeline_v2.state import PipelineState
from pipeline_v2.steps.embedding_refinement import EmbeddingRefiner
from pipeline_v2.steps.export import Exporter
from pipeline_v2.steps.segment import Segmenter
from pipeline_v2.steps.source_separation import Separator
from pipeline_v2.steps.speaker_diarization import Diarizer
from pipeline_v2.steps.standardization import Standardizer
from pipeline_v2.steps.vad_detection import VadDetector


class PipelineV2:
    def __init__(self, params: PipelineParams) -> None:
        self.params: PipelineParams = params
        self.standardizer: Standardizer = Standardizer(
            params.standardization, params.device_name
        )
        self.separator: Optional[Separator] = (
            Separator(params.source_separation, params.device_name)
            if params.source_separation.enable
            else None
        )
        self.diarizer: Diarizer = Diarizer(params.diarization, params.device_name)
        self.vad_detector: VadDetector = VadDetector(params.device_name)
        self.embedding_refiner: Optional[EmbeddingRefiner] = (
            EmbeddingRefiner(params.embedding_refinement, params.device_name)
            if params.embedding_refinement.enable
            else None
        )
        self.segmenter: Segmenter = Segmenter(params.segmenter, self.vad_detector.vad_model)
        self.exporter: Exporter = Exporter()
        self._funasr_warmup = self._load_funasr_warmup(params)

    def _load_funasr_warmup(self, params: PipelineParams):
        """Preload-only FunASR model. Not used by any stage; loading it
        warms up CUDA / numpy kernels and noticeably speeds up later
        numpy/torch work in the same worker.

        Cache path is preferred when it exists on disk; otherwise the hub
        id is used.
        """
        fw = params.funasr_warmup
        model_dir = (
            fw.asr_model_dir_cache
            if fw.asr_model_dir_cache and os.path.exists(fw.asr_model_dir_cache)
            else fw.asr_model_id
        )
        vad_model_dir = (
            fw.vad_model_dir_cache
            if fw.vad_model_dir_cache and os.path.exists(fw.vad_model_dir_cache)
            else fw.vad_model_id
        )
        return funasr_asr.load_asr_model(
            asr_model=fw.asr_model,
            model_dir=model_dir,
            vad_model_dir=vad_model_dir,
            device=params.device_name,
        )

    def load_models(self) -> None:
        """Reserved for future stages that need lazy model loading."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Stages: PipelineState -> PipelineState
    # ------------------------------------------------------------------
    def standardize(self, state: PipelineState) -> list[PipelineState]:
        results = self.standardizer.run(state.audio_path, log_tag=state.log_tag)
        if not results:
            raise PipelineError("standardize", "no chunks")
        states: list[PipelineState] = []
        for idx, r in enumerate(results):
            tag = make_extra_tags(audio_file=state.log_tag.get("audio_file", ""))
            tag["audio_file"] = f"{tag['audio_file']}#chunk{idx}"
            s = PipelineState(audio_path=state.audio_path, log_tag=tag)
            s.waveform = r.waveform
            s.sample_rate = r.sample_rate
            s.duration = r.duration
            states.append(s)
        return states

    def separate(self, state: PipelineState) -> PipelineState:
        if self.separator is None or state.waveform is None or state.sample_rate is None:
            return state
        waveform = self.separator.run(
            state.waveform, state.sample_rate, log_tag=state.log_tag
        )
        if waveform is None:
            raise PipelineError("separate", "waveform is None")
        state.waveform = waveform
        return state

    def diarize(self, state: PipelineState) -> PipelineState:
        if state.waveform is None or state.sample_rate is None:
            raise PipelineError("diarize", "waveform/sample_rate missing")
        result = self.diarizer.run(
            state.waveform, state.sample_rate, log_tag=state.log_tag
        )
        if result is None:
            raise PipelineError("diarize", "result is None")
        state.diarize_df, state.speaker_centroids = result
        return state

    def vad(self, state: PipelineState) -> PipelineState:
        if (
            state.diarize_df is None
            or state.waveform is None
            or state.sample_rate is None
        ):
            raise PipelineError("vad", "diarize_df/waveform/sample_rate missing")
        vad_list = self.vad_detector.run(
            state.diarize_df, state.waveform, state.sample_rate, log_tag=state.log_tag
        )
        if vad_list is None:
            raise PipelineError("vad", "vad_list is None")
        state.vad_list = vad_list
        return state

    def refine_embeddings(self, state: PipelineState) -> PipelineState:
        if self.embedding_refiner is None or state.vad_list is None:
            return state
        if state.waveform is None or state.sample_rate is None:
            raise PipelineError("refine_embeddings", "waveform/sample_rate missing")
        refined = self.embedding_refiner.run(
            state.vad_list, state.waveform, state.sample_rate, log_tag=state.log_tag
        )
        if refined is None:
            raise PipelineError("refine_embeddings", "refined is None")
        state.vad_list = refined
        return state

    def segment(self, state: PipelineState) -> PipelineState:
        if state.vad_list is None or state.waveform is None or state.sample_rate is None:
            raise PipelineError("segment", "vad_list/waveform/sample_rate missing")
        segment_list = self.segmenter.run(
            state.vad_list, state.waveform, state.sample_rate, log_tag=state.log_tag
        )
        if segment_list is None:
            raise PipelineError("segment", "segment_list is None")
        state.segment_list = segment_list
        return state

    def export(self, state: PipelineState, chunk_index: int, output_folder: str) -> PipelineState:
        if state.segment_list is None:
            raise PipelineError("export", "segment_list missing")
        if state.waveform is None or state.sample_rate is None:
            raise PipelineError("export", "waveform/sample_rate missing")
        path = self.exporter.run(
            state.segment_list, state.waveform, state.sample_rate,
            state.audio_path, chunk_index, output_folder,
            log_tag=state.log_tag,
        )
        if path is None:
            raise PipelineError("export", "path is None")
        state.export_path = path
        return state

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, audio_path: str, output_folder: str) -> list[PipelineState]:
        bootstrap = PipelineState(
            audio_path=audio_path,
            log_tag=make_extra_tags(audio_file=os.path.basename(audio_path)),
        )
        try:
            chunk_states = self.standardize(bootstrap)
            out: list[PipelineState] = []
            for idx, state in enumerate(chunk_states):
                t0 = time.perf_counter()
                state = self.separate(state)
                state = self.diarize(state)
                state = self.vad(state)
                vad_dur = sum(s.end - s.start for s in state.vad_list or [])
                state = self.refine_embeddings(state)
                refine_dur = sum(s.end - s.start for s in state.vad_list or [])
                state = self.segment(state)
                state = self.export(state, idx, output_folder)
                _log_chunk_stats(state, t0, vad_dur, refine_dur)
                out.append(state)
            return out
        except PipelineError as e:
            logger.error(f"pipeline_failed stage {e.stage} msg {e.message}", extra=bootstrap.log_tag)
            raise
        except Exception as e:
            logger.error(f"pipeline_failed unexpected {type(e).__name__} {e}", extra=bootstrap.log_tag)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _log_chunk_stats(
    state: PipelineState, t0: float, vad_dur: float, refine_dur: float
) -> None:
    seg_lens = [s.end - s.start for s in state.segment_list or []]
    seg_dur = sum(seg_lens)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    input_sec = state.duration or 0.0

    def pct(x: float) -> float:
        return (x / input_sec * 100.0) if input_sec > 0 else 0.0

    throughput = input_sec / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
    if seg_lens:
        arr = np.asarray(seg_lens)
        seg_mean = float(arr.mean())
        seg_p50, seg_p90, seg_p99 = (float(x) for x in np.percentile(arr, [50, 90, 99]))
    else:
        seg_mean = seg_p50 = seg_p90 = seg_p99 = 0.0

    logger.info(
        f"chunk_done input_sec {input_sec:.1f} wall_ms {wall_ms} "
        f"throughput {throughput:.2f} segments {len(seg_lens)} "
        f"retain_vad_percent {pct(vad_dur):.1f}% "
        f"retain_refine_percent {pct(refine_dur):.1f}% "
        f"retain_seg_percent {pct(seg_dur):.1f}% "
        f"seg_mean_sec {seg_mean:.2f} seg_p50_sec {seg_p50:.2f} "
        f"seg_p90_sec {seg_p90:.2f} seg_p99_sec {seg_p99:.2f}",
        extra=state.log_tag,
    )
