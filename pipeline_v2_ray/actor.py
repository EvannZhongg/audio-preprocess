"""GpuPipelineActor: one persistent Ray actor per pipe_slot.

On construction it detects its GPU, picks the matching pre-resolved
PipelineParams from the RayConfig shipped by the head node (no config file IO),
and builds a PipelineV2 with all models resident on cuda:0.

Concurrency model: the actor is created with Ray max_concurrency=N, so up to N
process_file() calls run on separate threads at once. Decode (ffmpeg, CPU) and
export (mp3 write, CPU/JuiceFS) run unlocked and therefore overlap freely; the
GPU stages are guarded by a single per-actor lock so exactly one file occupies
the GPU at a time. This overlaps IO with compute -- keeping the GPU busy --
without any explicit prefetch queue.

The actor does not self-recycle; the driver tracks per-actor lifecycle state
(files submitted, age) and decides when to retire it.
"""
from __future__ import annotations

import os
import threading
import time

import ray
import torch

import logger
from logger import make_extra_tags
from pipeline_v2.pipeline import PipelineV2
from pipeline_v2.state import PipelineState
from pipeline_v2_ray.config import RayConfig
from pipeline_v2_ray.result import FileResult


def _setup_env() -> None:
    """Redirect temp dirs onto the large scratch path and enable expandable
    CUDA segments. Mirrors main_v2.py; needed here because Ray actor processes
    on a pre-started cluster do not inherit the driver's environment."""
    large_temp = f"{os.getcwd()}/TEMP"
    os.makedirs(large_temp, exist_ok=True)
    os.environ["LARGE_TEMP_DIR"] = large_temp
    os.environ["TMPDIR"] = large_temp
    os.environ["TEMP"] = large_temp
    os.environ["TMP"] = large_temp
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@ray.remote
class GpuPipelineActor:
    def __init__(self, ray_config: RayConfig) -> None:
        # Actors on a pre-started cluster are separate daemons that do NOT
        # inherit the driver's environment, so set the temp-dir / allocator
        # vars here before any model touches disk or CUDA (funasr / brouhaha
        # rely on LARGE_TEMP_DIR to avoid overflowing /tmp).
        _setup_env()

        if not torch.cuda.is_available():
            raise RuntimeError("GpuPipelineActor scheduled on a node without CUDA")

        # ray_config arrives fully resolved from the head node: every profile
        # already carries a parsed PipelineParams, so we read no config files.
        self._gpu_name = torch.cuda.get_device_name(0)
        profile, params = ray_config.resolve_params(self._gpu_name)
        self._profile_name = profile.name

        self._pipeline = PipelineV2(params)

        # Only one file may occupy the GPU at a time; decode/export overlap.
        # (standardize's SileroVAD also runs on the GPU, but it's tiny and only
        # invoked for long-audio splitting -- sharing the CUDA context with the
        # locked GPU stages is harmless.)
        self._gpu_lock = threading.Lock()
        logger.info(
            f"ray_actor_ready gpu {self._gpu_name} profile {self._profile_name}"
        )

    def process_file(self, audio_path: str, output_folder: str, relative_path: str) -> dict:
        """Process one file; returns a serializable FileResult dict. Runs on a
        Ray worker thread (max_concurrency>1), overlapping its decode/export
        with other files' GPU work. Never raises: any error (including edge
        cases outside the inner per-stage handlers) becomes a failed result, so
        the driver never sees this as an actor-level crash."""
        try:
            return self._process_file_inner(audio_path, output_folder, relative_path)
        except Exception as e:  # noqa: BLE001 - last-resort guard; keep the actor alive
            logger.error(f"ray_process_file_error file {audio_path} err {type(e).__name__}: {e}")
            return FileResult(
                audio_path, success=False, error=f"{type(e).__name__}: {e}"
            ).to_dict()

    def _process_file_inner(self, audio_path: str, output_folder: str, relative_path: str) -> dict:
        log_tag = make_extra_tags(audio_file=os.path.basename(audio_path))
        # A decode failure means the whole file is unusable -> let the outer
        # guard in process_file turn it into a failed result.
        chunk_states = self._pipeline.standardize(
            PipelineState(audio_path=audio_path, relative_path=relative_path, log_tag=log_tag)
        )

        n_segments = 0
        failed_chunks = 0
        records: list = []
        for idx, state in enumerate(chunk_states):
            t0 = time.perf_counter()
            try:
                # GPU stages are exclusive; decode/export of other files overlap.
                with self._gpu_lock:
                    state, vad_dur, refine_dur = self._pipeline.run_gpu_stages(state)
                    torch.cuda.synchronize()
                self._pipeline.export(state, idx, output_folder)
                n_segments += len(state.segment_list or [])
                records.extend(state.export_records or [])
                PipelineV2.log_chunk_stats(state, t0, vad_dur, refine_dur)
            except Exception as e:  # noqa: BLE001 - one bad chunk must not sink the rest
                failed_chunks += 1
                logger.error(f"ray_chunk_failed {type(e).__name__}: {e}", extra=state.log_tag)

        success = failed_chunks == 0 and len(chunk_states) > 0
        error = "" if success else f"{failed_chunks}/{len(chunk_states)} chunks failed"
        # Any chunk failure fails the whole file: drop even the successful
        # chunks' segments so the segment table holds only fully-good files. The
        # file has no recorded segments -> resume reprocesses it on rerun; any
        # partial wav/json already on disk are harmless orphans, overwritten on
        # the deterministic-id rerun. Failures are surfaced via logs, not a table.
        return FileResult(
            audio_path, success=success, n_segments=n_segments,
            error=error, segments=records if success else [],
        ).to_dict()
