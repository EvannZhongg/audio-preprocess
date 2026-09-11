"""GpuPipelineActor: one persistent Ray actor per pipe_slot.

On construction it detects its GPU, picks the matching pre-resolved
PipelineParams from the RayConfig shipped by the head node (no config file IO),
and builds a PipelineV2 with all models resident on cuda:0.

Concurrency model: the actor is created with Ray max_concurrency=N, so up to N
process_file() calls run on separate threads at once. Decode (ffmpeg, CPU),
standardization/split and export (mp3 write, CPU/JuiceFS) all run unlocked and
therefore overlap freely across files -- each spawns its own subprocess or
works on purely local data, so there is no shared mutable state to race on.
The only genuinely thread-unsafe piece, Silero VAD inference (used both here
during long-file splitting and later in the VAD stage), guards itself via
SileroVAD._GLOBAL_LOCK (see models/vad.py) at the call-site granularity, so it
does not need to be serialized here too. The GPU stages are guarded by a
single per-actor lock so exactly one file occupies the GPU at a time. This
overlaps IO with compute -- keeping the GPU busy -- without any explicit
prefetch queue.

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
from pipeline_v2.state import (NO_SEGMENTS_MARKER, PIPELINE_VERSION,
                               PipelineState)
from pipeline_v2_ray.actors.base import PipelineActor, register_actor
from pipeline_v2_ray.config import RayConfig
from pipeline_v2_ray.result import FileResult
from pipeline_v2_ray.segments import error_record


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
    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@register_actor("v2_stage_1")
@ray.remote
class GpuPipelineActor(PipelineActor):
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

        # Serialize the shared GPU stages. Standardization does NOT need a
        # lock here: ffmpeg decode/probe each run in their own subprocess and
        # normalize() only touches local arrays, and the one genuinely
        # thread-unsafe piece (Silero VAD inference during long-file split)
        # already serializes itself via SileroVAD._GLOBAL_LOCK.
        self._gpu_lock = threading.Lock()
        logger.info(
            f"ray_actor_ready gpu {self._gpu_name} profile {self._profile_name}"
        )

    def process_file(self, audio_path: str, output_folder: str,
                     relative_path: str, shard: str, payload=None) -> dict:
        """Process one file; returns a serializable FileResult dict. Runs on a
        Ray worker thread (max_concurrency>1), overlapping its decode/export
        with other files' GPU work. Never raises: any error (including edge
        cases outside the inner per-stage handlers) becomes a failed result, so
        the driver never sees this as an actor-level crash.

        `payload` is unused: stage 1 decodes raw audio from scratch and has no
        upstream per-file context to receive."""
        try:
            return self._process_file_inner(audio_path, output_folder, relative_path, shard)
        except Exception as e:  # noqa: BLE001 - last-resort guard; keep the actor alive
            logger.error(f"ray_process_file_error file {audio_path} err {type(e).__name__}: {e}")
            return FileResult(
                audio_path, success=False, error=f"{type(e).__name__}: {e}"
            ).to_dict()

    def _process_file_inner(self, audio_path: str, output_folder: str,
                            relative_path: str, shard: str) -> dict:
        log_tag = make_extra_tags(audio_file=relative_path, version=PIPELINE_VERSION)
        try:
            # A decode failure means the whole file is unusable -> let the outer
            # guard in process_file turn it into a failed result. Unlocked: see
            # __init__ comment above for why this is thread-safe across
            # concurrently-running files.
            chunk_states = self._pipeline.standardize(
                PipelineState(audio_path=audio_path, relative_path=relative_path,
                              shard=shard, log_tag=log_tag)
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
            out_records = records if success else []
            if success and not out_records:
                # Processed fine but produced zero segments (VAD/segmenter found
                # nothing). Emit ONE sentinel row so this source appears in
                # segments_part parquet: the parquet IS the resume checkpoint, so
                # a file with no rows at all is indistinguishable from "never
                # processed" and every rerun would redo it forever (and a shard
                # made up entirely of such files would flush no parquet at all,
                # resetting that whole shard's resume state). Applies to both the
                # pipeline_v2_ray and pipeline_v3 drivers, which share this actor.
                out_records = [error_record(relative_path, shard, NO_SEGMENTS_MARKER)]
                logger.info(f"ray_zero_segments file {relative_path}", extra=log_tag)
            return FileResult(
                audio_path, success=success, n_segments=n_segments,
                error=error, segments=out_records,
            ).to_dict()
        finally:
            # Reclaim this file's cached GPU blocks (mirrors PipelineV2.run's
            # finally in the non-ray path) so a long-lived actor doesn't
            # accumulate VRAM. Under the GPU lock so it doesn't sync the device
            # while another concurrent file is mid-GPU-stage.
            with self._gpu_lock:
                torch.cuda.empty_cache()
                # Drop a dead resident DiariZen worker handle promptly so the
                # next file respawns instead of discovering the corpse mid-
                # chunk. Deliberately NOT a close(): that would defeat the
                # point of keeping the model loaded across files.
                try:
                    if self._pipeline.diarizer.reap_if_dead():
                        logger.info("ray_diarizen_worker_reaped", extra=log_tag)
                except Exception as e:  # noqa: BLE001 - never fail a file on this
                    logger.error(f"ray_diarizen_reap_error {type(e).__name__}: {e}")
