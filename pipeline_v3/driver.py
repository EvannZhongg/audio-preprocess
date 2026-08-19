"""MultiStagePipelineRunner: run several stages' actor pools in ONE process,
streaming a file straight from stage N into stage N+1 the moment stage N
finishes it, instead of draining stage N's whole shard before stage N+1
starts.

Generalizes pipeline_v2_ray.driver.ClusterDriver's single-stage run_batch
into a scheduler over N per-stage ActorPools (pipeline_v3.pool.ActorPool):
one shared `ray.wait` spans every selected stage's in-flight refs, so
whichever stage's file finishes first is handled first, and a stage's
success is fed straight into its `next_stage`'s pending queue (see
pipeline_v3.stages.StageDef.to_next_items) whenever that next stage is also
selected for this run. If it isn't (or a stage is terminal), the result is
just recorded like normal.

Design choice: shards are still processed strictly one at a time, in order
(matching pipeline_v2_ray) -- only the STAGES within one shard overlap. This
keeps actor-pool reuse, resume semantics, and per-shard output directories
exactly as before; only the "stage1 must fully finish before stage2 starts"
constraint is lifted.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.exceptions import RayActorError

import logger
from pipeline_v2_ray.result import FileResult
from pipeline_v3.config import StageRuntimeConfig
from pipeline_v3.pool import ActorPool
from pipeline_v3.stages import STAGE_REGISTRY, StageDef
from pipeline_v3.types import FileItem

RECONCILE_INTERVAL = 60.0   # seconds between cluster-size reconciliations (all stages)
WAIT_TIMEOUT = 5.0          # ray.wait poll timeout; also bounds reconcile latency
PROGRESS_INTERVAL = 30.0    # seconds between progress/throughput log lines, per stage
FLUSH_INTERVAL = 300.0      # seconds between time-based segment flushes, per stage

__all__ = ["MultiStagePipelineRunner"]


@dataclass
class _Progress:
    total: int
    t_start: float
    total_secs: float = 0.0
    n_done: int = 0
    n_failed: int = 0
    seg_total: int = 0
    audio_secs: float = 0.0
    last_t: float = 0.0
    last_done: int = 0


@dataclass
class _StageShardState:
    """One selected stage's live state while draining one shard."""
    key: str
    stage_def: StageDef
    pool: ActorPool
    concurrency: int
    prog: _Progress
    pending: deque = field(default_factory=deque)
    ref_owner: dict = field(default_factory=dict)   # ObjectRef -> Actor, this stage only
    seg_buffer: list = field(default_factory=list)
    seg_part: int = 0
    results: list = field(default_factory=list)
    last_flush: float = 0.0


class MultiStagePipelineRunner:
    """`stage_cfgs` must already be filtered to the stages selected for this
    run (and kept in pipeline order); each gets its own persistent ActorPool
    for the whole process lifetime (start -> run_shard* -> shutdown)."""

    def __init__(self, stage_cfgs: list[StageRuntimeConfig]) -> None:
        self._order = [c.key for c in stage_cfgs]
        self._cfg = {c.key: c for c in stage_cfgs}
        self._pools = {c.key: ActorPool(c) for c in stage_cfgs}
        self._last_reconcile = 0.0

    def start(self) -> None:
        for pool in self._pools.values():
            pool.start()
        self._last_reconcile = time.time()

    def shutdown(self) -> None:
        for pool in self._pools.values():
            pool.shutdown()

    def _reconcile_all(self) -> None:
        for pool in self._pools.values():
            pool.reconcile()

    def run_shard(self, shard_name: str, output_folder: str,
                   seed_items: dict[str, list[FileItem]]) -> dict[str, list[FileResult]]:
        """Drain one shard across every selected stage, streaming a file from
        stage N straight into stage N+1's pending queue as soon as it
        finishes (if stage N+1 is also selected this run). Blocks until every
        selected stage's pending+in-flight work for this shard is empty.

        `seed_items`: stage_key -> initial FileItems (e.g. from the manifest
        for the first selected stage, or scanned from an upstream stage's
        disk output for the rest -- built by the caller, see main_v3_ray.py).
        Returns stage_key -> list[FileResult].
        """
        shard_out = os.path.join(output_folder, shard_name)
        states: dict[str, _StageShardState] = {}
        ref_index: dict[ray.ObjectRef, str] = {}  # ObjectRef -> stage_key, spans every selected stage

        for key in self._order:
            sdef = STAGE_REGISTRY[key]
            cfg = self._cfg[key]
            done, seg_part = sdef.segment_resumer(shard_out)
            items = list(seed_items.get(key, []))
            if done:
                before = len(items)
                items = [it for it in items if it.relative_path not in done]
                if before != len(items):
                    logger.info(
                        f"ray_v3_shard_resume shard {shard_name} stage {key} "
                        f"skip {before - len(items)} done, resume at part {seg_part}, "
                        f"remaining {len(items)}"
                    )
            states[key] = _StageShardState(
                key=key, stage_def=sdef, pool=self._pools[key],
                concurrency=max(1, cfg.ray_config.defaults.max_concurrency),
                pending=deque(items), seg_part=seg_part,
                prog=_Progress(
                    total=len(items), t_start=time.time(), last_t=time.time(),
                    total_secs=sum(it.duration for it in items),
                ),
                last_flush=time.time(),
            )
            logger.info(f"ray_v3_shard_start shard {shard_name} stage {key} files {len(items)}")

        def fill() -> None:
            """Top every accepting actor, in every stage, up to its stage's
            concurrency -- new arrivals (streamed from an upstream stage)
            are picked up the next time fill() runs."""
            for key in self._order:
                st = states[key]
                for actor in st.pool.alive():
                    while len(actor.inflight) < st.concurrency and st.pending:
                        item = st.pending.popleft()
                        ref = actor.handle.process_file.remote(
                            item.audio_path, shard_out, item.relative_path, shard_name, item.payload
                        )
                        actor.inflight[ref] = item
                        actor.files_submitted += 1
                        st.ref_owner[ref] = actor
                        ref_index[ref] = key

        def flush(key: str) -> None:
            st = states[key]
            st.last_flush = time.time()
            if not st.seg_buffer:
                return
            st.stage_def.segment_writer(st.seg_buffer, shard_out, st.seg_part)
            logger.info(
                f"ray_v3_segments_flush shard {shard_name} stage {key} "
                f"part {st.seg_part} rows {len(st.seg_buffer)}"
            )
            st.seg_part += 1
            st.seg_buffer.clear()

        fill()

        while any(st.pending for st in states.values()) or ref_index:
            if time.time() - self._last_reconcile >= RECONCILE_INTERVAL:
                self._reconcile_all()
                fill()
                self._last_reconcile = time.time()

            for key, st in states.items():
                if time.time() - st.prog.last_t >= PROGRESS_INTERVAL:
                    self._log_progress(shard_name, key, st.prog)
                if st.seg_buffer and time.time() - st.last_flush >= FLUSH_INTERVAL:
                    flush(key)

            if not ref_index:
                # No work in flight anywhere but files remain -> no stage has
                # slots right now; wait for reconcile rather than busy-spin.
                time.sleep(1.0)
                continue

            ready, _ = ray.wait(list(ref_index), num_returns=1, timeout=WAIT_TIMEOUT)
            if not ready:
                continue
            ref = ready[0]
            key = ref_index.pop(ref)
            st = states[key]
            actor = st.ref_owner.pop(ref)
            item = actor.inflight.pop(ref)
            path = item.audio_path

            try:
                fr = FileResult.from_dict(ray.get(ref))
            except RayActorError as e:
                # The actor/machine itself died. It takes all its in-flight
                # files down for THIS stage. No retry (a poison file would
                # just crash the replacement too); mark them failed.
                logger.error(f"ray_v3_actor_crash stage {key} file {path} err {e}")
                st.results.append(FileResult(path, success=False, error=f"actor crashed: {e}"))
                st.seg_buffer.append(st.stage_def.error_record_fn(item.relative_path, shard_name, f"actor crashed: {e}"))
                st.prog.n_done += 1
                st.prog.n_failed += 1
                st.prog.audio_secs += item.duration
                for lost_ref, lost_item in list(actor.inflight.items()):
                    st.results.append(FileResult(lost_item.audio_path, success=False, error="actor crashed"))
                    st.seg_buffer.append(st.stage_def.error_record_fn(lost_item.relative_path, shard_name, "actor crashed"))
                    st.prog.n_done += 1
                    st.prog.n_failed += 1
                    st.prog.audio_secs += lost_item.duration
                    ref_index.pop(lost_ref, None)
                    st.ref_owner.pop(lost_ref, None)
                actor.inflight.clear()
                st.pool.drop(actor)
                continue
            except Exception as e:  # noqa: BLE001
                # This one task errored but the actor is still alive. Fail
                # just this file; keep the actor and its other in-flight work.
                logger.error(f"ray_v3_task_error stage {key} file {path} err {type(e).__name__}: {e}")
                st.results.append(FileResult(path, success=False, error=f"task error: {e}"))
                st.seg_buffer.append(st.stage_def.error_record_fn(item.relative_path, shard_name, f"task error: {e}"))
                st.prog.n_done += 1
                st.prog.n_failed += 1
                st.prog.audio_secs += item.duration
                fill()
                continue

            st.results.append(self._log_result(key, fr))
            st.prog.n_done += 1
            st.prog.audio_secs += item.duration
            if fr.success:
                st.seg_buffer.extend(fr.segments)
                st.prog.seg_total += len(fr.segments)
                # Stream straight into the next stage's queue, if it's also
                # running this invocation -- the core cross-stage overlap:
                # no waiting for this shard's whole `key` stage to finish.
                next_key = st.stage_def.next_stage
                if next_key and next_key in states and st.stage_def.to_next_items:
                    next_items = st.stage_def.to_next_items(fr, output_folder)
                    if next_items:
                        nst = states[next_key]
                        nst.pending.extend(next_items)
                        nst.prog.total += len(next_items)
                        nst.prog.total_secs += sum(it.duration for it in next_items)
            else:
                st.prog.n_failed += 1
                st.seg_buffer.append(st.stage_def.error_record_fn(item.relative_path, shard_name, fr.error))
            if len(st.seg_buffer) >= st.stage_def.seg_shard_size:
                flush(key)

            cfg = self._cfg[key]
            if not actor.draining and actor.needs_recycle(
                cfg.ray_config.defaults.max_files_per_actor,
                cfg.ray_config.defaults.max_age_seconds,
            ):
                logger.info(
                    f"ray_v3_actor_recycle stage {key} files {actor.files_submitted} "
                    f"age {int(time.time() - actor.born_at)}s"
                )
                actor.draining = True
            if actor.draining and not actor.inflight:
                st.pool.retire(actor)
                st.pool.drop(actor)

            fill()

        for key, st in states.items():
            flush(key)
            for actor in [a for a in st.pool.actors if a.draining and not a.inflight]:
                st.pool.retire(actor)
                st.pool.drop(actor)
            self._log_progress(shard_name, key, st.prog, tag="ray_v3_shard_done")

        return {key: st.results for key, st in states.items()}

    @staticmethod
    def _log_progress(shard_name: str, stage_key: str, p: _Progress, tag: str = "ray_v3_progress") -> None:
        now = time.time()
        elapsed = now - p.t_start
        cum = p.n_done / elapsed if elapsed > 0 else 0.0
        win_dt = now - p.last_t
        win = (p.n_done - p.last_done) / win_dt if win_dt > 0 else 0.0
        throughput = p.audio_secs / elapsed if elapsed > 0 else 0.0
        audio_hours = p.audio_secs / 3600.0
        total_hours = p.total_secs / 3600.0
        hours_per_day = throughput * 24.0
        pct = (100.0 * p.audio_secs / p.total_secs) if p.total_secs > 0 else 100.0
        eta_days = ((p.total_secs - p.audio_secs) / throughput / 86400.0) if throughput > 0 else 0.0
        logger.info(
            f"{tag} shard {shard_name} stage {stage_key} {p.n_done}/{p.total} files "
            f"failed {p.n_failed} audio {audio_hours:.1f}h/{total_hours:.1f}h ({pct:.1f}%) "
            f"{cum:.2f} files/s (now {win:.2f}) throughput {throughput:.1f}x ({hours_per_day:.0f}h/day) "
            f"segs {p.seg_total} eta {eta_days:.2f}day elapsed {elapsed:.0f}s"
        )
        p.last_t = now
        p.last_done = p.n_done

    @staticmethod
    def _log_result(stage_key: str, r: FileResult) -> FileResult:
        name = r.audio_path.rsplit("/", 1)[-1]
        if r.success:
            logger.info(f"ray_v3_file_done stage {stage_key} file {name} segments {r.n_segments}")
        else:
            logger.error(f"ray_v3_file_failed stage {stage_key} file {name} err {r.error}")
        return r
