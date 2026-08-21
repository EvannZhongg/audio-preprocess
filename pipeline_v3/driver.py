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

Shards are PIPELINED, not serialized: the first stage opens the next manifest
shard as soon as its own queue runs low, without waiting for any downstream
stage to drain the previous shard. This matters because the first stage
usually owns most of the cluster (it is the cheap, wide stage) and an idle
pool is not free -- it keeps its GPU fraction and `slot_*` reservation, and
the platform reclaims resources that sit at low load. So the invariant here
is: the first stage is never made to wait for a downstream stage; and once
its input is provably exhausted, its whole pool is released immediately
(ActorPool.retire_all) rather than left idling.

Consequently there is NO back-pressure on the hand-off queues: a slow
downstream stage is allowed to fall arbitrarily far behind. That is safe
because a stage's output is a final artifact (chunk wavs are never deleted),
so running ahead does not raise peak disk usage -- it only reaches it sooner,
and the upstream stage's flushed parquet is itself a durable queue that a
restart can rebuild from (see StageDef.load_output_from_disk). Only in-memory
backlog grows, which is logged (`ray_v3_backlog`) rather than throttled.

Per-shard state is fully keyed by (stage, shard): each pair keeps its own
resume set, segment buffer, `segments_part-*` counter and progress, since
output directories are per shard. A (stage, shard) pair is closed out --
flushed and logged as `ray_v3_shard_done` -- as soon as it has no pending or
in-flight work AND its upstream stage's same shard is already closed (i.e.
nothing can stream in anymore).
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import ray
from ray.exceptions import RayActorError

import logger
from pipeline_v2_ray.result import FileResult
from pipeline_v3.config import StageRuntimeConfig
from pipeline_v3.pool import Actor, ActorPool
from pipeline_v3.stages import STAGE_REGISTRY, StageDef
from pipeline_v3.types import FileItem

RECONCILE_INTERVAL = 60.0   # seconds between cluster-size reconciliations (all stages)
WAIT_TIMEOUT = 5.0          # ray.wait poll timeout; also bounds reconcile latency
PROGRESS_INTERVAL = 30.0    # seconds between progress/throughput log lines, per (stage, shard)
FLUSH_INTERVAL = 300.0      # seconds between time-based segment flushes, per (stage, shard)
BACKLOG_INTERVAL = 60.0     # seconds between cross-stage queue-depth log lines

__all__ = ["MultiStagePipelineRunner", "StageTotals"]


@dataclass
class StageTotals:
    """Whole-run counters for one stage. Deliberately just counts: keeping
    every FileResult (each carrying all of its segment rows) alive for the
    length of a run is what used to make the driver's memory grow without
    bound."""
    files: int = 0
    ok: int = 0
    failed: int = 0


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
    """One selected stage's live state for ONE shard. Several shards of the
    same stage can be open at once (shard pipelining), so everything that is
    scoped to an output directory -- the resume set, the segment buffer and
    its part counter -- lives here rather than on the stage."""
    key: str
    shard: str
    shard_out: str
    stage_def: StageDef
    pool: ActorPool
    concurrency: int
    prog: _Progress
    done_keys: set = field(default_factory=set)     # already-processed relative_paths (resume)
    pending: deque = field(default_factory=deque)
    ref_owner: dict = field(default_factory=dict)   # ObjectRef -> Actor, this (stage, shard) only
    seg_buffer: list = field(default_factory=list)
    seg_part: int = 0
    last_flush: float = 0.0


class MultiStagePipelineRunner:
    """`stage_cfgs` must already be filtered to the stages selected for this
    run (and kept in pipeline order); each gets its own persistent ActorPool
    for the whole process lifetime (start -> run -> shutdown)."""

    def __init__(self, stage_cfgs: list[StageRuntimeConfig]) -> None:
        self._order = [c.key for c in stage_cfgs]
        self._cfg = {c.key: c for c in stage_cfgs}
        self._pools = {c.key: ActorPool(c) for c in stage_cfgs}
        self._concurrency = {
            c.key: max(1, c.ray_config.defaults.max_concurrency) for c in stage_cfgs
        }
        # downstream_key -> upstream_key, over the FULL pipeline: tells us
        # whether a shard can still receive streamed items.
        self._upstream = {
            sdef.next_stage: key
            for key, sdef in STAGE_REGISTRY.items()
            if sdef.next_stage
        }
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

    # ------------------------------------------------------------------
    # main scheduling loop
    # ------------------------------------------------------------------
    def run(self, groups: list[tuple[str, list[FileItem]]], output_folder: str,
            seed_fn: Optional[Callable[[str], dict[str, list[FileItem]]]] = None,
            ) -> dict[str, StageTotals]:
        """Drain every shard in `groups` across every selected stage.

        `groups`: [(shard_name, [FileItem, ...]), ...] in shard order, feeding
        the FIRST selected stage (from the manifest, or from an upstream
        stage's disk output when the run starts mid-pipeline).

        `seed_fn(shard_name) -> {stage_key: [FileItem, ...]}`: optional extra
        seed for the NON-first selected stages of that shard, read from disk
        when the shard is opened. Covers resuming a multi-stage run where an
        earlier stage got ahead (and flushed) before a crash, ahead of this
        run's live streaming hand-off.

        Shards are opened as the first stage needs them and closed
        independently per stage, so the first stage never blocks on a slower
        downstream stage. Returns stage_key -> StageTotals.
        """
        first = self._order[0]
        unopened = deque(groups)
        states: dict[tuple[str, str], _StageShardState] = {}
        # per stage, the shards currently open, oldest first -- fill() drains
        # them in this order so old shards close out (and stop pinning their
        # segment buffer) before newer ones get attention.
        open_shards: dict[str, list[str]] = {key: [] for key in self._order}
        ref_index: dict[ray.ObjectRef, tuple[str, str]] = {}  # ObjectRef -> (stage_key, shard)
        totals = {key: StageTotals() for key in self._order}
        last_backlog = time.time()

        def open_shard() -> None:
            """Create every selected stage's state for the next shard."""
            shard, items = unopened.popleft()
            shard_out = os.path.join(output_folder, shard)
            extra = seed_fn(shard) if seed_fn is not None else {}
            for key in self._order:
                sdef = STAGE_REGISTRY[key]
                done, seg_part = sdef.segment_resumer(shard_out)
                seed = list(items) if key == first else list(extra.get(key, []))
                if done:
                    before = len(seed)
                    seed = [it for it in seed if it.relative_path not in done]
                    if before != len(seed):
                        logger.info(
                            f"ray_v3_shard_resume shard {shard} stage {key} "
                            f"skip {before - len(seed)} done, resume at part {seg_part}, "
                            f"remaining {len(seed)}"
                        )
                states[(key, shard)] = _StageShardState(
                    key=key, shard=shard, shard_out=shard_out, stage_def=sdef,
                    pool=self._pools[key], concurrency=self._concurrency[key],
                    done_keys=done or set(), pending=deque(seed), seg_part=seg_part,
                    prog=_Progress(
                        total=len(seed), t_start=time.time(), last_t=time.time(),
                        total_secs=sum(it.duration for it in seed),
                    ),
                    last_flush=time.time(),
                )
                open_shards[key].append(shard)
                logger.info(f"ray_v3_shard_start shard {shard} stage {key} files {len(seed)}")

        def first_stage_hungry() -> bool:
            """True when the first stage's queue no longer covers one full
            round of submissions, i.e. it is about to run dry and should pull
            in the next shard. Deliberately ignores every downstream queue --
            the first stage must never idle waiting for them."""
            pool = self._pools[first]
            if pool.finished:
                return False
            pending = sum(
                len(states[(first, s)].pending)
                for s in open_shards[first]
                if (first, s) in states
            )
            return pending < max(1, len(pool.alive()) * self._concurrency[first])

        def fill() -> None:
            """Top every accepting actor, in every stage, up to its stage's
            concurrency, taking from that stage's oldest open shard first --
            new arrivals (streamed from an upstream stage, or a freshly opened
            shard) are picked up the next time fill() runs."""
            for key in self._order:
                pool = self._pools[key]
                if pool.finished:
                    continue
                conc = self._concurrency[key]
                for actor in pool.alive():
                    while len(actor.inflight) < conc:
                        st = next(
                            (states[(key, s)] for s in open_shards[key]
                             if (key, s) in states and states[(key, s)].pending),
                            None,
                        )
                        if st is None:
                            break
                        item = st.pending.popleft()
                        ref = actor.handle.process_file.remote(
                            item.audio_path, st.shard_out, item.relative_path,
                            st.shard, item.payload,
                        )
                        actor.inflight[ref] = item
                        actor.files_submitted += 1
                        st.ref_owner[ref] = actor
                        ref_index[ref] = (key, st.shard)

        def flush(st: _StageShardState) -> None:
            st.last_flush = time.time()
            if not st.seg_buffer:
                return
            st.stage_def.segment_writer(st.seg_buffer, st.shard_out, st.seg_part)
            logger.info(
                f"ray_v3_segments_flush shard {st.shard} stage {st.key} "
                f"part {st.seg_part} rows {len(st.seg_buffer)}"
            )
            st.seg_part += 1
            st.seg_buffer.clear()

        def close_finished_shards() -> None:
            """Close out every (stage, shard) that is drained AND can no longer
            receive streamed items, i.e. whose upstream stage's same shard is
            already closed. Walks stages in pipeline order so an upstream
            closing frees its downstream in the same pass."""
            for key in self._order:
                upstream = self._upstream.get(key)
                for shard in list(open_shards[key]):
                    st = states.get((key, shard))
                    if st is None or st.pending or st.ref_owner:
                        continue
                    if upstream in self._pools and (upstream, shard) in states:
                        continue  # upstream still running this shard -> more may arrive
                    flush(st)
                    self._log_progress(shard, key, st.prog, tag="ray_v3_shard_done")
                    del states[(key, shard)]
                    open_shards[key].remove(shard)

        def release_exhausted_first_stage() -> None:
            """Every shard opened and the first stage done with all of them ->
            it will never get another file. Hand its GPUs and slots back now
            instead of holding an idle pool (which the platform would reclaim
            out from under us anyway)."""
            pool = self._pools[first]
            if pool.finished or unopened or open_shards[first]:
                return
            n = pool.retire_all()
            logger.info(
                f"ray_v3_stage_exhausted stage {first} released {n} actors "
                f"(input drained; downstream stages continue)"
            )

        while unopened or states or ref_index:
            close_finished_shards()
            release_exhausted_first_stage()
            while unopened and first_stage_hungry():
                open_shard()

            if time.time() - self._last_reconcile >= RECONCILE_INTERVAL:
                self._reconcile_all()
                self._last_reconcile = time.time()

            fill()

            now = time.time()
            for (key, shard), st in list(states.items()):
                # Only actively-running shards report progress; a stage can
                # hold dozens of queued-but-untouched shards while it catches
                # up, and those have nothing new to say.
                if st.ref_owner and now - st.prog.last_t >= PROGRESS_INTERVAL:
                    self._log_progress(shard, key, st.prog)
                if st.seg_buffer and now - st.last_flush >= FLUSH_INTERVAL:
                    flush(st)
            if now - last_backlog >= BACKLOG_INTERVAL:
                self._log_backlog(states, open_shards, len(unopened))
                last_backlog = now

            if not ref_index:
                if not unopened and not states:
                    break  # everything closed out on this pass
                # Nothing in flight anywhere but work remains -> no stage has
                # slots right now; wait for reconcile rather than busy-spin.
                time.sleep(1.0)
                continue

            ready, _ = ray.wait(list(ref_index), num_returns=1, timeout=WAIT_TIMEOUT)
            if not ready:
                continue
            ref = ready[0]
            key, shard = ref_index.pop(ref)
            st = states[(key, shard)]
            actor = st.ref_owner.pop(ref)
            item = actor.inflight.pop(ref)
            path = item.audio_path

            try:
                fr = FileResult.from_dict(ray.get(ref))
            except RayActorError as e:
                # The actor/machine itself died. It takes all its in-flight
                # files down. No retry (a poison file would just crash the
                # replacement too); mark them failed. With shards pipelined
                # those files can span SEVERAL shards of this stage, so each
                # is charged to its own (stage, shard) state.
                logger.error(
                    f"ray_v3_actor_crash stage {key} shard {shard} file {path} err {e}"
                )
                self._record_failure(st, totals[key], item, f"actor crashed: {e}")
                for lost_ref, lost_item in list(actor.inflight.items()):
                    lost_key, lost_shard = ref_index.pop(lost_ref, (key, shard))
                    lost_st = states.get((lost_key, lost_shard))
                    if lost_st is None:
                        continue
                    lost_st.ref_owner.pop(lost_ref, None)
                    self._record_failure(
                        lost_st, totals[lost_key], lost_item, "actor crashed"
                    )
                actor.inflight.clear()
                st.pool.drop(actor)
                continue
            except Exception as e:  # noqa: BLE001
                # This one task errored but the actor is still alive. Fail
                # just this file; keep the actor and its other in-flight work.
                logger.error(
                    f"ray_v3_task_error stage {key} shard {shard} file {path} "
                    f"err {type(e).__name__}: {e}"
                )
                self._record_failure(st, totals[key], item, f"task error: {e}")
                continue

            self._log_result(key, fr)
            totals[key].files += 1
            st.prog.n_done += 1
            st.prog.audio_secs += item.duration
            if fr.success:
                totals[key].ok += 1
                st.seg_buffer.extend(fr.segments)
                st.prog.seg_total += len(fr.segments)
                # Stream straight into the next stage's queue for THIS shard,
                # if that stage is also running this invocation -- the core
                # cross-stage overlap: no waiting for this shard's whole
                # `key` stage to finish.
                next_key = st.stage_def.next_stage
                if next_key and (next_key, shard) in states and st.stage_def.to_next_items:
                    nst = states[(next_key, shard)]
                    next_items = st.stage_def.to_next_items(fr, output_folder)
                    if nst.done_keys:
                        # A resumed run may have already processed these
                        # downstream; don't redo them.
                        next_items = [
                            it for it in next_items if it.relative_path not in nst.done_keys
                        ]
                    if next_items:
                        nst.pending.extend(next_items)
                        nst.prog.total += len(next_items)
                        nst.prog.total_secs += sum(it.duration for it in next_items)
            else:
                totals[key].failed += 1
                st.prog.n_failed += 1
                st.seg_buffer.append(
                    st.stage_def.error_record_fn(item.relative_path, shard, fr.error)
                )
            if len(st.seg_buffer) >= st.stage_def.seg_shard_size:
                flush(st)

            self._recycle_if_due(key, st.pool, actor)

        for key in self._order:
            pool = self._pools[key]
            for actor in [a for a in pool.actors if a.draining and not a.inflight]:
                pool.retire(actor)
                pool.drop(actor)
            tot = totals[key]
            logger.info(
                f"ray_v3_stage_totals stage {key} files {tot.files} "
                f"success {tot.ok} failed {tot.failed}"
            )
        return totals

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _record_failure(st: _StageShardState, tot: StageTotals,
                        item: FileItem, error: str) -> None:
        """Book a driver-side failure (crashed actor / task error) against the
        (stage, shard) it belongs to, so its shard's parquet still gets a row
        for the file and resume won't silently skip it."""
        st.seg_buffer.append(st.stage_def.error_record_fn(item.relative_path, st.shard, error))
        st.prog.n_done += 1
        st.prog.n_failed += 1
        st.prog.audio_secs += item.duration
        tot.files += 1
        tot.failed += 1

    def _recycle_if_due(self, key: str, pool: ActorPool, actor: Actor) -> None:
        defaults = self._cfg[key].ray_config.defaults
        if not actor.draining and actor.needs_recycle(
            defaults.max_files_per_actor, defaults.max_age_seconds
        ):
            logger.info(
                f"ray_v3_actor_recycle stage {key} files {actor.files_submitted} "
                f"age {int(time.time() - actor.born_at)}s"
            )
            actor.draining = True
        if actor.draining and not actor.inflight:
            pool.retire(actor)
            pool.drop(actor)

    def _log_backlog(self, states: dict, open_shards: dict, unopened: int) -> None:
        """Queue depth per stage. The hand-off queues are unbounded by design,
        so this is the signal that a downstream stage is falling behind."""
        parts = []
        for key in self._order:
            shards = [states[(key, s)] for s in open_shards[key] if (key, s) in states]
            parts.append(
                f"{key} shards {len(shards)} "
                f"pending {sum(len(st.pending) for st in shards)} "
                f"inflight {sum(len(st.ref_owner) for st in shards)} "
                f"actors {len(self._pools[key].actors)}"
            )
        logger.info(f"ray_v3_backlog unopened_shards {unopened} | " + " | ".join(parts))

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
    def _log_result(stage_key: str, r: FileResult) -> None:
        name = r.audio_path.rsplit("/", 1)[-1]
        if r.success:
            logger.info(f"ray_v3_file_done stage {stage_key} file {name} segments {r.n_segments}")
        else:
            logger.error(f"ray_v3_file_failed stage {stage_key} file {name} err {r.error}")
