"""ClusterDriver: size the actor pool from per-machine `pipe_slot` resources,
dispatch files with per-actor concurrency, and continuously reconcile the pool
against the (elastic) cluster so actors track machines joining and leaving.

Actor count target = total `pipe_slot` declared across the live cluster. Each
worker sets its own slot count at `ray start` time, so strong machines run more
actors than weak ones -- without the driver needing to know GPU types. Each
actor reserves a tiny GPU fraction only to get CUDA_VISIBLE_DEVICES set;
`pipe_slot` is the concurrency gate across machines.

Within one actor, up to `max_concurrency` files are in flight at once (the
actor runs them on separate threads and overlaps their decode/export with each
other's GPU work).

Elasticity: every RECONCILE_INTERVAL the driver re-reads the cluster's pipe_slot
total and grows or shrinks the pool:
  * grow  -> spawn actors (they land on freshly joined machines);
  * shrink -> mark surplus actors `draining` (no new work) so they retire
              gracefully once their in-flight files finish.
A machine reclaimed out from under us doesn't wait for reconcile: its actors
raise RayActorError, their in-flight files are marked failed (never retried --
a poison file would just crash the replacement), and the slot total drops so
reconcile simply stops replacing them.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.exceptions import RayActorError

import logger
from pipeline_v2_ray.actors.base import new_actor
from pipeline_v2_ray.config import (GPU_FRACTION_PER_ACTOR, PIPE_SLOT_RESOURCE,
                                    RayConfig)
from pipeline_v2_ray.result import FileResult
from pipeline_v2_ray.segments import (SEG_SHARD_SIZE, error_record,
                                       resume_state, write_segments_shard)

RECONCILE_INTERVAL = 60.0   # seconds between cluster-size reconciliations
WAIT_TIMEOUT = 5.0          # ray.wait poll timeout; also bounds reconcile latency
PROGRESS_INTERVAL = 30.0    # seconds between progress/throughput log lines


@dataclass
class FileItem:
    """One file to process. shard_name lives on the batch (run_batch arg), not
    per file, since a batch is exactly one manifest shard."""
    audio_path: str      # full filesystem path to decode
    relative_path: str   # path relative to audio root; export id hash + join key
    duration: float = 0.0  # source audio seconds (from manifest); for RTF throughput


@dataclass
class _Progress:
    """Mutable per-shard counters for throughput logging (kept in a dataclass so
    _log_progress is a plain method, not a closure over run_batch locals)."""
    total: int
    t_start: float
    total_secs: float = 0.0  # total source audio seconds in this batch (progress/ETA by duration)
    n_done: int = 0          # files completed (success or failed)
    n_failed: int = 0        # files that failed (subset of n_done)
    seg_total: int = 0       # segments produced (success only)
    audio_secs: float = 0.0  # source audio seconds of completed files (for RTF)
    last_t: float = 0.0      # wallclock of the last progress log
    last_done: int = 0       # n_done at the last progress log


@dataclass
class _Actor:
    handle: ray.actor.ActorHandle
    inflight: dict = field(default_factory=dict)   # ObjectRef -> audio_path
    files_submitted: int = 0
    born_at: float = field(default_factory=time.time)
    draining: bool = False                          # no new work; retire once inflight empty

    def needs_recycle(self, max_files: int, max_age: int) -> bool:
        return (
            self.files_submitted >= max_files
            or (time.time() - self.born_at) >= max_age
        )


class ClusterDriver:
    def __init__(self, ray_config: RayConfig, actor_name: str) -> None:
        self._config = ray_config
        # CLI-selected actor name; resolved to a ray actor handle per spawn via
        # actors.base.new_actor. The driver stays agnostic to the concrete actor.
        self._actor_name = actor_name
        # Actor pool persists across batches (start -> run_batch* -> shutdown),
        # so models are loaded once, not per manifest shard.
        self._actors: list[_Actor] = []
        self._last_reconcile: float = 0.0

    # ------------------------------------------------------------------
    # actor pool
    # ------------------------------------------------------------------
    def _cluster_slots(self) -> int:
        """Current total `pipe_slot` across all live workers. Changes as elastic
        machines join or leave."""
        return int(ray.cluster_resources().get(PIPE_SLOT_RESOURCE, 0))

    def _spawn_actor(self) -> _Actor:
        # new_actor resolves the CLI name to the registered actor and returns a
        # ready handle. Scheduling resources are driver policy, passed in here:
        # num_gpus=0.01 only makes Ray set CUDA_VISIBLE_DEVICES (per-card
        # placement), pipe_slot is the real per-machine concurrency gate.
        handle = new_actor(
            self._actor_name, self._config,
            num_gpus=GPU_FRACTION_PER_ACTOR,
            slot_resource=PIPE_SLOT_RESOURCE,
            max_concurrency=self._config.defaults.max_concurrency,
        )
        # No readiness probe: Ray queues process_file calls behind __init__, so
        # work simply waits for model loading. A fatal init (bad config, no GPU)
        # surfaces as RayActorError on the first ray.get and is handled there.
        return _Actor(handle=handle)

    def _retire(self, actor: _Actor) -> None:
        ray.kill(actor.handle, no_restart=True)

    def _alive(self) -> list[_Actor]:
        """Actors still accepting work (not draining)."""
        return [a for a in self._actors if not a.draining]

    def _drop(self, actor: _Actor) -> None:
        if actor in self._actors:
            self._actors.remove(actor)

    def _reconcile(self) -> None:
        """Grow or shrink the accepting-actor count toward the cluster's current
        pipe_slot total. Shrink is graceful: surplus actors drain. Operates on
        the persistent pool, so it works both at start() and mid-batch."""
        target = self._cluster_slots()
        accepting = self._alive()
        if target > len(accepting):
            for _ in range(target - len(accepting)):
                self._actors.append(self._spawn_actor())
            logger.info(f"ray_reconcile grow to {target} (was {len(accepting)})")
        elif target < len(accepting):
            # Drain the youngest first (least work invested), keeping veterans.
            for actor in sorted(accepting, key=lambda a: a.born_at, reverse=True)[
                : len(accepting) - target
            ]:
                actor.draining = True
            logger.info(f"ray_reconcile shrink to {target} (was {len(accepting)})")

    # ------------------------------------------------------------------
    # lifecycle: start -> run_batch* -> shutdown
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Build the initial actor pool from the current cluster size. Call once
        before run_batch; the pool then persists across batches."""
        if self._cluster_slots() == 0:
            raise RuntimeError(
                f"no '{PIPE_SLOT_RESOURCE}' resources in the cluster at startup; each "
                f"worker must declare it, e.g. ray start --resources='{{\"{PIPE_SLOT_RESOURCE}\": N}}'"
            )
        self._reconcile()
        self._last_reconcile = time.time()

    def shutdown(self) -> None:
        """Retire every actor. Call once after all batches (or in a finally on
        abort). Idempotent."""
        for actor in self._actors:
            try:
                self._retire(actor)
            except Exception:  # noqa: BLE001
                pass
        self._actors.clear()

    def run_batch(self, shard_name: str, items: list[FileItem], output_folder: str) -> list[FileResult]:
        """Dispatch one manifest shard's files and fully drain them before
        returning, so the shard's output is complete at a clean boundary. Uses
        the persistent pool (reconciling elastically as it goes). Segment records
        are buffered and written to <output>/<shard>/segments_part-NNNNN.parquet
        every SEG_SHARD_SIZE rows, with the remainder flushed when the shard
        finishes. Returns this shard's per-file results."""
        concurrency = max(1, self._config.defaults.max_concurrency)
        max_files = self._config.defaults.max_files_per_actor
        max_age = self._config.defaults.max_age_seconds

        # Everything in this batch is one manifest shard -> one output dir.
        shard_out = os.path.join(output_folder, shard_name)
        # Resume: skip files already recorded in this shard's segments_part
        # parquets, and continue part numbering after them (don't overwrite).
        done, seg_part = resume_state(shard_out)
        if done:
            items = [it for it in items if it.relative_path not in done]
            logger.info(
                f"ray_shard_resume shard {shard_name} skip {len(done)} done, "
                f"resume at part {seg_part}, remaining {len(items)}"
            )
        results: list[FileResult] = []
        pending: deque[FileItem] = deque(items)
        ref_owner: dict[ray.ObjectRef, _Actor] = {}
        seg_buffer: list = []

        def flush() -> None:
            nonlocal seg_part
            if not seg_buffer:
                return
            write_segments_shard(seg_buffer, shard_out, seg_part)
            logger.info(f"ray_segments_flush shard {shard_name} part {seg_part} rows {len(seg_buffer)}")
            seg_part += 1
            seg_buffer.clear()

        def fill() -> None:
            """Top every accepting actor up to `concurrency` in-flight files."""
            for actor in self._alive():
                while len(actor.inflight) < concurrency and pending:
                    item = pending.popleft()
                    # export adds audios/jsons + hash-bucket levels beneath shard_out.
                    ref = actor.handle.process_file.remote(
                        item.audio_path, shard_out, item.relative_path, shard_name
                    )
                    actor.inflight[ref] = item
                    actor.files_submitted += 1
                    ref_owner[ref] = actor

        logger.info(f"ray_shard_start shard {shard_name} files {len(pending)}")
        prog = _Progress(
            total=len(pending),
            total_secs=sum(it.duration for it in pending),  # audio seconds to process this batch
            t_start=time.time(), last_t=time.time(),
        )
        fill()

        while pending or ref_owner:
            # Periodic elasticity check (also runs when idle-waiting for slots).
            if time.time() - self._last_reconcile >= RECONCILE_INTERVAL:
                self._reconcile()
                fill()
                self._last_reconcile = time.time()

            # Periodic progress / throughput line.
            if time.time() - prog.last_t >= PROGRESS_INTERVAL:
                self._log_progress(shard_name, prog)

            if not ref_owner:
                # No work in flight but files remain -> cluster has no slots
                # right now (all machines reclaimed). Wait for reconcile to
                # find new ones rather than busy-spin.
                time.sleep(1.0)
                continue

            ready, _ = ray.wait(list(ref_owner), num_returns=1, timeout=WAIT_TIMEOUT)
            if not ready:
                continue
            ref = ready[0]
            actor = ref_owner.pop(ref)
            item = actor.inflight.pop(ref)
            path = item.audio_path

            try:
                fr = FileResult.from_dict(ray.get(ref))
            except RayActorError as e:
                # The actor/machine itself died (crash, OOM, reclaimed). It takes
                # all its in-flight files down. We do NOT retry them (a poison
                # file would just crash the replacement too); mark them failed.
                # Replacement is left to reconcile.
                logger.error(f"ray_actor_crash file {path} err {e}")
                results.append(FileResult(path, success=False, error=f"actor crashed: {e}"))
                seg_buffer.append(error_record(item.relative_path, shard_name, f"actor crashed: {e}"))
                prog.n_done += 1
                prog.n_failed += 1
                prog.audio_secs += item.duration
                for lost_ref, lost_item in actor.inflight.items():
                    results.append(FileResult(lost_item.audio_path, success=False, error="actor crashed"))
                    seg_buffer.append(error_record(lost_item.relative_path, shard_name, "actor crashed"))
                    prog.n_done += 1
                    prog.n_failed += 1
                    prog.audio_secs += lost_item.duration
                    ref_owner.pop(lost_ref, None)
                actor.inflight.clear()
                self._drop(actor)
                continue
            except Exception as e:  # noqa: BLE001
                # This one task errored but the actor is still alive. Fail just
                # this file; keep the actor and its other in-flight work.
                logger.error(f"ray_task_error file {path} err {type(e).__name__}: {e}")
                results.append(FileResult(path, success=False, error=f"task error: {e}"))
                seg_buffer.append(error_record(item.relative_path, shard_name, f"task error: {e}"))
                prog.n_done += 1
                prog.n_failed += 1
                prog.audio_secs += item.duration
                fill()
                continue

            results.append(self._log_result(fr))
            prog.n_done += 1
            prog.audio_secs += item.duration
            # Success -> its segment rows (already complete, incl. shard);
            # failure -> one placeholder error row (recorded so resume won't
            # retry it forever).
            if fr.success:
                seg_buffer.extend(fr.segments)
                prog.seg_total += len(fr.segments)
            else:
                prog.n_failed += 1
                seg_buffer.append(error_record(item.relative_path, shard_name, fr.error))
            if len(seg_buffer) >= SEG_SHARD_SIZE:
                flush()

            if not actor.draining and actor.needs_recycle(max_files, max_age):
                logger.info(
                    f"ray_actor_recycle files {actor.files_submitted} "
                    f"age {int(time.time() - actor.born_at)}s"
                )
                actor.draining = True

            # A draining actor (recycled or shrunk) with no work left retires.
            if actor.draining and not actor.inflight:
                self._retire(actor)
                self._drop(actor)

            fill()

        # Shard fully drained -> flush its trailing (< SEG_SHARD_SIZE) segments,
        # then retire any actors left draining with no work (avoid zombies
        # lingering across batches).
        flush()
        for actor in [a for a in self._actors if a.draining and not a.inflight]:
            self._retire(actor)
            self._drop(actor)
        self._log_progress(shard_name, prog, tag="ray_shard_done")
        return results

    @staticmethod
    def _log_progress(shard_name: str, p: "_Progress", tag: str = "ray_progress") -> None:
        """Emit a progress + throughput line. Progress and ETA are by audio
        DURATION (audio_secs / total_secs), which is more accurate than file
        count when file lengths vary. Throughput = audio seconds processed per
        wallclock second, i.e. 'N times realtime'."""
        now = time.time()
        elapsed = now - p.t_start
        cum = p.n_done / elapsed if elapsed > 0 else 0.0            # files/s since start
        win_dt = now - p.last_t
        win = (p.n_done - p.last_done) / win_dt if win_dt > 0 else 0.0  # files/s, recent window
        throughput = p.audio_secs / elapsed if elapsed > 0 else 0.0  # audio-s per wallclock-s (x realtime)
        audio_hours = p.audio_secs / 3600.0                          # audio hours processed so far
        total_hours = p.total_secs / 3600.0                          # audio hours in this batch
        hours_per_day = throughput * 24.0                            # audio-hours per wallclock-day
        pct = (100.0 * p.audio_secs / p.total_secs) if p.total_secs > 0 else 100.0
        # ETA by remaining audio duration / current throughput (wallclock secs -> days).
        eta_days = ((p.total_secs - p.audio_secs) / throughput / 86400.0) if throughput > 0 else 0.0
        logger.info(
            f"{tag} shard {shard_name} {p.n_done}/{p.total} files failed {p.n_failed} "
            f"audio {audio_hours:.1f}h/{total_hours:.1f}h ({pct:.1f}%) "
            f"{cum:.2f} files/s (now {win:.2f}) throughput {throughput:.1f}x ({hours_per_day:.0f}h/day) "
            f"segs {p.seg_total} eta {eta_days:.2f}day elapsed {elapsed:.0f}s"
        )
        p.last_t = now
        p.last_done = p.n_done

    @staticmethod
    def _log_result(r: FileResult) -> FileResult:
        name = r.audio_path.rsplit("/", 1)[-1]
        if r.success:
            logger.info(f"ray_file_done file {name} segments {r.n_segments}")
        else:
            logger.error(f"ray_file_failed file {name} err {r.error}")
        return r
