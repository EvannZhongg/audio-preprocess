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

import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import ray
from ray.exceptions import RayActorError

import logger
from pipeline_v2_ray.actor import GpuPipelineActor
from pipeline_v2_ray.config import (GPU_FRACTION_PER_ACTOR, PIPE_SLOT_RESOURCE,
                                    RayConfig)
from pipeline_v2_ray.result import FileResult

RECONCILE_INTERVAL = 60.0   # seconds between cluster-size reconciliations
WAIT_TIMEOUT = 5.0          # ray.wait poll timeout; also bounds reconcile latency


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
    def __init__(self, ray_config: RayConfig) -> None:
        self._config = ray_config

    # ------------------------------------------------------------------
    # actor pool
    # ------------------------------------------------------------------
    def _cluster_slots(self) -> int:
        """Current total `pipe_slot` across all live workers. Changes as elastic
        machines join or leave."""
        return int(ray.cluster_resources().get(PIPE_SLOT_RESOURCE, 0))

    def _spawn_actor(self) -> _Actor:
        handle = GpuPipelineActor.options(
            num_cpus=self._config.defaults.cpu_per_actor,
            num_gpus=GPU_FRACTION_PER_ACTOR,
            resources={PIPE_SLOT_RESOURCE: 1},
            max_concurrency=self._config.defaults.max_concurrency,
        ).remote(self._config)
        # No readiness probe: Ray queues process_file calls behind __init__, so
        # work simply waits for model loading. A fatal init (bad config, no GPU)
        # surfaces as RayActorError on the first ray.get and is handled there.
        return _Actor(handle=handle)

    def _retire(self, actor: _Actor) -> None:
        ray.kill(actor.handle, no_restart=True)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, audio_paths: list[str], output_folder: str) -> list[FileResult]:
        """Public entrypoint. Wraps the dispatch loop so that any unexpected
        error (cluster comms failure, actor submission race, etc.) still logs,
        kills every live actor, and returns whatever results completed rather
        than losing them and leaking actors."""
        results: list[FileResult] = []
        actors: list[_Actor] = []
        try:
            self._run(audio_paths, output_folder, results, actors)
        except Exception:  # noqa: BLE001
            logger.error(f"ray_driver_aborted {traceback.format_exc()}")
        finally:
            for actor in actors:
                try:
                    self._retire(actor)
                except Exception:  # noqa: BLE001
                    pass
        n_ok = sum(1 for r in results if r.success)
        logger.info(
            f"ray_driver_done files {len(results)} success {n_ok} "
            f"failed {len(results) - n_ok}"
        )
        return results

    def _run(
        self,
        audio_paths: list[str],
        output_folder: str,
        results: list[FileResult],
        actors: list[_Actor],
    ) -> None:
        concurrency = max(1, self._config.defaults.max_concurrency)
        max_files = self._config.defaults.max_files_per_actor
        max_age = self._config.defaults.max_age_seconds

        pending: deque[str] = deque(audio_paths)
        ref_owner: dict[ray.ObjectRef, _Actor] = {}
        logger.info(
            f"ray_driver_start files {len(audio_paths)} concurrency {concurrency} "
            f"reconcile_interval {RECONCILE_INTERVAL}s output {output_folder}"
        )

        def alive() -> list[_Actor]:
            """Actors still accepting work (not draining)."""
            return [a for a in actors if not a.draining]

        def fill() -> None:
            """Top every accepting actor up to `concurrency` in-flight files."""
            for actor in alive():
                while len(actor.inflight) < concurrency and pending:
                    path = pending.popleft()
                    ref = actor.handle.process_file.remote(path, output_folder)
                    actor.inflight[ref] = path
                    actor.files_submitted += 1
                    ref_owner[ref] = actor

        def reconcile() -> None:
            """Grow or shrink the accepting-actor count toward the cluster's
            current pipe_slot total. Shrink is graceful: surplus actors drain."""
            target = self._cluster_slots()
            accepting = alive()
            if target > len(accepting):
                for _ in range(target - len(accepting)):
                    actors.append(self._spawn_actor())
                logger.info(f"ray_reconcile grow to {target} (was {len(accepting)})")
            elif target < len(accepting):
                # Drain the youngest first (least work invested), keeping veterans.
                for actor in sorted(accepting, key=lambda a: a.born_at, reverse=True)[
                    : len(accepting) - target
                ]:
                    actor.draining = True
                logger.info(f"ray_reconcile shrink to {target} (was {len(accepting)})")

        def drop(actor: _Actor) -> None:
            if actor in actors:
                actors.remove(actor)

        # Initial pool from the current cluster size, then dispatch.
        if self._cluster_slots() == 0:
            raise RuntimeError(
                f"no '{PIPE_SLOT_RESOURCE}' resources in the cluster at startup; each "
                f"worker must declare it, e.g. ray start --resources='{{\"{PIPE_SLOT_RESOURCE}\": N}}'"
            )
        reconcile()
        fill()
        last_reconcile = time.time()

        while pending or ref_owner:
            # Periodic elasticity check (also runs when idle-waiting for slots).
            if time.time() - last_reconcile >= RECONCILE_INTERVAL:
                reconcile()
                fill()
                last_reconcile = time.time()

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
            path = actor.inflight.pop(ref)

            try:
                results.append(self._log_result(FileResult.from_dict(ray.get(ref))))
            except RayActorError as e:
                # The actor/machine itself died (crash, OOM, reclaimed). It
                # takes all its in-flight files down. We do NOT retry them (a
                # poison file would just crash the replacement too); mark them
                # failed. Replacement is left to reconcile, which respects the
                # (possibly shrunk) slot total.
                logger.error(f"ray_actor_crash file {path} err {e}")
                results.append(FileResult(path, success=False, error=f"actor crashed: {e}"))
                for lost_ref, lost_path in actor.inflight.items():
                    results.append(FileResult(lost_path, success=False, error="actor crashed"))
                    ref_owner.pop(lost_ref, None)
                actor.inflight.clear()
                drop(actor)
                continue
            except Exception as e:  # noqa: BLE001
                # This one task errored (unexpected exception escaping the
                # actor method, or a malformed result) but the actor is still
                # alive. Fail just this file; keep the actor and its other
                # in-flight work.
                logger.error(f"ray_task_error file {path} err {type(e).__name__}: {e}")
                results.append(FileResult(path, success=False, error=f"task error: {e}"))
                fill()
                continue

            if not actor.draining and actor.needs_recycle(max_files, max_age):
                logger.info(
                    f"ray_actor_recycle files {actor.files_submitted} "
                    f"age {int(time.time() - actor.born_at)}s"
                )
                actor.draining = True

            # A draining actor (recycled or shrunk) with no work left retires.
            if actor.draining and not actor.inflight:
                self._retire(actor)
                drop(actor)

            fill()

        # Normal completion: `run`'s finally retires all actors and logs the
        # summary. Leaving actors in place here lets that single path own both
        # normal and aborted cleanup.

    @staticmethod
    def _log_result(r: FileResult) -> FileResult:
        name = r.audio_path.rsplit("/", 1)[-1]
        if r.success:
            logger.info(f"ray_file_done file {name} segments {r.n_segments}")
        else:
            logger.error(f"ray_file_failed file {name} err {r.error}")
        return r
