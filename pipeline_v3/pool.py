"""ActorPool: elastic pool of persistent Ray actors for ONE pipeline stage.

Extracted from pipeline_v2_ray.driver.ClusterDriver's actor-pool management
(spawn / reconcile / drain / retire) so pipeline_v3 can run several stages'
pools side by side in one process, each gated by its OWN custom Ray resource
(slot_stage_1, slot_stage_2, ... configurable per stage in
configs/pipeline_v3.yaml) instead of the single shared `pipe_slot`
pipeline_v2_ray uses -- this is what lets stage_1 and stage_2 actors be
scheduled/sized independently while still optionally sharing one physical
GPU (each actor only reserves a tiny GPU_FRACTION_PER_ACTOR).

pipeline_v2_ray itself is intentionally left unmodified; this module
duplicates (not imports) the actor-pool logic so the two pipelines evolve
independently.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import ray

import logger
from pipeline_v2_ray.actors.base import new_actor
from pipeline_v2_ray.config import GPU_FRACTION_PER_ACTOR
from pipeline_v3.config import StageRuntimeConfig

__all__ = ["Actor", "ActorPool"]


@dataclass
class Actor:
    handle: ray.actor.ActorHandle
    inflight: dict = field(default_factory=dict)   # ObjectRef -> FileItem
    files_submitted: int = 0
    born_at: float = field(default_factory=time.time)
    draining: bool = False   # no new work; retire once inflight empties
    # Stable short identity. This dataclass is mutable and therefore
    # unhashable, so anything keyed per actor (see pipeline_v3.health) keys on
    # `uid` instead of the object.
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # Where this actor runs; filled in lazily by ActorPool.poll_node_info()
    # once `node_ref` resolves. Needed so slow-node alerts can name the
    # machine instead of an opaque actor id.
    node_ip: str = ""
    node_id: str = ""
    hostname: str = ""
    pid: int = 0
    # Pending PipelineActor.node_info() ObjectRef, or None once resolved/given
    # up on. Typed loosely to keep this dataclass ray-generic.
    node_ref: object = None

    def needs_recycle(self, max_files: int, max_age: int) -> bool:
        return (
            self.files_submitted >= max_files
            or (time.time() - self.born_at) >= max_age
        )

    def describe(self) -> str:
        """Fixed-shape identity for log lines: keeps `node ... host ... pid
        ... actor ...` greppable even before node info has resolved."""
        return (
            f"node {self.node_ip or '?'} host {self.hostname or '?'} "
            f"pid {self.pid or 0} actor {self.uid}"
        )


class ActorPool:
    """Sizes itself off its stage's dedicated cluster resource
    (`stage_cfg.resource_name`), growing/draining as the cluster's total for
    that resource changes -- same elasticity model as
    `pipeline_v2_ray.driver.ClusterDriver`, just parameterized per stage so
    several stages' pools coexist without fighting over one resource name.
    """

    def __init__(self, stage_cfg: StageRuntimeConfig) -> None:
        self.stage_cfg = stage_cfg
        self.actors: list[Actor] = []
        # Set once this stage can never receive work again (see retire_all).
        # Latches the pool at zero so reconcile() won't regrow it.
        self.finished: bool = False

    def cluster_slots(self) -> int:
        """Current total of this stage's custom resource across all live
        workers. Changes as elastic machines join or leave."""
        return int(ray.cluster_resources().get(self.stage_cfg.resource_name, 0))

    def _spawn(self) -> Actor:
        handle = new_actor(
            self.stage_cfg.actor_name, self.stage_cfg.ray_config,
            num_gpus=GPU_FRACTION_PER_ACTOR,
            slot_resource=self.stage_cfg.resource_name,
            max_concurrency=self.stage_cfg.ray_config.defaults.max_concurrency,
        )
        actor = Actor(handle=handle)
        # Fire-and-forget: ask the actor where it lives BEFORE it is given any
        # work, so the call sits right behind its (slow, model-loading)
        # __init__ instead of behind a queue of process_file calls. Never
        # waited on here -- poll_node_info() collects it later.
        try:
            actor.node_ref = handle.node_info.remote()
        except Exception as e:  # noqa: BLE001 - identity is a nice-to-have
            logger.warning(f"ray_v3_node_info_skip stage {self.stage_cfg.key} err {e}")
        return actor

    def poll_node_info(self) -> None:
        """Non-blockingly collect any node_info() results that have arrived.

        MUST NOT block: an actor's __init__ loads models for minutes, so
        `ray.get` here would stall the driver's whole scheduling loop. Uses a
        zero timeout and only reads refs that are already ready."""
        waiting = [a for a in self.actors if a.node_ref is not None]
        if not waiting:
            return
        by_ref = {a.node_ref: a for a in waiting}
        try:
            ready, _ = ray.wait(list(by_ref), num_returns=len(by_ref), timeout=0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_node_info_skip stage {self.stage_cfg.key} err {e}")
            return
        for ref in ready:
            actor = by_ref[ref]
            actor.node_ref = None   # resolved (or given up on): ask only once
            try:
                info = ray.get(ref)
            except Exception as e:  # noqa: BLE001 - actor may have died meanwhile
                logger.warning(
                    f"ray_v3_node_info_failed stage {self.stage_cfg.key} "
                    f"actor {actor.uid} err {e}"
                )
                continue
            actor.node_ip = info.get("node_ip", "") or ""
            actor.node_id = info.get("node_id", "") or ""
            actor.hostname = info.get("hostname", "") or ""
            actor.pid = int(info.get("pid", 0) or 0)
            # Logged once per actor so the actor-uid -> machine mapping used by
            # every later health alert is itself in the log.
            logger.info(
                f"ray_v3_actor_node stage {self.stage_cfg.key} {actor.describe()} "
                f"node_id {actor.node_id}"
            )

    def alive(self) -> list[Actor]:
        """Actors still accepting work (not draining)."""
        return [a for a in self.actors if not a.draining]

    def drop(self, actor: Actor) -> None:
        if actor in self.actors:
            self.actors.remove(actor)

    def retire(self, actor: Actor) -> None:
        ray.kill(actor.handle, no_restart=True)

    def reconcile(self) -> None:
        """Grow or shrink the accepting-actor count toward this stage's
        current resource total. Shrink is graceful: surplus actors drain."""
        if self.finished:
            return  # stage is done for good; never respawn into an idle pool
        target = self.cluster_slots()
        accepting = self.alive()
        if target > len(accepting):
            for _ in range(target - len(accepting)):
                self.actors.append(self._spawn())
            logger.info(
                f"ray_v3_reconcile stage {self.stage_cfg.key} grow to {target} "
                f"(was {len(accepting)})"
            )
        elif target < len(accepting):
            # Drain the youngest first (least work invested), keeping veterans.
            for actor in sorted(accepting, key=lambda a: a.born_at, reverse=True)[
                : len(accepting) - target
            ]:
                actor.draining = True
            logger.info(
                f"ray_v3_reconcile stage {self.stage_cfg.key} shrink to {target} "
                f"(was {len(accepting)})"
            )

    def start(self) -> None:
        """Build the initial actor pool from the current cluster size for
        this stage's resource. Call once before run_shard; the pool then
        persists across shards."""
        if self.cluster_slots() == 0:
            raise RuntimeError(
                f"no '{self.stage_cfg.resource_name}' resources in the cluster "
                f"for stage '{self.stage_cfg.key}'; declare it at ray start, e.g. "
                f"ray start --resources='{{\"{self.stage_cfg.resource_name}\": N}}'"
            )
        self.reconcile()

    def retire_all(self) -> int:
        """Kill every actor NOW and latch the pool closed; returns how many
        were killed.

        For a stage whose input is provably exhausted: an idle pool still
        holds its GPU fraction and its `slot_*` reservation, and the platform
        reclaims long-idle resources anyway -- so release them deliberately
        instead of letting a zero-load pool sit there. `finished` keeps
        reconcile() from growing it back on the next cluster poll. Idempotent.
        """
        self.finished = True
        n = len(self.actors)
        for actor in self.actors:
            try:
                self.retire(actor)
            except Exception:  # noqa: BLE001
                pass
        self.actors.clear()
        return n

    def shutdown(self) -> None:
        """Retire every actor. Call once after all shards (or in a finally on
        abort). Idempotent."""
        self.retire_all()
