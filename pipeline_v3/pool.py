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

    def needs_recycle(self, max_files: int, max_age: int) -> bool:
        return (
            self.files_submitted >= max_files
            or (time.time() - self.born_at) >= max_age
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
        return Actor(handle=handle)

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
