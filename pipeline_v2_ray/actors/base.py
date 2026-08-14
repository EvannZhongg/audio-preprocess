"""Actor interface + self-populating registry for the ray pipeline.

Dependency-light (no torch / pipeline import). `ACTOR_REGISTRY` starts empty and
each concrete actor injects itself via the `@register_actor("name")` decorator
when its module is imported -- so base never references a concrete actor (no
circular import, no heavy deps here). The driver stays actor-agnostic: it's just
handed whichever class the CLI selected. Adding a new actor type needs no change
to base or the driver -- decorate it and pass --actor <name>.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class PipelineActor(ABC):
    """Contract every processing actor must satisfy so the driver can dispatch to
    it uniformly."""

    @abstractmethod
    def process_file(self, audio_path: str, output_folder: str,
                     relative_path: str, shard: str, payload=None) -> dict:
        """Process one file; return a serializable FileResult dict. Must NEVER
        raise -- any error becomes a failed FileResult so the driver doesn't see
        an actor-level crash.

        `payload` is opaque, per-file extra context set on FileItem.payload by
        the caller building the batch; stage 1 (raw-audio decode) ignores it,
        stage 2 uses it to carry the stage-1 segments being re-processed."""
        ...


# name -> registered actor class. Populated by @register_actor as actor modules
# are imported (self-registration), so base holds no concrete-actor references.
ACTOR_REGISTRY: dict[str, type] = {}


def register_actor(name: str) -> Callable[[type], type]:
    """Decorator that injects the decorated actor class into ACTOR_REGISTRY under
    `name`. Place it above @ray.remote so the registered value is the ray
    ActorClass the driver instantiates."""
    def deco(cls: type) -> type:
        ACTOR_REGISTRY[name] = cls
        return cls
    return deco


def new_actor(name: str, ray_config, *, num_gpus: float, slot_resource: str,
              max_concurrency: int):
    """Resolve the registered actor by its CLI name and return a ready ray actor
    HANDLE (already .options(...).remote(cfg)'d). The actor's module must already
    be imported so its @register_actor decorator has run. Scheduling resources
    are passed in by the driver so this module stays free of ray/config imports.

    num_cpus is pinned to 0: CPU is not a scheduling gate -- the pipe_slot
    resource alone caps actors per machine."""
    try:
        cls = ACTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown actor '{name}'; choices: {sorted(ACTOR_REGISTRY)}"
        )
    return cls.options(
        num_cpus=0,
        num_gpus=num_gpus,
        resources={slot_resource: 1},
        max_concurrency=max_concurrency,
    ).remote(ray_config)
