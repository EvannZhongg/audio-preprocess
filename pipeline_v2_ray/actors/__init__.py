"""Processing actors for the ray pipeline.

Importing this package registers every actor: `base` holds the interface +
registry + resolver, and each concrete actor module self-registers via
`@register_actor` when imported below. So `import pipeline_v2_ray.actors`
populates ACTOR_REGISTRY with all actors at once (the CLI reads the choices from
it). The driver imports from `.base` directly, staying free of the concrete
actors' heavy deps.

Add a new actor: drop a module in this package and add a `from . import <mod>`
line here -- nothing else changes.
"""
from pipeline_v2_ray.actors.base import (ACTOR_REGISTRY, PipelineActor,
                                         new_actor, register_actor)

# Side-effect imports: each registers its actor(s) into ACTOR_REGISTRY.
from pipeline_v2_ray.actors import v2_stage_1  # noqa: F401,E402
from pipeline_v2_ray.actors import v2_stage_2  # noqa: F401,E402

__all__ = ["ACTOR_REGISTRY", "PipelineActor", "new_actor", "register_actor"]
