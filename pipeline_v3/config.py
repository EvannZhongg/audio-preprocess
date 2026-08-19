"""Parse configs/pipeline_v3.yaml into typed, frozen per-stage config objects.

Generalizes pipeline_v2_ray/config.py's duplicated (RayConfig, Stage2RayConfig)
pair into one stage-agnostic `StageRayConfig` (params typed as `object`), so
every stage -- regardless of which Params class (PipelineParams,
Stage2Params, ...) it resolves to -- shares one config shape and the
scheduler in pipeline_v3/driver.py never needs to know which stage it's
looking at.

The yaml has one `stages:` mapping, in pipeline order (stage_1, stage_2, ...
extend by adding more keys). Each stage entry is a routing table only -- it
maps a GPU name substring to a pipeline config JSON and carries actor
lifecycle / backpressure knobs -- exactly like pipeline_v2_ray.yaml's
top-level shape, just nested one level under its stage key. It does NOT
encode which Params class or actor-registration semantics a stage uses --
that's `pipeline_v3.stages.STAGE_REGISTRY`'s job (see stages.py); yaml and
STAGE_REGISTRY must agree on stage keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Type

import yaml

from pipeline_v2_ray.config import ProfileNotFoundError

__all__ = [
    "Defaults", "HardwareProfile", "StageRayConfig", "StageRuntimeConfig",
    "default_resource_name", "load_pipeline_v3_config",
]


@dataclass(frozen=True)
class Defaults:
    max_files_per_actor: int
    max_age_seconds: int
    max_concurrency: int   # concurrent files per actor; also the actor's Ray max_concurrency
                           # and the scheduler's in-flight depth per actor.


@dataclass(frozen=True)
class HardwareProfile:
    name: str                 # profile key, e.g. "v100"
    match: str                # substring matched against torch.cuda.get_device_name(0)
    pipeline_config: str      # absolute path to the pipeline config JSON (head node only)
    params: object            # pre-resolved PipelineParams/Stage2Params/...; shipped to actors verbatim


@dataclass(frozen=True)
class StageRayConfig:
    """Same routing-table shape/behavior as pipeline_v2_ray.config.RayConfig
    and Stage2RayConfig, merged into one type generic over the params class
    (kept as `object` rather than a TypeVar since it crosses the Ray
    serialization boundary -- actors just call `.resolve_params(gpu_name)`
    and use whatever params object comes back)."""
    defaults: Defaults
    hardware: dict[str, HardwareProfile]

    def match_profile(self, gpu_name: str) -> HardwareProfile:
        needle = gpu_name.lower()
        for profile in self.hardware.values():
            if profile.match.lower() in needle:
                return profile
        tried = [p.match for p in self.hardware.values()]
        raise ProfileNotFoundError(gpu_name, tried)

    def resolve_params(self, gpu_name: str) -> tuple[HardwareProfile, object]:
        """Match the actor's GPU to a profile and return its pre-resolved
        params object. No file IO: params were parsed on the head node and
        shipped inside this (serialized) StageRayConfig."""
        profile = self.match_profile(gpu_name)
        return profile, profile.params


def default_resource_name(stage_key: str) -> str:
    """Default Ray custom resource name for a stage when the yaml doesn't
    override it with an explicit `resource:` -- e.g. "stage_1" -> "slot_stage_1"."""
    return f"slot_{stage_key}"


@dataclass(frozen=True)
class StageRuntimeConfig:
    """Everything the driver needs to run one stage: which actor to spawn
    (already registered in pipeline_v2_ray.actors.ACTOR_REGISTRY), which
    cluster resource gates its concurrency, and its resolved hardware-routing
    config."""
    key: str
    actor_name: str
    resource_name: str
    ray_config: StageRayConfig


def _load_defaults(raw: dict) -> Defaults:
    return Defaults(
        max_files_per_actor=int(raw["max_files_per_actor"]),
        max_age_seconds=int(raw["max_age_seconds"]),
        max_concurrency=int(raw["max_concurrency"]),
    )


def _load_stage_ray_config(raw: dict, repo_root: str, params_cls: Type) -> StageRayConfig:
    defaults = _load_defaults(raw["defaults"])
    hardware: dict[str, HardwareProfile] = {}
    for name, spec in raw["hardware"].items():
        cfg_path = spec["pipeline_config"]
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(repo_root, cfg_path)
        # Parse + validate on the head, pinned to cuda:0 (Ray sets
        # CUDA_VISIBLE_DEVICES so the actor's GPU is always index 0).
        params = params_cls.from_config(cfg_path).model_copy(update={"device_name": "cuda:0"})
        hardware[name] = HardwareProfile(
            name=name, match=spec["match"], pipeline_config=cfg_path, params=params,
        )
    if not hardware:
        raise ValueError("stage ray config has no hardware profiles")
    return StageRayConfig(defaults=defaults, hardware=hardware)


def load_pipeline_v3_config(path: str, params_cls_by_stage: dict[str, Type]) -> list[StageRuntimeConfig]:
    """Load configs/pipeline_v3.yaml on the head node.

    `params_cls_by_stage` (stage_key -> Params class) comes from
    `pipeline_v3.stages.STAGE_REGISTRY`, keeping "which Params class a stage
    uses" defined once, in code, next to that stage's writer/resumer/
    converter -- this loader only cross-checks the yaml agrees on stage keys.

    Returns stage runtime configs in FILE ORDER (the pipeline's stage order,
    e.g. [stage_1, stage_2]); every profile's pipeline_config JSON is parsed
    here so a misconfigured JSON fails fast on the head at startup, and
    worker actors receive fully-resolved params without touching any config
    file or this yaml.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    stages_raw = raw.get("stages")
    if not stages_raw:
        raise ValueError(f"pipeline_v3 config {path} has no 'stages' section")

    # the yaml lives in configs/, pipeline_config paths are repo-root relative.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))

    configs: list[StageRuntimeConfig] = []
    for stage_key, spec in stages_raw.items():
        if stage_key not in params_cls_by_stage:
            raise ValueError(
                f"stage '{stage_key}' in {path} has no registered StageDef "
                f"(pipeline_v3.stages.STAGE_REGISTRY); known: {sorted(params_cls_by_stage)}"
            )
        actor_name = spec["actor"]
        resource_name = spec.get("resource") or default_resource_name(stage_key)
        ray_config = _load_stage_ray_config(spec, repo_root, params_cls_by_stage[stage_key])
        configs.append(StageRuntimeConfig(
            key=stage_key, actor_name=actor_name,
            resource_name=resource_name, ray_config=ray_config,
        ))
    return configs
