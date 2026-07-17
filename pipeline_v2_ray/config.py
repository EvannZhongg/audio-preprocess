"""Parse configs/pipeline_v2_ray.yaml into typed, frozen config objects and resolve the
hardware profile for the GPU an actor lands on.

The yaml is a routing table only: it maps a GPU name to a pipeline config JSON
and carries actor lifecycle / backpressure knobs. It does NOT extend
PipelineParams -- the matched JSON is loaded through the existing
`PipelineParams.from_config`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from pipeline_v2.params import PipelineParams


# Custom Ray resource each worker declares to cap how many pipeline actors it
# hosts, e.g. `ray start --resources='{"pipe_slot": 3}'`. Strong machines
# declare more slots, weak ones fewer -- concurrency is decided per-machine,
# not by GPU type. See configs/pipeline_v2_ray.yaml for the rationale.
PIPE_SLOT_RESOURCE = "pipe_slot"

# Tiny GPU reservation per actor: only large enough to make Ray set
# CUDA_VISIBLE_DEVICES (so the actor sees its card as cuda:0). The real
# per-machine concurrency gate is PIPE_SLOT_RESOURCE, not this. Kept small so
# many actors can co-locate on one physical GPU without exhausting Ray's GPU
# accounting.
GPU_FRACTION_PER_ACTOR = 0.01


@dataclass(frozen=True)
class Defaults:
    max_files_per_actor: int
    max_age_seconds: int
    max_concurrency: int   # concurrent files per actor; also the actor's Ray max_concurrency
                           # and the driver's in-flight depth per actor.


@dataclass(frozen=True)
class HardwareProfile:
    name: str                 # profile key, e.g. "v100"
    match: str                # substring matched against torch.cuda.get_device_name(0)
    pipeline_config: str      # absolute path to the pipeline config JSON (head node only)
    params: PipelineParams    # pre-resolved on the head; shipped to actors verbatim


@dataclass(frozen=True)
class RayConfig:
    defaults: Defaults
    hardware: dict[str, HardwareProfile]

    def match_profile(self, gpu_name: str) -> HardwareProfile:
        """Return the first profile whose `match` is a (case-insensitive)
        substring of `gpu_name`. Raises if none match so an actor fails fast
        at startup rather than silently running the wrong config."""
        needle = gpu_name.lower()
        for profile in self.hardware.values():
            if profile.match.lower() in needle:
                return profile
        tried = [p.match for p in self.hardware.values()]
        raise ProfileNotFoundError(gpu_name, tried)


    def resolve_params(self, gpu_name: str) -> tuple[HardwareProfile, PipelineParams]:
        """Match the actor's GPU to a profile and return its pre-resolved
        PipelineParams. No file IO: params were parsed on the head node and
        shipped inside this (serialized) RayConfig, so worker nodes need
        neither the yaml nor the config_for_*.json files."""
        profile = self.match_profile(gpu_name)
        return profile, profile.params


class ProfileNotFoundError(RuntimeError):
    def __init__(self, gpu_name: str, tried: list[str]) -> None:
        super().__init__(
            f"no hardware profile matched GPU '{gpu_name}'; tried matches {tried}"
        )
        self.gpu_name = gpu_name
        self.tried = tried


def load_ray_config(path: str) -> RayConfig:
    """Load and validate configs/pipeline_v2_ray.yaml on the head node.

    Every profile's pipeline_config JSON is parsed into a PipelineParams here
    (pinned to cuda:0) and carried inside the returned RayConfig. This means a
    misconfigured JSON fails fast on the head at startup, and worker actors
    receive fully-resolved params without touching any config file.

    `pipeline_config` paths in the yaml are resolved relative to the repo root
    (the yaml's grandparent, since the yaml lives in configs/), so the config
    is relocatable and works regardless of the process cwd.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    d = raw["defaults"]
    defaults = Defaults(
        max_files_per_actor=int(d["max_files_per_actor"]),
        max_age_seconds=int(d["max_age_seconds"]),
        max_concurrency=int(d["max_concurrency"]),
    )

    # the yaml lives in configs/, pipeline_config paths are repo-root relative.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))

    hardware: dict[str, HardwareProfile] = {}
    for name, spec in raw["hardware"].items():
        cfg_path = spec["pipeline_config"]
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(repo_root, cfg_path)
        # Parse + validate on the head, pinned to cuda:0 (Ray sets
        # CUDA_VISIBLE_DEVICES so the actor's GPU is always index 0).
        params = PipelineParams.from_config(cfg_path).model_copy(
            update={"device_name": "cuda:0"}
        )
        hardware[name] = HardwareProfile(
            name=name,
            match=spec["match"],
            pipeline_config=cfg_path,
            params=params,
        )

    if not hardware:
        raise ValueError(f"ray config {path} has no hardware profiles")

    return RayConfig(defaults=defaults, hardware=hardware)
