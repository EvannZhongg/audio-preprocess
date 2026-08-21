"""QC options plus the production thresholds every verdict is measured against.

Thresholds are NOT hard-coded as "the" values here. Each
`configs/config_for_*.json` carries different numbers (the a10 profile uses
`merge_gap=0.8` where params.py's fallback is `2.0`), so a QC run that
hard-coded a threshold would confidently report wrong conclusions. Instead
`Thresholds.from_config` goes through `pipeline_v2.params.PipelineParams
.from_config` -- literally the same parser stage 1 uses -- so QC verdicts and
production behaviour cannot drift apart.

Only pydantic/stdlib is imported at module level: the parquet-only analyzers
fan out over a process pool and must not pay for torch/librosa imports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants mirrored from production, with the source of truth noted.
#
# These are duplicated rather than imported because the only module that
# defines them (pipeline_v2/steps/embedding_refinement.py) drags in torch,
# librosa and sklearn at import time, which would dominate the runtime of the
# pure-parquet analyzers that need nothing but the numbers.
# qc/models_bundle.py cross-checks them against the real module whenever the
# models are actually loaded, so a future upstream edit surfaces as a loud
# warning instead of a silent mis-classification.
# ---------------------------------------------------------------------------

# embedding_refinement.py `_MIN_SEGMENT_DURATION_S`: segments shorter than this
# never get a speaker embedding, which means Segmenter can never merge them --
# they are *structurally* unmergeable, not "missed merges".
EMBED_MIN_SEGMENT_S = 1.0
# embedding_refinement.py `_MAX_SEGMENT_DURATION_S`: longer spans are dropped
# outright before segmentation.
EMBED_MAX_SEGMENT_S = 30.0
# embedding_refinement.py `_WINDOW_SIZE_S` / `_WINDOW_STEP_S`.
EMBED_WINDOW_S = 1.1
EMBED_WINDOW_STEP_S = 0.4
# dnsmos.py `INPUT_LENGTH`: shorter clips are self-concatenated until they
# reach this length, which biases scores on short segments.
DNSMOS_INPUT_LENGTH_S = 9.01
# brouhaha_metrics.py returns this sentinel pair when scoring throws.
BROUHAHA_FAILURE_SENTINEL = -420.69
# pipeline_v2/state.py `ASR_ACCESS_FAILED_MARKER`: a *retriable* stage-2
# failure (the remote ASR was unreachable), not a property of the audio.
ASR_RETRIABLE_MARKER = "asr_access_failed"

# Fallbacks mirroring pipeline_v2/params.py:144-161, used only when no
# production config was supplied. Reported as `is_default=True` so the report
# can warn that the numbers are not the ones the data was produced with.
_THRESHOLD_FALLBACKS: dict[str, Any] = {
    "merge_gap": 2.0,
    "min_segment_length": 3.0,
    "max_segment_length": 30.0,
    "intra_similarity_threshold": 0.64,
    "inter_similarity_threshold": 0.7,
    "grace_period_start": 0.0,
    "grace_period_end": 0.02,
    "fixed_dnsmos_threshold": 3.0,
    "fixed_c50_threshold": 40.0,
    "fixed_snr_threshold": 40.0,
    "metrics_strategy": "average",
    "use_brouhaha": False,
}

ALL_STEPS = ("yield", "duration", "merge", "speaker", "background")
# Steps that need audio decoded and models run; they all share one GPU pass.
GPU_STEPS = ("speaker", "background", "merge")


def _load_pipeline_params_cls():
    """Get production's `PipelineParams` without pulling in torch.

    `import pipeline_v2.params` executes `pipeline_v2/__init__.py`, which
    imports `PipelineV2` and therefore torch, pyannote and friends. That is
    fine inside the pipeline, but it would make the parquet-only QC steps
    (requirements 1, 2, and the structural half of 5) unusable on any machine
    without the full GPU stack -- exactly the machines you want to run cheap
    statistics on.

    So: try the ordinary import first (correct and fastest where the deps
    exist), and if it fails on a missing dependency, load `params.py` directly
    from its file. That module imports only json/typing/pydantic, so loading it
    standalone yields the very same class -- the production key mapping is
    reused verbatim either way, never re-implemented here.

    The `sys.modules` registration before `exec_module` is load-bearing:
    params.py uses `from __future__ import annotations`, so pydantic resolves
    its field types by looking the module up in `sys.modules[cls.__module__]`.
    Skip the registration and every model comes back "not fully defined", which
    QC would then quietly paper over with default thresholds -- i.e. report
    confident numbers measured against the wrong bar.
    """
    try:
        from pipeline_v2.params import PipelineParams

        return PipelineParams
    except ImportError:
        import importlib.util
        import sys

        params_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pipeline_v2", "params.py",
        )
        if not os.path.exists(params_path):
            raise
        mod_name = "_qc_pipeline_v2_params"
        cached = sys.modules.get(mod_name)
        if cached is not None:
            return cached.PipelineParams
        spec = importlib.util.spec_from_file_location(mod_name, params_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        return module.PipelineParams


@dataclass(frozen=True)
class Thresholds:
    """The production numbers QC compares against, plus their provenance."""

    merge_gap: float
    min_segment_length: float
    max_segment_length: float
    intra_similarity_threshold: float
    inter_similarity_threshold: float
    grace_period_start: float
    grace_period_end: float
    fixed_dnsmos_threshold: float
    fixed_c50_threshold: float
    fixed_snr_threshold: float
    metrics_strategy: str
    use_brouhaha: bool
    source: str
    is_default: bool
    # Parsed-but-unvalidated note explaining why a config fell back, if it did.
    fallback_reason: Optional[str] = None

    @classmethod
    def defaults(cls, reason: Optional[str] = None, source: str = "<built-in defaults>") -> "Thresholds":
        return cls(source=source, is_default=True, fallback_reason=reason, **_THRESHOLD_FALLBACKS)

    @classmethod
    def from_config(cls, config_path: Optional[str]) -> "Thresholds":
        """Parse a `configs/config_for_*.json` through production's own parser.

        Falls back to the built-in defaults (flagged as such) on any failure
        instead of aborting: a QC run that can still produce four of five
        sections is far more useful than one that refuses to start because a
        config lacks, say, `huggingface_token`.
        """
        if not config_path:
            return cls.defaults(reason="no --config supplied")
        if not os.path.exists(config_path):
            return cls.defaults(reason=f"config not found: {config_path}", source=config_path)
        try:
            params = _load_pipeline_params_cls().from_config(config_path)
        except Exception as exc:  # noqa: BLE001 - any parse failure degrades, never aborts
            return cls.defaults(
                reason=f"{type(exc).__name__}: {exc}", source=config_path
            )
        return cls.from_pipeline_params(params, config_path)

    @classmethod
    def from_pipeline_params(cls, params: Any, source: str) -> "Thresholds":
        seg = params.segmenter
        emb = params.embedding_refinement
        met = params.metrics
        return cls(
            merge_gap=seg.merge_gap,
            min_segment_length=seg.min_segment_length,
            max_segment_length=seg.max_segment_length,
            intra_similarity_threshold=seg.intra_similarity_threshold,
            inter_similarity_threshold=emb.inter_similarity_threshold,
            grace_period_start=seg.grace_period_start,
            grace_period_end=seg.grace_period_end,
            fixed_dnsmos_threshold=met.fixed_dnsmos_threshold,
            fixed_c50_threshold=met.fixed_c50_threshold,
            fixed_snr_threshold=met.fixed_snr_threshold,
            metrics_strategy=met.strategy,
            use_brouhaha=met.use_brouhaha,
            source=source,
            is_default=False,
        )

    def to_dict(self) -> dict:
        """Report-safe view.

        This is the ONLY way thresholds reach a report. The parsed config also
        holds `huggingface_token` and third-party API keys (they sit in plain
        text in configs/config_for_*.json), so QC never serialises the config
        object -- only this explicit allow-list of numeric thresholds.
        """
        return {
            "source": self.source,
            "is_default": self.is_default,
            "fallback_reason": self.fallback_reason,
            "merge_gap": self.merge_gap,
            "min_segment_length": self.min_segment_length,
            "max_segment_length": self.max_segment_length,
            "intra_similarity_threshold": self.intra_similarity_threshold,
            "inter_similarity_threshold": self.inter_similarity_threshold,
            "grace_period_start": self.grace_period_start,
            "grace_period_end": self.grace_period_end,
            "fixed_dnsmos_threshold": self.fixed_dnsmos_threshold,
            "fixed_c50_threshold": self.fixed_c50_threshold,
            "fixed_snr_threshold": self.fixed_snr_threshold,
            "metrics_strategy": self.metrics_strategy,
            "use_brouhaha": self.use_brouhaha,
        }


@dataclass
class QCConfig:
    """Everything one QC invocation needs, resolved from the CLI."""

    output_root: str
    report_dir: str
    work_dir: str

    thresholds: Thresholds
    config_path: Optional[str] = None

    manifest: Optional[str] = None
    shards: Optional[list[str]] = None
    steps: tuple[str, ...] = ALL_STEPS

    # Sampling. `sample_n <= 0` means "no sampling, score everything".
    sample_n: int = 2000
    sample_per_shard: bool = True
    pair_sample_n: int = 2000

    # Parallelism.
    workers: int = 32
    gpu_workers: int = 1
    devices: tuple[str, ...] = ("cuda:0",)
    merge_buckets: int = 64

    # QC-only thresholds (no production counterpart), echoed in the report.
    bak_pass: float = 4.0
    bak_warn: float = 3.0
    dia_min_reliable_s: float = 2.0

    keep_work: bool = False

    def wants(self, step: str) -> bool:
        return step in self.steps

    @property
    def needs_gpu_pass(self) -> bool:
        return any(self.wants(s) for s in GPU_STEPS)

    def runtime_to_dict(self) -> dict:
        """Report-safe echo of how this run was parameterised."""
        return {
            "output_root": self.output_root,
            "manifest": self.manifest,
            "shards": self.shards,
            "steps": list(self.steps),
            "sample_n": self.sample_n,
            "sample_per_shard": self.sample_per_shard,
            "pair_sample_n": self.pair_sample_n,
            "workers": self.workers,
            "gpu_workers": self.gpu_workers,
            "devices": list(self.devices),
            "merge_buckets": self.merge_buckets,
            "bak_pass": self.bak_pass,
            "bak_warn": self.bak_warn,
            "dia_min_reliable_s": self.dia_min_reliable_s,
        }
