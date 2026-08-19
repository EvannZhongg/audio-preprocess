"""Explicit, validated parameter schema for PipelineV2."""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class StandardizationParams(BaseModel):
    target_sample_rate: int
    num_threads: int
    ffmpeg_timeout: int
    max_audio_duration_seconds: float
    max_file_size_bytes: int
    target_dbfs: float
    chunk_min_seconds: float
    chunk_max_seconds: float


class SourceSeparationParams(BaseModel):
    enable: bool
    provider: Literal["smru", "uvr"]
    smru_conf: dict[str, Any]
    uvr_conf: dict[str, Any]


class DiarizationParams(BaseModel):
    provider: Literal["pyannote"]
    huggingface_token: str
    pyannote_model: str
    pyannote_model_dir_cache: Optional[str]


class EmbeddingRefinementParams(BaseModel):
    enable: bool
    eres2net_model_path: str
    inter_similarity_threshold: float   # min cosine sim across windows vs reference
    refinement_batch_size: int


class SegmenterParams(BaseModel):
    merge_gap: float                   # seconds: same-speaker gap shorter than this is merged
    min_segment_length: float          # seconds: drop segments shorter than this
    max_segment_length: float          # seconds: split or drop segments longer than this
    intra_similarity_threshold: float  # min cosine sim between adjacent embeddings to merge
    grace_period_start: float          # seconds: extend start backward
    grace_period_end: float            # seconds: extend end forward


class MetricsParams(BaseModel):
    dnsmos_model_path: str
    use_brouhaha: bool
    brouhaha_model: str
    brouhaha_model_dir_cache: Optional[str]
    huggingface_token: str
    fixed_c50_threshold: float
    fixed_snr_threshold: float
    strategy: Literal["fixed", "average"]
    fixed_dnsmos_threshold: float


class FunasrWarmupParams(BaseModel):
    """Preload-only FunASR model. Not invoked by any current stage; loading
    it warms up CUDA / numpy kernels and measurably speeds up later
    numpy/torch work in the same worker.

    Both the hub id and the local cache path are carried through verbatim;
    the loader decides at runtime which one to use (cache if it exists,
    otherwise the hub id).
    """
    asr_model: str                 # kind, e.g. "SenseVoice"
    asr_model_id: str              # hub id, e.g. "iic/SenseVoiceSmall"
    asr_model_dir_cache: str       # local path (may not exist)
    vad_model_id: str              # hub id, e.g. "fsmn-vad"
    vad_model_dir_cache: str       # local path (may not exist)


class PipelineParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    device_name: str

    standardization: StandardizationParams
    source_separation: SourceSeparationParams
    diarization: DiarizationParams
    embedding_refinement: EmbeddingRefinementParams
    segmenter: SegmenterParams
    metrics: MetricsParams
    funasr_warmup: FunasrWarmupParams

    @classmethod
    def from_config(cls, config_path: str) -> "PipelineParams":
        """Map legacy v1 config schema (configs/config_for_*.json) onto v2.

        Pure key remapping — no existence checks, fallbacks, or runtime
        judgement here. Defaults mirror the values previously hard-coded in
        the legacy pipeline (ray_task/config.py, standardization.py,
        main_process.py, vad_process.py, global_var.py).
        """
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        strategy = cfg.get("strategy_parameters", {})
        embedding = cfg.get("embedding_refinement", {})
        separate = cfg.get("separate", {})
        pyannote = cfg.get("pyannote", {})
        funasr = cfg.get("funasr", {})
        entrypoint = cfg.get("entrypoint", {})
        metrics_cfg = cfg.get("metrics", {})
        brouhaha_cfg = metrics_cfg.get("brouhaha", {})
        mos_model_cfg = cfg.get("mos_model", {})

        v2 = {
            "device_name": cfg.get("device_name", "cuda:0"),
            "standardization": {
                "target_sample_rate": entrypoint.get("SAMPLE_RATE", 24000),
                "num_threads": cfg.get("threads", 4),
                "ffmpeg_timeout": 200,
                "max_audio_duration_seconds": 5 * 3600,
                "max_file_size_bytes": 5 * 1024 * 1024 * 1024,
                "target_dbfs": -20.0,
                "chunk_min_seconds": 600.0,
                "chunk_max_seconds": 1800.0,
            },
            "source_separation": {
                "enable": separate.get("enable", True),
                "provider": separate.get("provider", "uvr"),
                "smru_conf": separate.get("smru", {}),
                "uvr_conf": separate.get("uvr", {}),
            },
            "diarization": {
                "provider": pyannote.get("provider", "pyannote"),
                "huggingface_token": cfg["huggingface_token"],
                "pyannote_model": pyannote.get("model", "pyannote/speaker-diarization-3.1"),
                "pyannote_model_dir_cache": pyannote.get("model_dir_cache"),
            },
            "embedding_refinement": {
                "enable": embedding.get("enable", True),
                "eres2net_model_path": embedding["eres2net_model_path"],
                "inter_similarity_threshold": strategy.get("inter_similarity_threshold", 0.7),
                "refinement_batch_size": strategy.get("refinement_batch_size", 64),
            },
            "segmenter": {
                "merge_gap": strategy.get("merge_gap", 2.0),
                "min_segment_length": strategy.get("min_segment_length", 3.0),
                "max_segment_length": strategy.get("max_segment_length", 30.0),
                "intra_similarity_threshold": strategy.get("intra_similarity_threshold", 0.64),
                "grace_period_start": 0.0,
                "grace_period_end": 0.02,
            },
            "metrics": {
                "dnsmos_model_path": mos_model_cfg.get("primary_model_path", "ckpts/sig_bak_ovr.onnx"),
                "use_brouhaha": metrics_cfg.get("use_brouhaha", False),
                "brouhaha_model": brouhaha_cfg.get("model", "pyannote/brouhaha"),
                "brouhaha_model_dir_cache": brouhaha_cfg.get("model_dir_cache"),
                "huggingface_token": cfg["huggingface_token"],
                "fixed_c50_threshold": strategy.get("fixed_c50_threshold", 40.0),
                "fixed_snr_threshold": strategy.get("fixed_snr_threshold", 40.0),
                "strategy": strategy.get("strategy", "average"),
                "fixed_dnsmos_threshold": strategy.get("fixed_dnsmos_threshold", 3.0),
            },
            "funasr_warmup": {
                "asr_model": "SenseVoice",
                "asr_model_id": funasr.get("model", "iic/SenseVoiceSmall"),
                "asr_model_dir_cache": funasr.get("model_dir_cache", ""),
                "vad_model_id": funasr.get("vad_model", "fsmn-vad"),
                "vad_model_dir_cache": funasr.get("vad_model_dir_cache", ""),
            },
        }
        return cls.model_validate(v2)


class Stage2Params(BaseModel):
    """Parameter schema for the stage-2 pipeline (ASR + v1 post-processing).

    Sub-sections are kept as raw dicts (rather than fully-typed sub-models)
    because they are passed through almost verbatim to v1's model loaders /
    functions (`init_pipeline_global`, `annotate_domains`, etc.), which already
    validate/consume their own keys. This avoids duplicating v1's schema here
    and keeps this file in sync automatically when v1's config gains new keys.
    """
    model_config = ConfigDict(frozen=True)

    device_name: str

    # Which ASR backend to load. "whisper" is the local fallback (default) and
    # runs entirely on-device; "qwen3_asr" is the remote service. Both expose
    # the same duck-typed contract (transcribe -> {"segments", "language"}), so
    # switching only requires changing this one field — stage-2 runner/actor/
    # parquet/resume logic is unaffected.
    #
    # NOTE: sourced from the config json's "stage2_asr_provider" key, NOT the
    # legacy top-level "asr_provider" — that key already has different
    # semantics for v1's own cross-validation pipeline (funasr/paraformer/
    # whisper/qwen3_asr/gemini, see pipeline/global_var.py:load_asr_model) and
    # some shared config files (e.g. config_for_v100_for_zh.json) set it to
    # values ("funasr") stage 2 doesn't understand. Keeping a dedicated key
    # means any legacy config can be reused for stage 2 unmodified, always
    # safely defaulting to the local whisper fallback.
    asr_provider: str = "whisper"

    # Local ASR (faster-whisper / WhisperX) config section. Consumed only when
    # asr_provider == "whisper".
    whisper: dict[str, Any] = {}
    # Remote ASR (Qwen3, Polaris + HTTP) config section. Consumed only when
    # asr_provider == "qwen3_asr".
    qwen3_asr: dict[str, Any]
    domain_annotation: dict[str, Any]
    speaking_rate: dict[str, Any]
    silence_filter: dict[str, Any]
    alignment: dict[str, Any]
    text_quality: dict[str, Any]

    # --- ASR cross-validation (mirrors v1's pipeline/asr_process.py asr()
    # "ASR Cross-Validation Logic" block) ---
    # Which v1 `load_asr_model` provider (gemini/funasr/funasr_nano/
    # paraformer/whisper/qwen3_asr) to use for the *second*, verification-only
    # ASR pass. Sourced from the legacy top-level "validation_asr_provider"
    # key -- already present, unmodified, in every shared config json.
    validation_asr_provider: str = "whisper"
    # {"enable": bool, "language": str, "wer_threshold": float}, sourced
    # verbatim from the legacy top-level "asr_validation" key. Consumed by
    # stage2/runner.py exactly like v1's asr(): enable gates the whole
    # feature, language drives the detected-language filter + CER-vs-WER
    # choice, wer_threshold is the pass/fail cutoff.
    asr_validation: dict[str, Any] = {}
    # Full, unmodified original config dict. Needed only so the validation
    # ASR model can be loaded via v1's `pipeline.global_var.load_asr_model`
    # (which indexes cfg["funasr"]/cfg["paraformer"]/cfg["gemini"]/etc. --
    # sub-sections stage2 otherwise never parses), without duplicating v1's
    # schema here.
    raw_cfg: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config_path: str) -> "Stage2Params":
        """Read the relevant sections straight out of a v1 config json
        (configs/config_for_*.json). No remapping: v1's post-processing
        functions and model loaders consume these dicts directly.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        return cls.model_validate(
            {
                "device_name": cfg.get("device_name", "cuda:0"),
                "asr_provider": cfg.get("stage2_asr_provider", "whisper"),
                "whisper": cfg.get("whisper", {}),
                "qwen3_asr": cfg.get("qwen3_asr", {}),
                "domain_annotation": cfg.get("domain_annotation", {}),
                "speaking_rate": cfg.get("speaking_rate", {}),
                "silence_filter": cfg.get("silence_filter", {}),
                "alignment": cfg.get("alignment", {}),
                "text_quality": cfg.get("text_quality", {}),
                "validation_asr_provider": cfg.get("validation_asr_provider", "whisper"),
                "asr_validation": cfg.get("asr_validation", {}),
                "raw_cfg": cfg,
            }
        )
