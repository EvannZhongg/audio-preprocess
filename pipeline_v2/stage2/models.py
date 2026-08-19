"""Stage-2 model loading: pluggable ASR (local whisper / remote Qwen3) +
v1 post-processing models.

The ASR backend is selected by `Stage2Params.asr_provider`: `"whisper"`
(local faster-whisper/WhisperX, the default fallback that needs no remote
service) or `"qwen3_asr"` (remote Qwen3). Both expose the same duck-typed
`transcribe(...) -> {"segments", "language"}` contract, so switching is a
one-line config change and the stage-2 runner/actor/parquet logic is
unaffected.

Stage 2 reuses v1's `pipeline.global_var.PipelineParam` process-global
singleton so it can call v1's post-processing functions
(`annotate_domains`, `analyze_speaking_rate`, `detect_abnormal_silence`,
`compute_alignment_score`, `text_quality_prediction`) and their
`filter_by_*` counterparts completely unmodified. Each Ray actor runs in
its own OS process, so this process-global singleton is safe: one stage-2
GPU actor == one `PipelineParam`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from pipeline_v2.params import Stage2Params


@dataclass
class Stage2Models:
    """Handle bundle returned by `load_stage2_models`.

    `asr_model` is used directly by the stage-2 runner; the rest are also
    stashed on `pipeline.global_var.PipelineParam` (matching how v1's
    postprocessing functions read them) and kept here only so callers can
    inspect / log what actually loaded.
    """

    asr_model: Any
    vad_model: Optional[Any] = None
    domain_classifier: Optional[Any] = None
    aligner: Optional[Any] = None
    ppl_scorer: Optional[Any] = None
    spell_checker: Optional[Any] = None
    llm_text_scorer: Optional[Any] = None
    # Second ASR model used only to cross-validate `asr_model`'s output
    # (pipeline/asr_process.py asr()'s "ASR Cross-Validation Logic"). None
    # when `params.asr_validation.enable` is false, the configured provider
    # collides with the primary one, or loading it failed -- all of which
    # degrade gracefully to "no cross-validation" rather than aborting.
    validation_asr_model: Optional[Any] = None


# Default disfluency initial prompt migrated verbatim from v1's whisper
# branch (pipeline/global_var.py load_asr_model) so local ASR keeps the same
# behaviour (biases the decoder to keep hesitations / fillers).
_WHISPER_INITIAL_PROMPT = (
    "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? "
    "okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,"
    "哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。"
    "äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, "
    "t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
)

_DEFAULT_WHISPER_MODEL = "Systran/faster-distil-whisper-large-v3"


def _load_asr_model(provider: str, params: Stage2Params, device_name: str, logger):
    """Lightweight stage-2 ASR factory dispatching by provider.

    Both branches return an object exposing the same duck-typed contract used
    by the stage-2 runner:
        transcribe(audio_16k_mono, vad_segments)
            -> {"segments": [{text, start, end, speaker, ...}], "language": str}

    - "whisper":   local faster-whisper / WhisperX (default fallback; works
                   without any remote service and reports the real language).
    - "qwen3_asr": remote Qwen3 (Polaris service discovery + HTTP). Switch to
                   this by setting `asr_provider: "qwen3_asr"` in the config
                   once the remote service is ready — no other code changes.

    Robustness handling for the whisper branch (model_dir_cache existence
    check, cpu -> float32 fallback, compute_type, threads, disfluency
    initial_prompt) is migrated from v1's `pipeline.global_var.load_asr_model`
    whisper branch so local runs avoid known pitfalls.
    """
    if provider == "whisper":
        import models.whisper_asr as whisper_asr_mod

        w_cfg = params.whisper or {}
        model_path = w_cfg.get("model", _DEFAULT_WHISPER_MODEL)
        model_dir_cache = w_cfg.get("model_dir_cache", "")
        if model_dir_cache and os.path.exists(model_dir_cache):
            model_path = model_dir_cache

        compute_type = w_cfg.get("compute_type", "float16")
        if device_name == "cpu" and compute_type != "float32":
            compute_type = "float32"

        threads = int(w_cfg.get("threads", 4))
        initial_prompt = w_cfg.get("initial_prompt", _WHISPER_INITIAL_PROMPT)
        # Forward the configured language (if any) so a fixed-language
        # tokenizer is built at load time instead of falling back to
        # per-audio auto-detection (slower, and unreliable on
        # English-only-distilled checkpoints).
        language = w_cfg.get("language") or None

        logger.info(
            f"Stage2: loading local whisper ASR (model={model_path}, "
            f"device={device_name}, compute_type={compute_type}, "
            f"language={language or 'auto-detect'})"
        )
        return whisper_asr_mod.load_asr_model(
            model_path=model_path,
            device=device_name,
            compute_type=compute_type,
            threads=threads,
            language=language,
            asr_options={"initial_prompt": initial_prompt},
        )

    if provider == "qwen3_asr":
        import models.qwen3_asr as qwen3_asr_mod

        asr_cfg = params.qwen3_asr or {}
        # Chunk-level batching knobs: segments of one chunk are grouped into
        # `batch_size`-sized batch requests, and up to `max_group_workers` of
        # those requests fly concurrently. Both are optional -- omitting them
        # keeps the model wrapper's own defaults, so existing configs work
        # unchanged.
        batch_size = int(
            asr_cfg.get("batch_size", qwen3_asr_mod.Qwen3ASR.DEFAULT_BATCH_SIZE)
        )
        max_group_workers = int(
            asr_cfg.get(
                "max_group_workers", qwen3_asr_mod.Qwen3ASR.DEFAULT_MAX_GROUP_WORKERS
            )
        )
        logger.info(
            f"Stage2: loading remote qwen3_asr (device={device_name}, "
            f"batch_size={batch_size}, max_group_workers={max_group_workers})"
        )
        return qwen3_asr_mod.load_asr_model(
            namespace=asr_cfg.get("namespace", "Test"),
            service=asr_cfg.get(
                "service", "audio_process_qwen3_asr_service"
            ),
            model_name=asr_cfg.get("model_name", "Qwen/Qwen3-ASR-1.7B"),
            device=device_name,
            hot_words=asr_cfg.get("hot_words", ""),
            batch_size=batch_size,
            max_group_workers=max_group_workers,
        )

    raise ValueError(
        f"Unknown stage2 asr_provider: {provider!r} "
        "(supported: 'whisper', 'qwen3_asr')"
    )


def load_stage2_models(params: Stage2Params, logger=None) -> Stage2Models:
    """Load the ASR backend (per `params.asr_provider`) plus whichever v1
    post-processing models are enabled in `params`, and populate
    `pipeline.global_var.PipelineParam` so v1's unmodified postprocessing
    functions can find them.

    Must be called exactly once per process (i.e. once per Ray actor
    instance, during actor init).
    """
    from pipeline.global_var import PipelineParam
    from utils.logger import Logger

    if logger is None:
        logger = Logger.get_logger("stage2_worker")

    device_name = params.device_name

    # v1's postprocessing functions all pull `PipelineParam.logger` /
    # `PipelineParam.<model>` directly (module-level globals), so stage 2
    # must populate the same singleton rather than passing models around.
    PipelineParam.logger = logger
    PipelineParam.device = device_name

    # 1. ASR - always required. Loaded via a provider factory so the remote
    # Qwen3 service can be swapped for a local whisper fallback (or back)
    # purely through config (`asr_provider`); the runner is interface-agnostic.
    asr_model = _load_asr_model(
        params.asr_provider, params, device_name, logger
    )
    PipelineParam.asr_model = asr_model

    # 1b. ASR cross-validation (v1 pipeline/asr_process.py asr()'s "ASR
    # Cross-Validation Logic") - a second, independently configured ASR
    # model used only to score the primary model's transcripts. Config-gated
    # by the legacy top-level `asr_validation.enable` / `validation_asr_provider`
    # keys (unchanged, already present in the shared config jsons). Loaded
    # via v1's own `load_asr_model` factory rather than stage 2's lightweight
    # whisper/qwen3_asr-only `_load_asr_model`, so all six v1 providers
    # (gemini/funasr/funasr_nano/paraformer/whisper/qwen3_asr) are usable for
    # validation regardless of which primary ASR stage 2 is running.
    validation_asr_model = None
    val_cfg = params.asr_validation or {}
    if val_cfg.get("enable", False):
        validation_provider = params.validation_asr_provider
        if validation_provider == params.asr_provider:
            # v1 hard-asserts on this instead; stage 2 treats cross-validation
            # as an optional enhancement, so it degrades to "disabled" and
            # keeps the actor usable rather than crashing.
            logger.error(
                "Stage2: validation_asr_provider "
                f"({validation_provider!r}) is the same as asr_provider; "
                "skipping ASR cross-validation (comparing a model against "
                "itself would be meaningless)."
            )
        else:
            try:
                from types import SimpleNamespace

                from pipeline.global_var import load_asr_model as v1_load_asr_model

                # v1's load_asr_model only ever reads `cli_args.threads`
                # (whisper branch); a minimal stand-in avoids depending on
                # stage 2's own CLI arg object.
                cli_args = SimpleNamespace(threads=int((params.whisper or {}).get("threads", 4)))
                logger.info(
                    f"Stage2: loading validation ASR (provider={validation_provider}) "
                    "for cross-validation"
                )
                validation_asr_model = v1_load_asr_model(
                    params.raw_cfg, validation_provider, device_name, cli_args
                )
            except Exception as e:
                logger.error(
                    f"Stage2: failed to load validation ASR "
                    f"({validation_provider}): {e}"
                )
                validation_asr_model = None
    PipelineParam.validation_asr_model = validation_asr_model

    # 2. VAD - feeds speaking_rate + silence_filter (both read
    # `PipelineParam.vad_model` directly); load if either is enabled.
    vad_model = None
    sr_cfg = params.speaking_rate
    silence_cfg = params.silence_filter
    if sr_cfg.get("enable", False) or silence_cfg.get("enable", False):
        import torch

        from models import vad as vad_mod

        logger.debug("Stage2: loading VAD model")
        vad_model = vad_mod.SileroVAD(device=torch.device(device_name))
    PipelineParam.vad_model = vad_model

    # 3. Domain classifier (Qwen3-Omni, LLM API)
    domain_classifier = None
    da_cfg = params.domain_annotation
    if da_cfg.get("enable", False):
        from models.domain_classifier import Qwen3OmniDomainClassifier
        from utils.domain_enums import (resolve_acoustic_enums,
                                        resolve_speaker_enums,
                                        resolve_text_enums)

        llm_cfg = da_cfg.get("llm", {})
        text_enums = resolve_text_enums(da_cfg.get("text_domain", {}))
        acoustic_enums = resolve_acoustic_enums(da_cfg.get("acoustic_domain", {}))
        speaker_enums = resolve_speaker_enums(da_cfg.get("speaker_domain", {}))
        try:
            domain_classifier = Qwen3OmniDomainClassifier(
                api_url=llm_cfg.get("api_url", ""),
                api_token=llm_cfg.get("api_token", ""),
                model_id=llm_cfg.get("model_id", ""),
                text_enums=text_enums,
                acoustic_enums=acoustic_enums,
                speaker_enums=speaker_enums,
                timeout=llm_cfg.get("timeout", 60),
                max_retries=llm_cfg.get("max_retries", 2),
            )
        except Exception as e:
            logger.error(f"Stage2: failed to init Qwen3OmniDomainClassifier: {e}")
    PipelineParam.domain_classifier = domain_classifier

    # 4. Forced alignment (WhisperX)
    aligner = None
    al_cfg = params.alignment
    if al_cfg.get("enable", False):
        from models.alignment import WhisperXAligner

        try:
            aligner = WhisperXAligner(
                device=device_name,
                model_dir=al_cfg.get("model_dir_cache"),
                language_models=al_cfg.get("language_models", {}),
            )
        except Exception as e:
            logger.error(f"Stage2: failed to init WhisperXAligner: {e}")
    PipelineParam.aligner = aligner

    # 5. Text quality scorers (PPL / spell / LLM)
    ppl_scorer = None
    spell_checker = None
    llm_text_scorer = None
    tq_cfg = params.text_quality
    if tq_cfg.get("enable", False):
        from models.text_quality import (PerplexityScorer,
                                         Qwen3OmniTextScorer, SpellChecker)

        ppl_cfg = tq_cfg.get("ppl", {})
        if ppl_cfg.get("enable", False):
            ppl_model = ppl_cfg.get("model", "Qwen/Qwen2.5-0.5B")
            ppl_cache = ppl_cfg.get("model_dir_cache", "")
            ppl_path = ppl_cache if ppl_cache and os.path.exists(ppl_cache) else ppl_model
            try:
                ppl_scorer = PerplexityScorer(model_path=ppl_path, device=device_name)
            except Exception as e:
                logger.error(f"Stage2: failed to load PPL model: {e}")

        spell_cfg = tq_cfg.get("spell", {})
        if spell_cfg.get("enable", False):
            try:
                spell_checker = SpellChecker()
            except Exception as e:
                logger.error(f"Stage2: failed to init SpellChecker: {e}")

        llm_cfg = tq_cfg.get("llm", {})
        if llm_cfg.get("enable", False):
            try:
                llm_text_scorer = Qwen3OmniTextScorer(
                    api_url=llm_cfg.get("api_url", ""),
                    api_token=llm_cfg.get("api_token", ""),
                    model_id=llm_cfg.get("model_id", ""),
                    timeout=llm_cfg.get("timeout", 30),
                    concurrency=llm_cfg.get("concurrency", 16),
                    max_retries=llm_cfg.get("max_retries", 2),
                )
            except Exception as e:
                logger.error(f"Stage2: failed to init Qwen3OmniTextScorer: {e}")
    PipelineParam.ppl_scorer = ppl_scorer
    PipelineParam.spell_checker = spell_checker
    PipelineParam.llm_text_scorer = llm_text_scorer

    logger.info(
        f"Stage2 models loaded: asr={params.asr_provider} "
        f"validation_asr={validation_asr_model is not None} "
        f"vad={vad_model is not None} "
        f"domain={domain_classifier is not None} align={aligner is not None} "
        f"ppl={ppl_scorer is not None} spell={spell_checker is not None} "
        f"llm={llm_text_scorer is not None}"
    )

    return Stage2Models(
        asr_model=asr_model,
        vad_model=vad_model,
        domain_classifier=domain_classifier,
        aligner=aligner,
        ppl_scorer=ppl_scorer,
        spell_checker=spell_checker,
        llm_text_scorer=llm_text_scorer,
        validation_asr_model=validation_asr_model,
    )
