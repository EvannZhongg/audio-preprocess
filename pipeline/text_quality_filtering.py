"""
Text quality prediction & filtering pipeline step.

Mirrors the structure of `pipeline/metrics_prediction.py`:
- `text_quality_prediction()` computes scores and attaches them in-place to segments
- `filter_by_text_quality()` drops segments below configured thresholds

All scorers are independently optional and gracefully degrade on failure
(score = -1) so a single broken scorer does not break the pipeline.
"""
import tqdm

from models.text_quality import check_semantic_completeness
from utils.logger import time_logger


@time_logger
def text_quality_prediction(asr_result, text_quality_cfg):
    """Compute text quality scores for each segment, in-place.

    Args:
        asr_result: list of segment dicts. Each must have at least "text".
                    Optional: "language" (default "ru").
        text_quality_cfg: dict with sub-configs:
            - ppl: {"enable": bool}
            - semantic: {"enable": bool, "min_char_count": int}
            - spell: {"enable": bool}
            - llm: {"enable": bool}

    Returns:
        The same list with new fields added per segment:
            ppl, semantic_score, spell_score, llm_quality, tts_suitability
    """
    from pipeline.global_var import PipelineParam

    logger = PipelineParam.logger
    ppl_scorer = PipelineParam.ppl_scorer
    spell_checker = PipelineParam.spell_checker
    llm_scorer = PipelineParam.llm_text_scorer

    if not asr_result:
        return asr_result

    ppl_enabled = text_quality_cfg.get("ppl", {}).get("enable", False) and ppl_scorer is not None
    semantic_enabled = text_quality_cfg.get("semantic", {}).get("enable", False)
    spell_enabled = text_quality_cfg.get("spell", {}).get("enable", False) and spell_checker is not None
    llm_enabled = text_quality_cfg.get("llm", {}).get("enable", False) and llm_scorer is not None

    semantic_min_char = text_quality_cfg.get("semantic", {}).get("min_char_count", 3)
    semantic_require_end_punct = text_quality_cfg.get("semantic", {}).get("require_end_punct", False)
    # Optimization: skip the expensive spell check when semantic_score is already
    # very low (likely repetition / fragment). Set spell_skip_below=null to disable.
    spell_skip_below = text_quality_cfg.get("spell", {}).get("skip_below_semantic", 0.3)

    # initialize default values for all segments (so missing keys won't break export)
    for seg in asr_result:
        seg.setdefault("ppl", -1.0)
        seg.setdefault("semantic_score", -1.0)
        seg.setdefault("spell_score", -1.0)
        seg.setdefault("llm_quality", -1.0)
        seg.setdefault("tts_suitability", -1.0)

    # ---------- per-segment scoring (PPL / semantic / spell) ----------
    for seg in tqdm.tqdm(asr_result, desc="TEXT_QUALITY"):
        text = seg.get("text", "")
        lang = seg.get("language", "ru")
        if not text or not text.strip():
            continue

        if ppl_enabled:
            try:
                seg["ppl"] = ppl_scorer.score(text)
            except Exception as e:
                logger.warning(f"PPL scoring exception: {e}")
                seg["ppl"] = -1.0

        if semantic_enabled:
            try:
                seg["semantic_score"] = check_semantic_completeness(
                    text,
                    language=lang,
                    min_char_count=semantic_min_char,
                    require_end_punct=semantic_require_end_punct,
                )
            except Exception as e:
                logger.warning(f"Semantic check exception: {e}")
                seg["semantic_score"] = -1.0

        if spell_enabled:
            # Early-exit: if semantic_score already says this is severely broken
            # (repetition / fragment), skip the expensive spell check.
            # The segment will be dropped by semantic filter anyway.
            sem = seg.get("semantic_score", -1.0)
            if (spell_skip_below is not None
                    and sem >= 0
                    and sem < spell_skip_below):
                seg["spell_score"] = -1.0  # mark as not-scored
            else:
                try:
                    seg["spell_score"] = spell_checker.score(text, language=lang)
                except Exception as e:
                    logger.warning(f"Spell check exception: {e}")
                    seg["spell_score"] = -1.0

    # ---------- batch LLM scoring (one API call per segment, run concurrently) ----------
    if llm_enabled:
        items = [
            {"idx": i, "text": seg.get("text", "")}
            for i, seg in enumerate(asr_result)
        ]
        try:
            scores = llm_scorer.score_batch(items)
            for i, seg in enumerate(asr_result):
                s = scores.get(i, {})
                seg["llm_quality"] = s.get("quality", -1.0)
                seg["tts_suitability"] = s.get("tts", -1.0)
        except Exception as e:
            logger.warning(f"LLM batch scoring failed: {e}")

    # log average non-negative scores
    def _avg(key):
        vals = [s.get(key, -1) for s in asr_result if s.get(key, -1) >= 0]
        return sum(vals) / len(vals) if vals else -1

    logger.info(
        f"Text quality avg: ppl={_avg('ppl'):.2f}, "
        f"semantic={_avg('semantic_score'):.2f}, "
        f"spell={_avg('spell_score'):.2f}, "
        f"llm_quality={_avg('llm_quality'):.2f}, "
        f"tts_suitability={_avg('tts_suitability'):.2f}"
    )

    return asr_result


def filter_by_text_quality(segments, text_quality_cfg):
    """Drop segments that fall below configured thresholds.

    Threshold = None means that dimension is not used for filtering.
    Score = -1 means scorer failed for that segment; we DO NOT drop these
    (avoid penalizing scorer failures).

    Args:
        segments: list of segment dicts after text_quality_prediction.
        text_quality_cfg: dict with "thresholds" sub-config:
            - ppl_max: float | None
            - semantic_min: float | None
            - spell_min: float | None
            - llm_quality_min: float | None
            - tts_min: float | None

    Returns:
        Filtered list.
    """
    from pipeline.global_var import PipelineParam
    logger = PipelineParam.logger

    thresholds = text_quality_cfg.get("thresholds", {}) or {}
    ppl_max = thresholds.get("ppl_max")
    semantic_min = thresholds.get("semantic_min")
    spell_min = thresholds.get("spell_min")
    llm_quality_min = thresholds.get("llm_quality_min")
    tts_min = thresholds.get("tts_min")

    if not any(v is not None for v in (ppl_max, semantic_min, spell_min, llm_quality_min, tts_min)):
        logger.debug("No text quality thresholds configured; skipping filter.")
        return segments

    filtered = []
    drop_counts = {"ppl": 0, "semantic": 0, "spell": 0, "llm": 0, "tts": 0}

    for seg in segments:
        ppl = seg.get("ppl", -1)
        sem = seg.get("semantic_score", -1)
        spl = seg.get("spell_score", -1)
        llm_q = seg.get("llm_quality", -1)
        tts = seg.get("tts_suitability", -1)

        # only filter when scorer succeeded (>= 0) AND threshold set
        if ppl_max is not None and ppl >= 0 and ppl > ppl_max:
            drop_counts["ppl"] += 1
            continue
        if semantic_min is not None and sem >= 0 and sem < semantic_min:
            drop_counts["semantic"] += 1
            continue
        if spell_min is not None and spl >= 0 and spl < spell_min:
            drop_counts["spell"] += 1
            continue
        if llm_quality_min is not None and llm_q >= 0 and llm_q < llm_quality_min:
            drop_counts["llm"] += 1
            continue
        if tts_min is not None and tts >= 0 and tts < tts_min:
            drop_counts["tts"] += 1
            continue
        filtered.append(seg)

    logger.info(
        f"Text quality filter: kept {len(filtered)}/{len(segments)}; "
        f"dropped by ppl={drop_counts['ppl']}, semantic={drop_counts['semantic']}, "
        f"spell={drop_counts['spell']}, llm={drop_counts['llm']}, tts={drop_counts['tts']}"
    )
    return filtered
