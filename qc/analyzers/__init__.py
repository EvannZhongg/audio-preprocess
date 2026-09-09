"""QC analyzers.

Each module owns exactly one requirement and exposes an `analyze(...)` that
returns a plain dict (already report-shaped, via the accumulators'
`to_dict()`), so `qc/report.py` never needs to know how a statistic was
computed.

`run_all` sequences them: the parquet-only analyzers first (cheap, full
dataset), then a single GPU pass shared by the model-based ones -- decoding
audio once for all three re-checks rather than once per check.
"""
from __future__ import annotations

import logger
from qc.config import QCConfig


def run_all(cfg: QCConfig) -> dict:
    """Run the selected analyzers and return `{section_name: payload}`."""
    from qc.layout import discover_shards

    shards = discover_shards(cfg.output_root, cfg.shards)
    sections: dict = {
        "shards": {
            "discovered": [s.name for s in shards],
            "status": {s.name: s.status for s in shards},
        }
    }
    if not shards:
        sections["shards"]["warning"] = (
            f"no shard subdirectories found under {cfg.output_root}; "
            "is --output pointing at the pipeline_v3 output root?"
        )
        return sections

    if cfg.wants("yield"):
        from qc.analyzers import yield_rate

        logger.info("qc_step_start yield")
        sections["yield"] = yield_rate.analyze(cfg, shards)

    if cfg.wants("duration"):
        from qc.analyzers import duration_dist

        logger.info("qc_step_start duration")
        sections["duration"] = duration_dist.analyze(cfg, shards)

    # Structural mergeability is parquet-only; the embedding half of it needs
    # the GPU pass, so it is computed first and enriched afterwards.
    merge_state = None
    try:
        if cfg.wants("merge"):
            from qc.analyzers import mergeability

            logger.info("qc_step_start merge")
            merge_state = mergeability.analyze_structure(cfg, shards)

        if cfg.needs_gpu_pass:
            from qc.analyzers import recheck_pass

            logger.info("qc_step_start recheck")
            recheck = recheck_pass.run(cfg, shards, merge_state)
            if cfg.wants("speaker"):
                sections["speaker"] = recheck["speaker"]
            if cfg.wants("background"):
                sections["background"] = recheck["background"]

        if merge_state is not None:
            from qc.analyzers import mergeability

            sections["merge"] = mergeability.finalize(cfg, merge_state)
    finally:
        # The hash buckets are a large intermediate (a copy of every kept
        # segment's geometry) and are never resumable, unlike the GPU verdict
        # cache -- so they go as soon as the pass is done, even on failure.
        if merge_state is not None and not cfg.keep_work:
            from qc.analyzers import mergeability

            mergeability.cleanup(merge_state)

    return sections
