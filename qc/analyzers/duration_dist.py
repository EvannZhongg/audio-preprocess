"""Requirement 2: the duration distribution of the final output.

Two bucketings are reported side by side, deliberately:

  * coarse (<4 / 4-8 / 8-15 / >=15s) reproduces tmp/stat_parquet.py:30-37, so
    QC's headline numbers are directly comparable with the figures already
    circulated from that ad-hoc script.
  * fine (0-3 / 3-5 / 5-7 / 7-9 / 9-12 / 12-15 / 15+s) reproduces
    misc/analyze_output.py:61-63, which is the granularity TTS training data
    mixes are usually reasoned about.

Both are given by segment count *and* by summed duration, because they tell
different stories: short segments dominate the count while long ones dominate
the hours, so a count-only view systematically understates how much of the
corpus is long-form audio.

Percentiles come from a 600-bin histogram rather than from the raw values. Exact
quantiles would require holding every duration in memory; at ~10ms bin width the
error is invisible at the precision a P90 is read.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import logger
from qc.accumulators import AnomalyCounter, DurationStats
from qc.config import QCConfig
from qc.layout import ShardLayout
from qc.loaders import ParquetReadError, read_columns, stage2_kept_mask


def _scan_part(path: str, min_length: float, max_length: float) -> tuple[DurationStats, Optional[str]]:
    stats = DurationStats()
    try:
        table = read_columns(path, ["seg_duration", "language", "error",
                                    "dropped_by_silence", "dropped_by_alignment",
                                    "dropped_by_text_quality",
                                    "dropped_by_speaking_rate",
                                    "dropped_by_asr_validation"])
    except ParquetReadError as exc:
        stats.parts_failed = 1
        return stats, str(exc)
    stats.parts_read = 1

    kept = stage2_kept_mask(table).to_pylist()
    durations = table["seg_duration"].to_pylist()
    languages = table["language"].to_pylist()
    for keep, dur, lang in zip(kept, durations, languages):
        if not keep or dur is None:
            continue
        stats.add(float(dur), lang, min_length, max_length)
    return stats, None


def analyze(cfg: QCConfig, shards: list[ShardLayout]) -> dict:
    """Duration distribution of stage-2 kept segments, overall and per shard."""
    th = cfg.thresholds
    anomalies = AnomalyCounter()
    overall = DurationStats()
    per_shard: dict[str, DurationStats] = {}

    jobs = [
        (shard.name, part)
        for shard in shards
        for part in shard.stage2_parts
    ]
    if not jobs:
        return {
            "overall": overall.to_dict(),
            "per_shard": {},
            "notes": ["No stage-2 parquet found: the final output does not exist yet, "
                      "so there is no duration distribution to report."],
            "anomalies": anomalies.to_dict(),
        }

    logger.info(f"qc_duration_scan parts {len(jobs)} workers {cfg.workers}")
    with ProcessPoolExecutor(max_workers=min(cfg.workers, len(jobs))) as pool:
        futures = {
            pool.submit(_scan_part, part, th.min_segment_length, th.max_segment_length):
                (shard_name, part)
            for shard_name, part in jobs
        }
        done = 0
        for fut in as_completed(futures):
            shard_name, part = futures[fut]
            try:
                stats, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                anomalies.add("worker_error", f"{part}: {exc}")
                continue
            if err:
                anomalies.add("part_unreadable", err)
            overall.merge(stats)
            per_shard.setdefault(shard_name, DurationStats()).merge(stats)
            done += 1
            if done % 200 == 0:
                logger.info(f"qc_duration_progress {done}/{len(jobs)}")

    notes = [
        f"Buckets are measured against the production bounds from {th.source}: "
        f"min_segment_length={th.min_segment_length}s, "
        f"max_segment_length={th.max_segment_length}s."
    ]
    if overall.below_min_length:
        notes.append(
            f"{overall.below_min_length} kept segment(s) are shorter than "
            f"min_segment_length ({th.min_segment_length}s). Segmenter filters on length "
            "BEFORE the grace period is applied (pipeline_v2/steps/segment.py:65-74), so a "
            "small number of marginally-short segments is expected; a large number suggests "
            "the thresholds in --config do not match the run that produced this data."
        )
    if overall.above_max_length:
        notes.append(
            f"{overall.above_max_length} kept segment(s) exceed max_segment_length "
            f"({th.max_segment_length}s), which Segmenter should have split or dropped "
            "(pipeline_v2/steps/segment.py:107-114). Worth investigating."
        )
    unknown = overall.by_language.get("unknown")
    if unknown and overall.hist.n and unknown["count"] / overall.hist.n > 0.5:
        notes.append(
            "More than half of the kept segments have no language recorded, so the "
            "per-language split is not meaningful for this run."
        )

    return {
        "overall": overall.to_dict(),
        "per_shard": {name: s.to_dict() for name, s in sorted(per_shard.items())},
        "notes": notes,
        "anomalies": anomalies.to_dict(),
    }
