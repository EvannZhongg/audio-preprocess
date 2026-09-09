"""Requirement 1: how much usable data survived each stage.

Three levels, all measured two ways (files and seconds) because they answer
different questions and diverge sharply:

  level 0  the raw corpus, from the build_manifest.py manifest
  level 1  stage-1 segments (`segments_part-*.parquet`)
  level 2  stage-2 kept segments (`stage2_segments_part-*.parquet`)

The divergence is not noise, it is a real property of the pipeline: a stage-1
file fails as a *unit*, and when it does the actor discards the segments of its
already-successful chunks too (pipeline_v2_ray/actors/v2_stage_1.py:134-144,
`segments=records if success else []`). So an 8-hour file that failed on its
last chunk contributes zero seconds while counting as one failed file. Reporting
only one of the two ratios would therefore either hide or exaggerate the loss,
which is why both are always emitted and the mechanism is spelled out in the
report's notes.
"""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import logger
from qc.accumulators import (AnomalyCounter, RawLevel, Stage1Level, Stage2Level,
                             YieldFunnel)
from qc.config import QCConfig
from qc.layout import ShardLayout, discover_manifest_parts
from qc.loaders import (DROP_COLUMNS, ParquetReadError, all_true_mask,
                        classify_error, error_mask, mask_count, masked_sum,
                        read_columns, stage1_error_mask, stage1_valid_mask,
                        stage2_drop_mask, stage2_kept_mask,
                        stage2_retriable_error_mask)

# Cap on how many distinct ids a worker tracks. Above this the exact file count
# is dropped (reported as None) rather than letting a pathological shard blow up
# a worker's memory -- the ratios matter, an exact id set does not.
_IDENTITY_CAP = 4_000_000


def _scan_manifest_part(path: str) -> tuple[RawLevel, Optional[str]]:
    level = RawLevel()
    try:
        table = read_columns(path, ["relative_path", "duration"])
    except ParquetReadError as exc:
        level.parts_failed = 1
        return level, str(exc)
    level.parts_read = 1
    level.files = table.num_rows
    durations = table["duration"].to_pylist()
    for d in durations:
        if d is None or d <= 0.0:
            # build_manifest.py writes 0.0 when mutagen could not read a header
            # (source_scan/manifest.py:148-157). Those files were still fed to
            # the pipeline, so they count toward the file total but must not be
            # silently treated as zero-length input.
            level.files_unknown_duration += 1
            continue
        level.duration += float(d)
    return level, None


def _scan_stage1_part(path: str) -> tuple[Stage1Level, Optional[str]]:
    level = Stage1Level()
    try:
        table = read_columns(
            path, ["source", "chunk_audio_path", "seg_duration", "error"]
        )
    except ParquetReadError as exc:
        level.parts_failed = 1
        return level, str(exc)
    level.parts_read = 1
    level.rows = table.num_rows

    valid = stage1_valid_mask(table)
    failed = stage1_error_mask(table)
    level.valid_segments = mask_count(valid)
    level.failed_file_rows = mask_count(failed)
    level.valid_duration = masked_sum(table, "seg_duration", valid)

    sources = table["source"].to_pylist()
    errors = table["error"].to_pylist()
    chunks = table["chunk_audio_path"].to_pylist()
    for src, err, chunk in zip(sources, errors, chunks):
        if err:
            level.error_types[classify_error(err)] += 1
            if src:
                level.failed_sources.add(src)
            continue
        if src:
            level.ok_sources.add(src)
        if chunk:
            level.chunk_paths.add(chunk)
    if (len(level.ok_sources) + len(level.failed_sources) + len(level.chunk_paths)
            > _IDENTITY_CAP):
        level.track_identities = False
        level.ok_sources = set()
        level.failed_sources = set()
        level.chunk_paths = set()
    return level, None


def _scan_stage2_part(path: str) -> tuple[Stage2Level, Optional[str]]:
    level = Stage2Level()
    try:
        table = read_columns(path, ["seg_duration", "error", *DROP_COLUMNS])
    except ParquetReadError as exc:
        level.parts_failed = 1
        return level, str(exc)
    level.parts_read = 1
    level.rows = table.num_rows
    level.total_duration = masked_sum(
        table, "seg_duration", all_true_mask(table.num_rows)
    )

    kept = stage2_kept_mask(table)
    level.kept_rows = mask_count(kept)
    level.kept_duration = masked_sum(table, "seg_duration", kept)

    err = error_mask(table)
    level.error_rows = mask_count(err)
    level.error_duration = masked_sum(table, "seg_duration", err)
    level.retriable_error_rows = mask_count(stage2_retriable_error_mask(table))

    # Not mutually exclusive: one segment can trip several filters, so these
    # columns sum to more than the dropped total. Reported as-is (same as
    # tmp/stat_stage2.py:55) because "which filter is costing us the most" is
    # the question they answer.
    for name in DROP_COLUMNS:
        m = stage2_drop_mask(table, name)
        level.drop_rows[name] = mask_count(m)
        level.drop_duration[name] = masked_sum(table, "seg_duration", m)

    if level.error_rows:
        for e in table["error"].to_pylist():
            if e:
                level.error_types[classify_error(e)] += 1
    level.kept_utt_ids = level.kept_rows
    return level, None


def analyze(cfg: QCConfig, shards: list[ShardLayout]) -> dict:
    """Build the per-shard and overall funnels."""
    anomalies = AnomalyCounter()
    per_shard: dict[str, YieldFunnel] = {s.name: YieldFunnel() for s in shards}
    overall = YieldFunnel()

    jobs: list[tuple[str, str, str]] = []  # (kind, shard_name, path)
    for shard in shards:
        for part in shard.stage1_parts:
            jobs.append(("stage1", shard.name, part))
        for part in shard.stage2_parts:
            jobs.append(("stage2", shard.name, part))

    manifest_parts = discover_manifest_parts(cfg.manifest)
    if cfg.manifest and not manifest_parts:
        anomalies.add("manifest_not_found", cfg.manifest)
    for part in manifest_parts:
        jobs.append(("manifest", "<raw>", part))

    scanners = {
        "manifest": _scan_manifest_part,
        "stage1": _scan_stage1_part,
        "stage2": _scan_stage2_part,
    }
    total = len(jobs)
    logger.info(f"qc_yield_scan parts {total} workers {cfg.workers}")
    done = 0
    if not jobs:
        return _assemble(cfg, shards, per_shard, overall, anomalies, bool(manifest_parts))

    with ProcessPoolExecutor(max_workers=min(cfg.workers, max(1, total))) as pool:
        futures = {
            pool.submit(scanners[kind], path): (kind, shard_name, path)
            for kind, shard_name, path in jobs
        }
        for fut in as_completed(futures):
            kind, shard_name, path = futures[fut]
            try:
                level, err = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad part never sinks the pass
                anomalies.add(f"{kind}_worker_error", f"{path}: {exc}")
                continue
            if err:
                anomalies.add(f"{kind}_part_unreadable", err)
            if kind == "manifest":
                overall.raw.merge(level)
            elif kind == "stage1":
                per_shard[shard_name].stage1.merge(level)
                overall.stage1.merge(level)
            else:
                per_shard[shard_name].stage2.merge(level)
                overall.stage2.merge(level)
            done += 1
            if done % 200 == 0:
                logger.info(f"qc_yield_progress {done}/{total}")

    return _assemble(cfg, shards, per_shard, overall, anomalies, bool(manifest_parts))


def _assemble(cfg: QCConfig, shards: list[ShardLayout],
              per_shard: dict[str, YieldFunnel], overall: YieldFunnel,
              anomalies: AnomalyCounter, has_manifest: bool) -> dict:
    notes = []
    if not has_manifest:
        notes.append(
            "No manifest supplied, so the level-0 (raw corpus) baseline is missing: "
            "stage-1 yield is reported in absolute terms only. Pass --manifest to get "
            "end-to-end rates."
        )
    notes.append(
        "Stage-1 failures are per FILE, not per chunk: when any chunk of a file fails, "
        "the actor discards that file's already-successful segments too "
        "(pipeline_v2_ray/actors/v2_stage_1.py:134-144). Stage-1 yield measured in "
        "seconds is therefore structurally lower than measured in files, and the gap is "
        "an upper bound on what a chunk-level retry could recover."
    )
    if overall.stage2.retriable_error_rows:
        notes.append(
            f"{overall.stage2.retriable_error_rows} stage-2 rows failed with "
            "'asr_access_failed' -- a transient remote-ASR outage that the next run "
            "reprocesses (pipeline_v2_ray/stage2_segments.py:89-98). They are recoverable "
            "yield, not permanent loss."
        )
    empty = [s.name for s in shards if s.status == "empty"]
    stage1_only = [s.name for s in shards if s.status == "stage1_only"]
    if stage1_only:
        notes.append(
            f"{len(stage1_only)} shard(s) have stage-1 output but no stage-2 parquet, i.e. "
            f"stage 2 has not run there yet: {', '.join(stage1_only[:5])}"
            + (" ..." if len(stage1_only) > 5 else "")
        )
    if empty:
        notes.append(f"{len(empty)} shard directory(ies) contain no parquet at all.")

    return {
        "overall": overall.to_dict(),
        "per_shard": {
            name: funnel.to_dict()
            for name, funnel in per_shard.items()
            if funnel.stage1.rows or funnel.stage2.rows
        },
        "shard_status": {s.name: s.status for s in shards},
        "notes": notes,
        "anomalies": anomalies.to_dict(),
    }
