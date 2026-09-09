"""Drives the one GPU pass that answers requirements 3, 4 and 5's embedding half.

Sequence:

  1. stream stage-2 kept segments and sample them deterministically
  2. group the sample (plus the mergeability pass's candidate pairs) by chunk
  3. drop whatever the cache already holds, so a rerun resumes
  4. fan the groups out over model-bearing workers
  5. reduce the cache into the report accumulators

Steps 3 and 5 both go through the cache rather than through worker return
values, deliberately: reduction then behaves identically whether the verdicts
were produced by this run or recovered from an interrupted one, so there is a
single aggregation path instead of a fresh-run one and a resumed one that could
disagree.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import logger
from qc.accumulators import AnomalyCounter
from qc.config import QCConfig
from qc.gpu_worker import ChunkGroup, PairTask, SegmentTask
from qc.layout import ShardLayout
from qc.loaders import ParquetReadError, read_columns, stage2_kept_mask
from qc.sampling import SampleInfo, SmallestNSampler

# Chunks per dispatched batch. Small enough that progress is visible and a
# crashed worker loses little, large enough that IPC overhead stays irrelevant.
_GROUPS_PER_BATCH = 8


def _collect_segment_candidates(part: str, shard: str) -> tuple[list, int, Optional[str]]:
    """Kept stage-2 segments of one part, as re-check candidates."""
    try:
        table = read_columns(part, [
            "utt_id", "chunk_audio_path", "speaker_id", "start", "end",
            "seg_duration", "error", "dropped_by_silence", "dropped_by_alignment",
            "dropped_by_text_quality", "dropped_by_speaking_rate",
            "dropped_by_asr_validation",
        ])
    except ParquetReadError as exc:
        return [], 0, str(exc)

    kept = stage2_kept_mask(table).to_pylist()
    cols = {
        name: table[name].to_pylist()
        for name in ("utt_id", "chunk_audio_path", "speaker_id", "start", "end",
                     "seg_duration")
    }
    out = []
    for i, keep in enumerate(kept):
        if not keep:
            continue
        utt = cols["utt_id"][i]
        chunk = cols["chunk_audio_path"][i]
        start, end = cols["start"][i], cols["end"][i]
        if not utt or not chunk or start is None or end is None:
            continue
        out.append({
            "utt_id": utt, "shard": shard, "chunk_audio_path": chunk,
            "start": float(start), "end": float(end),
            "seg_duration": float(cols["seg_duration"][i] or (end - start)),
            "speaker_id": cols["speaker_id"][i],
        })
    return out, len(out), None


def _recorded_metrics(cfg: QCConfig, shards: list[ShardLayout],
                      wanted_utt_ids: set) -> dict:
    """Production's own dnsmos/c50/snr/min_similarity for the sampled segments.

    Read from stage-1 parquet (stage 2's schema does not carry them) and only for
    the sample, so the lookup dict stays small. Without these there is no drift
    comparison -- and drift is what tells you whether a difference between QC and
    production is a real quality issue or just a measurement difference.
    """
    if not wanted_utt_ids:
        return {}
    found: dict = {}
    for shard in shards:
        for part in shard.stage1_parts:
            try:
                table = read_columns(part, [
                    "utt_id", "speaker_min_similarity", "dnsmos", "c50", "snr", "error",
                ])
            except ParquetReadError:
                continue
            utts = table["utt_id"].to_pylist()
            sims = table["speaker_min_similarity"].to_pylist()
            dns = table["dnsmos"].to_pylist()
            c50s = table["c50"].to_pylist()
            snrs = table["snr"].to_pylist()
            for i, utt in enumerate(utts):
                if utt and utt in wanted_utt_ids and utt not in found:
                    found[utt] = {
                        "rec_min_similarity": sims[i],
                        "rec_dnsmos": dns[i],
                        "rec_c50": c50s[i],
                        "rec_snr": snrs[i],
                    }
            if len(found) >= len(wanted_utt_ids):
                return found
    return found


def _build_segment_tasks(cfg: QCConfig, shards: list[ShardLayout],
                         anomalies: AnomalyCounter) -> tuple[list, SampleInfo]:
    """Sample the final output, per shard or globally."""
    parts = [(s.name, p) for s in shards for p in s.stage2_parts]
    if not parts:
        return [], SampleInfo(mode="none")

    per_shard_rows: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=min(cfg.workers, len(parts))) as pool:
        futures = {
            pool.submit(_collect_segment_candidates, part, shard): (shard, part)
            for shard, part in parts
        }
        for fut in as_completed(futures):
            shard, part = futures[fut]
            try:
                rows, _, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                anomalies.add("candidate_worker_error", f"{part}: {exc}")
                continue
            if err:
                anomalies.add("part_unreadable", err)
            per_shard_rows.setdefault(shard, []).extend(rows)

    info = SampleInfo(mode="full" if cfg.sample_n <= 0 else f"smallest-{cfg.sample_n}")
    picked: list = []
    if cfg.sample_per_shard:
        # Per shard so a big shard cannot crowd the others out of the sample.
        for shard, rows in per_shard_rows.items():
            sampler = SmallestNSampler(cfg.sample_n)
            for row in rows:
                sampler.offer(row["utt_id"], row)
            picked.extend(sampler.result())
            info.merge(sampler.info())
    else:
        sampler = SmallestNSampler(cfg.sample_n)
        for rows in per_shard_rows.values():
            for row in rows:
                sampler.offer(row["utt_id"], row)
        picked = sampler.result()
        info.merge(sampler.info())

    recorded = _recorded_metrics(cfg, shards, {r["utt_id"] for r in picked})
    tasks = [
        SegmentTask(
            utt_id=row["utt_id"], shard=row["shard"],
            chunk_audio_path=row["chunk_audio_path"],
            start=row["start"], end=row["end"], seg_duration=row["seg_duration"],
            speaker_id=row["speaker_id"],
            **recorded.get(row["utt_id"], {}),
        )
        for row in picked
    ]
    return tasks, info


def _build_pair_tasks(merge_state) -> list:
    if merge_state is None or not merge_state.pair_candidates:
        return []
    return [
        PairTask(
            pair_key=c["pair_key"], shard=c["shard"] or "<unknown>",
            chunk_audio_path=c["chunk_audio_path"],
            prev_start=c["prev"]["start"], prev_end=c["prev"]["end"],
            cur_start=c["cur"]["start"], cur_end=c["cur"]["end"],
        )
        for c in merge_state.pair_candidates
    ]


def _group_by_chunk(segment_tasks: list, pair_tasks: list) -> list:
    """Bundle work by chunk wav so each file is decoded exactly once."""
    groups: dict[str, ChunkGroup] = {}
    for task in segment_tasks:
        g = groups.get(task.chunk_audio_path)
        if g is None:
            g = ChunkGroup(task.chunk_audio_path, task.shard)
            groups[task.chunk_audio_path] = g
        g.segments.append(task)
    for pair in pair_tasks:
        g = groups.get(pair.chunk_audio_path)
        if g is None:
            g = ChunkGroup(pair.chunk_audio_path, pair.shard)
            groups[pair.chunk_audio_path] = g
        g.pairs.append(pair)
    # Biggest chunks first: with a handful of workers, starting the long jobs
    # early keeps the tail from being one worker finishing alone.
    return sorted(groups.values(), key=lambda g: -g.size)


def _resolve_devices(cfg: QCConfig) -> tuple[list[str], list[str]]:
    """Assign a device per worker, falling back to CPU when CUDA is absent."""
    warnings: list[str] = []
    devices = list(cfg.devices)
    wants_cuda = any(d.startswith("cuda") for d in devices)
    if wants_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                warnings.append(
                    "CUDA requested but torch.cuda.is_available() is False; falling back "
                    "to CPU. Model re-checks on CPU are far slower -- consider a smaller "
                    "--sample-n."
                )
                devices = ["cpu"]
            else:
                count = torch.cuda.device_count()
                usable = [
                    d for d in devices
                    if not d.startswith("cuda:") or int(d.split(":", 1)[1]) < count
                ]
                if not usable:
                    warnings.append(
                        f"none of --devices {cfg.devices} exist (torch sees {count} GPU(s)); "
                        "falling back to cuda:0"
                    )
                    usable = ["cuda:0"]
                elif len(usable) != len(devices):
                    warnings.append(
                        f"ignoring device(s) beyond the {count} GPU(s) torch can see"
                    )
                devices = usable
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not query CUDA ({exc}); falling back to CPU")
            devices = ["cpu"]
    assigned = [devices[i % len(devices)] for i in range(cfg.gpu_workers)]
    return assigned, warnings


def _worker_entry(cfg, devices: list, need_speaker: bool,
                  need_background: bool, need_embedding: bool, groups: list) -> dict:
    """One worker process: load models once, then drain its groups.

    The device is claimed from a shared queue on first use rather than passed in
    per batch. `ProcessPoolExecutor` gives no guarantee about which process picks
    up which task, so deriving the device from a batch index would let every
    process end up on the same card while the others idled -- and would also let a
    process switch identity between batches, writing to another worker's cache
    file. Claiming a slot once, on first call, pins one device (and one cache
    file) to one process for its whole life.

    Models live in the process's module state, so second and later batches handed
    to the same process skip loading entirely.
    """
    from qc import gpu_worker

    if not gpu_worker._STATE:
        slot = _claim_slot(len(devices))
        gpu_worker.init_worker(
            cfg, slot, devices[slot % len(devices)],
            need_speaker, need_background, need_embedding,
        )
    totals = gpu_worker.run_groups(groups)
    totals["summary"] = gpu_worker.worker_summary()
    return totals


# Shared across the worker processes; each claims one slot on startup.
_SLOT_QUEUE = None


def _init_slot_queue(queue) -> None:
    global _SLOT_QUEUE
    _SLOT_QUEUE = queue


def _claim_slot(n_devices: int) -> int:
    """Take the next free worker slot, falling back to a PID-derived one.

    The fallback only matters if the queue is somehow exhausted (more processes
    than slots); a distinct-per-process value still keeps cache files separate,
    which is the property that must not break.
    """
    if _SLOT_QUEUE is not None:
        try:
            return int(_SLOT_QUEUE.get_nowait())
        except Exception:  # noqa: BLE001 - empty queue or a closed manager
            pass
    return os.getpid() % max(1, n_devices)


def run(cfg: QCConfig, shards: list[ShardLayout], merge_state) -> dict:
    """Execute the GPU pass and reduce it into report sections."""
    from qc.analyzers import background_recheck, speaker_recheck
    from qc.cache import load_done_ids

    anomalies = AnomalyCounter()
    need_speaker = cfg.wants("speaker")
    need_background = cfg.wants("background")
    need_embedding = cfg.wants("merge")

    segment_tasks: list = []
    sample_info = SampleInfo(mode="none")
    if need_speaker or need_background:
        segment_tasks, sample_info = _build_segment_tasks(cfg, shards, anomalies)
    pair_tasks = _build_pair_tasks(merge_state) if need_embedding else []

    if not segment_tasks and not pair_tasks:
        return _empty_result(cfg, sample_info, anomalies,
                             "no segments or pairs to re-check", merge_state)

    if not cfg.config_path:
        # The models need paths, a pyannote cache and an auth token, all of which
        # live in the production config. Better to say so plainly than to emit a
        # section full of nulls.
        msg = ("model re-checks need --config (model paths and the pyannote cache come "
               "from the production config json); skipping requirements 3/4 and the "
               "embedding half of requirement 5")
        logger.warning(f"qc_recheck_skipped {msg}")
        anomalies.add("skipped_no_config", msg)
        return _empty_result(cfg, sample_info, anomalies, msg, merge_state)

    done_ids = load_done_ids(cfg.work_dir)
    if done_ids:
        before_seg, before_pair = len(segment_tasks), len(pair_tasks)
        segment_tasks = [t for t in segment_tasks if t.utt_id not in done_ids]
        pair_tasks = [p for p in pair_tasks if p.pair_key not in done_ids]
        logger.info(
            f"qc_recheck_resume cached {len(done_ids)} skipping "
            f"{before_seg - len(segment_tasks)} segments "
            f"{before_pair - len(pair_tasks)} pairs"
        )

    groups = _group_by_chunk(segment_tasks, pair_tasks)
    devices, device_warnings = _resolve_devices(cfg)
    for w in device_warnings:
        logger.warning(f"qc_recheck_device {w}")
        anomalies.add("device_fallback", w)

    if groups:
        logger.info(
            f"qc_recheck_start chunks {len(groups)} segments {len(segment_tasks)} "
            f"pairs {len(pair_tasks)} workers {cfg.gpu_workers} devices {devices}"
        )
        _dispatch(cfg, groups, devices, need_speaker, need_background,
                  need_embedding, anomalies)
    else:
        logger.info("qc_recheck_all_cached nothing left to score")

    result: dict = {"sampling": sample_info.to_dict()}
    result["speaker"] = (
        speaker_recheck.reduce(cfg, sample_info, anomalies)
        if need_speaker else None
    )
    result["background"] = (
        background_recheck.reduce(cfg, sample_info, anomalies)
        if need_background else None
    )
    if need_embedding and merge_state is not None:
        _apply_pair_similarity(cfg, merge_state)
        # Fold this pass's anomalies into the merge state as well. Without this,
        # a run of `--steps merge` alone would report "N pairs failed to score"
        # with the actual cause (model load failure, CPU fallback, ...) recorded
        # only in sections the user never asked for.
        merge_state.anomalies.merge(anomalies)
    return result


def _dispatch(cfg: QCConfig, groups: list, devices: list[str], need_speaker: bool,
              need_background: bool, need_embedding: bool,
              anomalies: AnomalyCounter) -> None:
    """Fan chunk groups out over the model workers."""
    import multiprocessing as mp

    batches = [
        groups[i:i + _GROUPS_PER_BATCH]
        for i in range(0, len(groups), _GROUPS_PER_BATCH)
    ]
    total_items = sum(g.size for g in groups)
    done_items = 0

    # One slot per worker, claimed on startup so each process owns exactly one
    # device and one cache file for its lifetime.
    slot_queue = mp.Manager().Queue()
    for slot in range(cfg.gpu_workers):
        slot_queue.put(slot)

    # max_workers == gpu_workers keeps one model set per process; a larger pool
    # would load the models more times than there are GPUs to use them.
    with ProcessPoolExecutor(
        max_workers=cfg.gpu_workers,
        initializer=_init_slot_queue,
        initargs=(slot_queue,),
    ) as pool:
        futures = {
            pool.submit(_worker_entry, cfg, devices, need_speaker,
                        need_background, need_embedding, batch): i
            for i, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_index = futures[fut]
            try:
                totals = fut.result()
            except Exception as exc:  # noqa: BLE001
                anomalies.add("gpu_worker_error", f"batch {batch_index}: {exc}")
                logger.error(f"qc_recheck_worker_failed batch {batch_index} {exc}")
                continue
            done_items += totals["segments"] + totals["pairs"]
            for kind, n in totals.get("errors", {}).items():
                anomalies.add(kind, f"batch {batch_index}")
                if n > 1:
                    # `add` already counted one; fold in the rest without
                    # duplicating the example detail.
                    anomalies.counts[kind] += n - 1
            summary = totals.get("summary") or {}
            for w in summary.get("load_errors", []):
                anomalies.add("model_load_error", w)
            for w in summary.get("constant_warnings", []):
                anomalies.add("production_constant_changed", w)
            logger.info(f"qc_recheck_progress {done_items}/{total_items} items")


def _apply_pair_similarity(cfg: QCConfig, merge_state) -> None:
    """Fold the sampled pair similarities into the mergeability stats."""
    from qc.cache import iter_records

    th = cfg.thresholds
    stats = merge_state.stats
    for rec in iter_records(cfg.work_dir, kind="pair"):
        sim = rec.get("similarity")
        if sim is None:
            stats.sim_failed_pairs += 1
            continue
        stats.sim_checked_pairs += 1
        stats.sim_hist.add(float(sim))
        if float(sim) >= th.intra_similarity_threshold:
            stats.sim_pass_pairs += 1


def _empty_result(cfg: QCConfig, sample_info: SampleInfo,
                  anomalies: AnomalyCounter, reason: str, merge_state) -> dict:
    """Result for a pass that could not run, carrying the reason everywhere.

    The reason is also pushed into `merge_state`, so a report that shows
    "N pairs failed to score" always shows why right next to it instead of
    leaving the number unexplained.
    """
    if merge_state is not None:
        merge_state.notes.append(f"Embedding similarity was not measured: {reason}")
        merge_state.anomalies.merge(anomalies)
    note = {"skipped": True, "reason": reason,
            "sampling": sample_info.to_dict(),
            "anomalies": anomalies.to_dict()}
    return {
        "sampling": sample_info.to_dict(),
        "speaker": dict(note) if cfg.wants("speaker") else None,
        "background": dict(note) if cfg.wants("background") else None,
    }
