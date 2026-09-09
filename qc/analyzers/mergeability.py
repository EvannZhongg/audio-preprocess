"""Requirement 5: adjacent segments that look like they should have been merged.

The production merge rule (pipeline_v2/steps/segment.py:120-140) requires ALL of:

  1. same `speaker`
  2. both segments carry a `reference_embedding`
  3. `cosine_similarity(prev.emb, cur.emb) >= intra_similarity_threshold`
  4. `gap = cur.start - prev.end < merge_gap`
  5. `merged_dur = cur.end - prev.start < max_segment_length`

Conditions 1, 4 and 5 are readable straight from parquet. Condition 2 is
*inferable*: `EmbeddingRefiner` skips embedding for spans under 1s
(embedding_refinement.py:85-90), so a sub-1s segment provably had
`reference_embedding=None` and could never merge -- that is a structural
property of the pipeline, not a missed opportunity, and it is reported as its
own category so it never inflates the "missed merges" number. Condition 3 needs
a GPU, so it is measured on a sample and reported as a separate pass rate.

Two subtleties that would silently corrupt the counts if ignored:

  * The relevant adjacency is between *stage-2 surviving* segments, not stage-1
    ones. Stage 2 drops segments, so two segments that were not neighbours in
    stage 1 can become neighbours in the final output. Reading stage-1 parquet
    here would answer a question nobody asked.
  * Segments from one chunk can straddle two `stage2_segments_part-*.parquet`
    files, because the driver flushes every 100k rows regardless of chunk
    boundaries. Grouping within a single part would therefore lose the pairs at
    every flush boundary. Hence the two-pass hash-bucket shuffle below: pass 1
    re-partitions by `hash(chunk_audio_path)`, guaranteeing that a chunk's
    segments all land in one bucket; pass 2 groups and sorts inside each bucket.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import logger
from qc.accumulators import AnomalyCounter, MergeabilityStats
from qc.config import EMBED_MIN_SEGMENT_S, QCConfig
from qc.layout import ShardLayout
from qc.loaders import ParquetReadError, read_columns, stage2_kept_mask
from qc.sampling import SmallestNSampler

_BUCKET_COLUMNS = (
    "utt_id", "chunk_audio_path", "speaker_id", "start", "end", "seg_duration",
)


def _bucket_of(chunk_audio_path: str, buckets: int) -> int:
    """Stable bucket id for a chunk path.

    SHA-1 rather than `hash()`: Python salts string hashing per process, so with
    `hash()` two worker processes would disagree about where a chunk belongs and
    the shuffle would split chunks apart -- exactly the bug the shuffle exists
    to prevent.
    """
    digest = hashlib.sha1(chunk_audio_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def _shuffle_part(path: str, work_dir: str, buckets: int, shard: str) -> tuple[dict, Optional[str]]:
    """Pass 1: read one stage-2 part, write its kept rows into hash buckets."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    stats = {"rows": 0, "parts_read": 0, "parts_failed": 0, "skipped_no_chunk": 0}
    try:
        table = read_columns(path, [
            "utt_id", "chunk_audio_path", "speaker_id", "start", "end",
            "seg_duration", "error", "dropped_by_silence", "dropped_by_alignment",
            "dropped_by_text_quality", "dropped_by_speaking_rate",
            "dropped_by_asr_validation",
        ])
    except ParquetReadError as exc:
        stats["parts_failed"] = 1
        return stats, str(exc)
    stats["parts_read"] = 1

    kept = stage2_kept_mask(table).to_pylist()
    cols = {name: table[name].to_pylist() for name in _BUCKET_COLUMNS}

    per_bucket: dict[int, list[dict]] = {}
    for i, keep in enumerate(kept):
        if not keep:
            continue
        chunk = cols["chunk_audio_path"][i]
        if not chunk:
            stats["skipped_no_chunk"] += 1
            continue
        start = cols["start"][i]
        end = cols["end"][i]
        if start is None or end is None:
            stats["skipped_no_chunk"] += 1
            continue
        row = {
            "utt_id": cols["utt_id"][i],
            "chunk_audio_path": chunk,
            "speaker_id": cols["speaker_id"][i],
            "start": float(start),
            "end": float(end),
            "seg_duration": float(cols["seg_duration"][i] or (end - start)),
            "shard": shard,
        }
        per_bucket.setdefault(_bucket_of(chunk, buckets), []).append(row)
        stats["rows"] += 1

    schema = pa.schema([
        ("utt_id", pa.string()), ("chunk_audio_path", pa.string()),
        ("speaker_id", pa.string()), ("start", pa.float64()),
        ("end", pa.float64()), ("seg_duration", pa.float64()),
        ("shard", pa.string()),
    ])
    # One file per (bucket, source part) so parallel writers never contend, and
    # a crashed pass leaves only whole files behind.
    token = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    for bucket, rows in per_bucket.items():
        bucket_dir = os.path.join(work_dir, f"bucket-{bucket:04d}")
        os.makedirs(bucket_dir, exist_ok=True)
        out = os.path.join(bucket_dir, f"{token}.parquet")
        tmp = out + ".tmp"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp)
        os.replace(tmp, out)
    return stats, None


def _classify_pair(prev: dict, cur: dict, merge_gap: float,
                   max_segment_length: float) -> tuple[str, float]:
    """Which production merge condition blocks this pair, in check order.

    Order matters: it is a first-blocker-wins classification, so the categories
    partition the pairs (they sum to the pair total) instead of overlapping.
    """
    gap = cur["start"] - prev["end"]
    merged_dur = cur["end"] - prev["start"]
    if prev["speaker_id"] != cur["speaker_id"]:
        return "different_speaker", gap
    # Condition 2 of the production rule, inferred: EmbeddingRefiner leaves
    # `reference_embedding=None` on spans under 1s, and Segmenter refuses to
    # merge a pair when either side lacks one.
    if (prev["seg_duration"] < EMBED_MIN_SEGMENT_S
            or cur["seg_duration"] < EMBED_MIN_SEGMENT_S):
        return "no_embedding_short_segment", gap
    if gap >= merge_gap:
        return "gap_too_large", gap
    if merged_dur >= max_segment_length:
        return "merged_too_long", gap
    # Every condition production can check from geometry passes. Whether it
    # would ACTUALLY have merged still depends on embedding similarity, which
    # the sampled GPU pass measures separately.
    return "mergeable_by_structure", gap


def _process_bucket(bucket_dir: str, merge_gap: float, max_segment_length: float,
                    pair_sample_n: int) -> tuple[MergeabilityStats, list, Optional[str]]:
    """Pass 2: group one bucket by chunk, sort by start, classify each pair."""
    import glob

    import pyarrow.parquet as pq

    stats = MergeabilityStats()
    sampler = SmallestNSampler(pair_sample_n)
    files = sorted(glob.glob(os.path.join(bucket_dir, "*.parquet")))
    if not files:
        return stats, [], None

    rows: list[dict] = []
    try:
        for f in files:
            rows.extend(pq.read_table(f).to_pylist())
    except Exception as exc:  # noqa: BLE001
        stats.buckets_failed = 1
        return stats, [], f"{bucket_dir}: {type(exc).__name__}: {exc}"
    stats.buckets_read = 1

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["chunk_audio_path"], []).append(row)

    for chunk, segs in groups.items():
        stats.chunks += 1
        if len(segs) < 2:
            stats.single_segment_chunks += 1
            continue
        segs.sort(key=lambda r: r["start"])
        for prev, cur in zip(segs, segs[1:]):
            category, gap = _classify_pair(prev, cur, merge_gap, max_segment_length)
            stats.add_pair(category, prev["seg_duration"] + cur["seg_duration"], gap)
            if category != "mergeable_by_structure":
                continue
            # Only structurally-mergeable pairs are worth an embedding: for
            # every other category production would have refused regardless of
            # similarity, so scoring them would burn GPU on a foregone answer.
            key = f"{prev['utt_id']}|{cur['utt_id']}"
            sampler.offer(key, {
                "pair_key": key,
                "shard": prev.get("shard"),
                "chunk_audio_path": chunk,
                "prev": {"utt_id": prev["utt_id"], "start": prev["start"],
                         "end": prev["end"], "seg_duration": prev["seg_duration"]},
                "cur": {"utt_id": cur["utt_id"], "start": cur["start"],
                        "end": cur["end"], "seg_duration": cur["seg_duration"]},
            })
    return stats, sampler.result(), None


class MergeState:
    """Structural results plus the pair sample awaiting the GPU pass."""

    def __init__(self) -> None:
        self.stats = MergeabilityStats()
        self.pair_candidates: list = []
        self.anomalies = AnomalyCounter()
        self.notes: list[str] = []
        self.bucket_root: Optional[str] = None
        self.pair_population = 0


def analyze_structure(cfg: QCConfig, shards: list[ShardLayout]) -> MergeState:
    """Everything about requirement 5 that parquet alone can answer."""
    th = cfg.thresholds
    state = MergeState()

    parts = [(s.name, p) for s in shards for p in s.stage2_parts]
    if not parts:
        state.notes.append(
            "No stage-2 parquet found, so there is no final output whose adjacency "
            "could be analysed."
        )
        return state

    bucket_root = os.path.join(cfg.work_dir, "merge_buckets")
    shutil.rmtree(bucket_root, ignore_errors=True)
    os.makedirs(bucket_root, exist_ok=True)
    state.bucket_root = bucket_root

    logger.info(
        f"qc_merge_shuffle parts {len(parts)} buckets {cfg.merge_buckets} "
        f"workers {cfg.workers}"
    )
    with ProcessPoolExecutor(max_workers=min(cfg.workers, len(parts))) as pool:
        futures = {
            pool.submit(_shuffle_part, path, bucket_root, cfg.merge_buckets, shard): path
            for shard, path in parts
        }
        shuffled = 0
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                stats, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                state.anomalies.add("shuffle_worker_error", f"{path}: {exc}")
                continue
            if err:
                state.anomalies.add("part_unreadable", err)
            if stats["skipped_no_chunk"]:
                state.anomalies.add("row_without_chunk_path", str(stats["skipped_no_chunk"]))
            shuffled += stats["rows"]
        logger.info(f"qc_merge_shuffle_done rows {shuffled}")

    bucket_dirs = sorted(
        os.path.join(bucket_root, d) for d in os.listdir(bucket_root)
        if d.startswith("bucket-")
    )
    if not bucket_dirs:
        state.notes.append("No kept stage-2 segments to analyse for mergeability.")
        return state

    # Split the pair-sample budget across buckets so the sample is spread over
    # the corpus rather than concentrated in whichever bucket finished first.
    per_bucket_pairs = (
        0 if cfg.pair_sample_n <= 0
        else max(1, cfg.pair_sample_n // len(bucket_dirs))
    )
    logger.info(f"qc_merge_pairs buckets {len(bucket_dirs)} workers {cfg.workers}")
    with ProcessPoolExecutor(max_workers=min(cfg.workers, len(bucket_dirs))) as pool:
        futures = {
            pool.submit(_process_bucket, d, th.merge_gap, th.max_segment_length,
                        per_bucket_pairs): d
            for d in bucket_dirs
        }
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                stats, candidates, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                state.anomalies.add("bucket_worker_error", f"{d}: {exc}")
                continue
            if err:
                state.anomalies.add("bucket_unreadable", err)
            state.stats.merge(stats)
            state.pair_candidates.extend(candidates)

    state.pair_population = state.stats.structurally_mergeable
    logger.info(
        f"qc_merge_structure pairs {state.stats.pairs} "
        f"mergeable {state.stats.structurally_mergeable} "
        f"pair_sample {len(state.pair_candidates)}"
    )
    return state


def finalize(cfg: QCConfig, state: MergeState) -> dict:
    """Assemble the report section once the GPU pass has filled in similarity."""
    th = cfg.thresholds
    notes = list(state.notes)
    notes.append(
        "Categories are first-blocker-wins over production's merge conditions "
        "(pipeline_v2/steps/segment.py:120-140), so they partition the pairs."
    )
    notes.append(
        f"'no_embedding_short_segment' means at least one side is under "
        f"{EMBED_MIN_SEGMENT_S}s, for which EmbeddingRefiner never computes an embedding "
        "(pipeline_v2/steps/embedding_refinement.py:85-90). Such pairs can NEVER merge in "
        "production, so they are not missed merges."
    )
    notes.append(
        f"Gaps are measured on the final boundaries, which already include the grace "
        f"period (grace_period_end={th.grace_period_end}s, applied last in "
        "pipeline_v2/steps/segment.py:222-235). That is why many gaps are exactly 0."
    )
    if state.stats.negative_gap_pairs:
        notes.append(
            f"{state.stats.negative_gap_pairs} pair(s) have a negative gap (the grace "
            "period pushed one boundary past the next segment's start); they are clamped "
            "to 0 for the gap distribution."
        )
    if state.stats.sim_checked_pairs:
        notes.append(
            f"Embedding similarity was measured on {state.stats.sim_checked_pairs} of "
            f"{state.pair_population} structurally-mergeable pairs; the 'estimated' missed-merge "
            f"count extrapolates that pass rate against intra_similarity_threshold="
            f"{th.intra_similarity_threshold}."
        )
    else:
        notes.append(
            "Embedding similarity was not measured, so only the structural classification is "
            "available and the missed-merge count is an upper bound: some of those pairs would "
            "have been rejected by the similarity check anyway. Run with --steps including "
            "'merge' plus --config on a machine with the models available to complete it."
        )
    if state.stats.sim_failed_pairs:
        notes.append(
            f"{state.stats.sim_failed_pairs} sampled pair(s) could not be scored (see the "
            "anomalies below for the cause); they are excluded from the pass rate rather than "
            "counted as failures to merge."
        )
    return {
        "overall": state.stats.to_dict(),
        "notes": notes,
        "anomalies": state.anomalies.to_dict(),
    }


def cleanup(state: MergeState) -> None:
    if state.bucket_root:
        shutil.rmtree(state.bucket_root, ignore_errors=True)
