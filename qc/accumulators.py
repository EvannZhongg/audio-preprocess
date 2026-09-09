"""Mergeable statistic containers passed between worker processes and the main one.

The hard constraint that shapes this file: a production shard holds millions of
segments, and QC fans work out over a process pool. If a worker returned rows,
the main process would have to hold the whole dataset in RAM (and pay to pickle
it). So every worker returns one of these -- a fixed-size bag of counters and
histogram bins whose size depends on the number of *buckets*, not the number of
rows. The main process just calls `merge` repeatedly.

Every container also implements `to_dict()`, which is what lands in the JSON
report, so there is one definition of each statistic rather than a
worker-side and a report-side copy that can drift.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# Coarse duration buckets, matching tmp/stat_parquet.py:30-37 so QC's headline
# distribution is comparable with the numbers already circulated from that script.
COARSE_EDGES = (4.0, 8.0, 15.0)
COARSE_LABELS = ("<4s", "4-8s", "8-15s", ">=15s")

# Fine buckets, matching misc/analyze_output.py:61-63.
FINE_EDGES = (3.0, 5.0, 7.0, 9.0, 12.0, 15.0)
FINE_LABELS = ("0-3s", "3-5s", "5-7s", "7-9s", "9-12s", "12-15s", "15s+")

# Segment-length bands used when reporting model verdicts. Splitting by length
# is not cosmetic: pyannote degrades to "one speaker" on very short audio and
# DNSMOS self-concatenates clips under 9.01s, so a single pooled number would
# hide both artefacts.
LENGTH_BAND_EDGES = (2.0, 4.0, 8.0)
LENGTH_BAND_LABELS = ("<2s", "2-4s", "4-8s", ">=8s")


def bucket_index(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _safe_div(a: float, b: float) -> Optional[float]:
    return (a / b) if b else None


def _pct(a: float, b: float) -> Optional[float]:
    r = _safe_div(a, b)
    return round(r * 100.0, 4) if r is not None else None


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------

@dataclass
class Histogram:
    """Fixed-bin histogram over a known range, plus exact count/sum/min/max.

    Percentiles are estimated from the bins rather than computed exactly,
    because exact quantiles need every value retained -- unacceptable at
    millions of rows. With the default 400 bins the error is bounded by one bin
    width, which is far below the precision anyone reads a P90 at.
    """

    lo: float = 0.0
    hi: float = 60.0
    bins: int = 400
    counts: list[int] = field(default_factory=list)
    n: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    below: int = 0
    above: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * self.bins

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value
        if value < self.lo:
            self.below += 1
            return
        if value >= self.hi:
            self.above += 1
            return
        idx = int((value - self.lo) / (self.hi - self.lo) * self.bins)
        self.counts[min(idx, self.bins - 1)] += 1

    def merge(self, other: "Histogram") -> None:
        if other.n == 0:
            return
        if other.bins != self.bins or other.lo != self.lo or other.hi != self.hi:
            raise ValueError("cannot merge histograms with different binning")
        self.n += other.n
        self.total += other.total
        self.below += other.below
        self.above += other.above
        for i, c in enumerate(other.counts):
            self.counts[i] += c
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        if other.maximum is not None:
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)

    @property
    def mean(self) -> Optional[float]:
        return _safe_div(self.total, self.n)

    def percentile(self, q: float) -> Optional[float]:
        """Bin-interpolated percentile. Out-of-range mass is honoured by
        clamping to lo/hi, so a P99 is never reported inside the range when
        1% of the data sits above `hi`."""
        if self.n == 0:
            return None
        target = q / 100.0 * self.n
        if target <= self.below:
            return self.minimum
        seen = self.below
        width = (self.hi - self.lo) / self.bins
        for i, c in enumerate(self.counts):
            if seen + c >= target and c > 0:
                frac = (target - seen) / c
                return round(self.lo + (i + frac) * width, 4)
            seen += c
        return self.maximum

    def to_dict(self, with_bins: bool = False) -> dict:
        out = {
            "count": self.n,
            "sum": round(self.total, 4),
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "min": round(self.minimum, 4) if self.minimum is not None else None,
            "max": round(self.maximum, 4) if self.maximum is not None else None,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
            "below_range": self.below,
            "above_range": self.above,
        }
        if with_bins:
            out["range"] = [self.lo, self.hi]
            out["bins"] = self.counts
        return out


# ---------------------------------------------------------------------------
# requirement 1: yield funnel
# ---------------------------------------------------------------------------

@dataclass
class RawLevel:
    """Level 0 -- the raw corpus, as described by the manifest."""

    files: int = 0
    duration: float = 0.0
    files_unknown_duration: int = 0
    parts_read: int = 0
    parts_failed: int = 0

    def merge(self, other: "RawLevel") -> None:
        self.files += other.files
        self.duration += other.duration
        self.files_unknown_duration += other.files_unknown_duration
        self.parts_read += other.parts_read
        self.parts_failed += other.parts_failed

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "duration_seconds": round(self.duration, 3),
            "duration_hours": round(self.duration / 3600.0, 4),
            "files_unknown_duration": self.files_unknown_duration,
            "parts_read": self.parts_read,
            "parts_failed": self.parts_failed,
        }


@dataclass
class Stage1Level:
    """Level 1 -- per-segment stage-1 output."""

    rows: int = 0
    valid_segments: int = 0
    valid_duration: float = 0.0
    failed_file_rows: int = 0
    error_types: Counter = field(default_factory=Counter)
    ok_sources: set = field(default_factory=set)
    failed_sources: set = field(default_factory=set)
    chunk_paths: set = field(default_factory=set)
    parts_read: int = 0
    parts_failed: int = 0
    # Bounded: identity sets are dropped once past this size, since the exact
    # file count matters far less than not exhausting memory on a huge shard.
    track_identities: bool = True

    def merge(self, other: "Stage1Level") -> None:
        self.rows += other.rows
        self.valid_segments += other.valid_segments
        self.valid_duration += other.valid_duration
        self.failed_file_rows += other.failed_file_rows
        self.error_types.update(other.error_types)
        self.parts_read += other.parts_read
        self.parts_failed += other.parts_failed
        if self.track_identities and other.track_identities:
            self.ok_sources |= other.ok_sources
            self.failed_sources |= other.failed_sources
            self.chunk_paths |= other.chunk_paths
        else:
            self.track_identities = False
            self.ok_sources = set()
            self.failed_sources = set()
            self.chunk_paths = set()

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "valid_segments": self.valid_segments,
            "valid_duration_seconds": round(self.valid_duration, 3),
            "valid_duration_hours": round(self.valid_duration / 3600.0, 4),
            "failed_file_rows": self.failed_file_rows,
            "files_with_output": len(self.ok_sources) if self.track_identities else None,
            "files_failed": len(self.failed_sources) if self.track_identities else None,
            "chunk_wavs": len(self.chunk_paths) if self.track_identities else None,
            "error_types": dict(self.error_types.most_common(30)),
            "parts_read": self.parts_read,
            "parts_failed": self.parts_failed,
        }


@dataclass
class Stage2Level:
    """Level 2 -- per-segment stage-2 output; `kept_*` is the final yield."""

    rows: int = 0
    total_duration: float = 0.0
    kept_rows: int = 0
    kept_duration: float = 0.0
    error_rows: int = 0
    error_duration: float = 0.0
    retriable_error_rows: int = 0
    drop_rows: Counter = field(default_factory=Counter)
    drop_duration: Counter = field(default_factory=Counter)
    error_types: Counter = field(default_factory=Counter)
    kept_utt_ids: int = 0
    parts_read: int = 0
    parts_failed: int = 0

    def merge(self, other: "Stage2Level") -> None:
        self.rows += other.rows
        self.total_duration += other.total_duration
        self.kept_rows += other.kept_rows
        self.kept_duration += other.kept_duration
        self.error_rows += other.error_rows
        self.error_duration += other.error_duration
        self.retriable_error_rows += other.retriable_error_rows
        self.drop_rows.update(other.drop_rows)
        self.drop_duration.update(other.drop_duration)
        self.error_types.update(other.error_types)
        self.kept_utt_ids += other.kept_utt_ids
        self.parts_read += other.parts_read
        self.parts_failed += other.parts_failed

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "total_duration_seconds": round(self.total_duration, 3),
            "total_duration_hours": round(self.total_duration / 3600.0, 4),
            "kept_rows": self.kept_rows,
            "kept_duration_seconds": round(self.kept_duration, 3),
            "kept_duration_hours": round(self.kept_duration / 3600.0, 4),
            "error_rows": self.error_rows,
            "error_duration_seconds": round(self.error_duration, 3),
            "retriable_error_rows": self.retriable_error_rows,
            "keep_rate_by_rows_pct": _pct(self.kept_rows, self.rows),
            "keep_rate_by_duration_pct": _pct(self.kept_duration, self.total_duration),
            "dropped_by": {
                name: {
                    "rows": self.drop_rows.get(name, 0),
                    "duration_seconds": round(self.drop_duration.get(name, 0.0), 3),
                    "duration_hours": round(self.drop_duration.get(name, 0.0) / 3600.0, 4),
                    "rows_pct_of_total": _pct(self.drop_rows.get(name, 0), self.rows),
                }
                for name in sorted(set(self.drop_rows) | set(self.drop_duration))
            },
            "error_types": dict(self.error_types.most_common(30)),
            "parts_read": self.parts_read,
            "parts_failed": self.parts_failed,
        }


@dataclass
class YieldFunnel:
    """The three levels for one shard (or, merged, for the whole run)."""

    raw: RawLevel = field(default_factory=RawLevel)
    stage1: Stage1Level = field(default_factory=Stage1Level)
    stage2: Stage2Level = field(default_factory=Stage2Level)

    def merge(self, other: "YieldFunnel") -> None:
        self.raw.merge(other.raw)
        self.stage1.merge(other.stage1)
        self.stage2.merge(other.stage2)

    def to_dict(self) -> dict:
        raw_dur = self.raw.duration
        s1_dur = self.stage1.valid_duration
        return {
            "raw": self.raw.to_dict(),
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict(),
            "rates": {
                "stage1_duration_pct_of_raw": _pct(s1_dur, raw_dur),
                "stage1_files_pct_of_raw": (
                    _pct(len(self.stage1.ok_sources), self.raw.files)
                    if self.stage1.track_identities else None
                ),
                "stage2_duration_pct_of_stage1": _pct(self.stage2.kept_duration, s1_dur),
                "stage2_duration_pct_of_raw": _pct(self.stage2.kept_duration, raw_dur),
                "stage2_rows_pct_of_stage1": _pct(self.stage2.kept_rows, self.stage1.valid_segments),
            },
        }


# ---------------------------------------------------------------------------
# requirement 2: duration distribution
# ---------------------------------------------------------------------------

@dataclass
class BucketStats:
    """Counts and summed duration per bucket, for one bucketing scheme."""

    labels: tuple[str, ...]
    counts: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.labels)
        if not self.durations:
            self.durations = [0.0] * len(self.labels)

    def add(self, idx: int, duration: float) -> None:
        self.counts[idx] += 1
        self.durations[idx] += duration

    def merge(self, other: "BucketStats") -> None:
        for i in range(len(self.labels)):
            self.counts[i] += other.counts[i]
            self.durations[i] += other.durations[i]

    def to_dict(self) -> dict:
        total_n = sum(self.counts)
        total_d = sum(self.durations)
        return {
            label: {
                "count": self.counts[i],
                "count_pct": _pct(self.counts[i], total_n),
                "duration_hours": round(self.durations[i] / 3600.0, 4),
                "duration_pct": _pct(self.durations[i], total_d),
            }
            for i, label in enumerate(self.labels)
        }


@dataclass
class DurationStats:
    """Requirement 2: how long the final segments are."""

    coarse: BucketStats = field(default_factory=lambda: BucketStats(COARSE_LABELS))
    fine: BucketStats = field(default_factory=lambda: BucketStats(FINE_LABELS))
    hist: Histogram = field(default_factory=lambda: Histogram(0.0, 60.0, 600))
    by_language: dict = field(default_factory=dict)
    below_min_length: int = 0
    above_max_length: int = 0
    parts_read: int = 0
    parts_failed: int = 0

    def add(self, duration: float, language: Optional[str],
            min_length: float, max_length: float) -> None:
        self.coarse.add(bucket_index(duration, COARSE_EDGES), duration)
        self.fine.add(bucket_index(duration, FINE_EDGES), duration)
        self.hist.add(duration)
        if duration < min_length:
            self.below_min_length += 1
        if duration > max_length:
            self.above_max_length += 1
        lang = language or "unknown"
        entry = self.by_language.get(lang)
        if entry is None:
            entry = {"count": 0, "duration": 0.0}
            self.by_language[lang] = entry
        entry["count"] += 1
        entry["duration"] += duration

    def merge(self, other: "DurationStats") -> None:
        self.coarse.merge(other.coarse)
        self.fine.merge(other.fine)
        self.hist.merge(other.hist)
        self.below_min_length += other.below_min_length
        self.above_max_length += other.above_max_length
        self.parts_read += other.parts_read
        self.parts_failed += other.parts_failed
        for lang, entry in other.by_language.items():
            cur = self.by_language.setdefault(lang, {"count": 0, "duration": 0.0})
            cur["count"] += entry["count"]
            cur["duration"] += entry["duration"]

    def to_dict(self) -> dict:
        total_n = self.hist.n
        total_d = self.hist.total
        return {
            "segments": total_n,
            "total_hours": round(total_d / 3600.0, 4),
            "summary": self.hist.to_dict(),
            "coarse_buckets": self.coarse.to_dict(),
            "fine_buckets": self.fine.to_dict(),
            "by_language": {
                lang: {
                    "count": e["count"],
                    "count_pct": _pct(e["count"], total_n),
                    "duration_hours": round(e["duration"] / 3600.0, 4),
                    "duration_pct": _pct(e["duration"], total_d),
                }
                for lang, e in sorted(
                    self.by_language.items(), key=lambda kv: -kv[1]["count"]
                )
            },
            "outside_production_bounds": {
                "below_min_segment_length": self.below_min_length,
                "above_max_segment_length": self.above_max_length,
            },
            "parts_read": self.parts_read,
            "parts_failed": self.parts_failed,
        }


# ---------------------------------------------------------------------------
# requirement 5: adjacent-pair mergeability
# ---------------------------------------------------------------------------

# Reasons a same-speaker adjacent pair cannot be merged, in the order
# pipeline_v2/steps/segment.py:120-140 checks them. `no_embedding` comes first
# among the blockers because it is structural: a sub-1s segment never gets an
# embedding (embedding_refinement.py:85-90) and therefore can never merge, no
# matter how close or similar it is.
PAIR_CATEGORIES = (
    "different_speaker",
    "no_embedding_short_segment",
    "gap_too_large",
    "merged_too_long",
    "mergeable_by_structure",
)


@dataclass
class MergeabilityStats:
    """Requirement 5: adjacent same-speaker pairs that look mergeable.

    Split in two layers on purpose. The structural conditions (same speaker,
    gap, merged length, has-embedding) are computed over *every* pair, cheaply,
    from parquet. The embedding-similarity condition needs a GPU and is
    therefore measured on a sample; its hit rate is reported separately so the
    structural counts can be extrapolated instead of silently mixing an exact
    count with an estimated one.
    """

    pairs: int = 0
    chunks: int = 0
    category_counts: Counter = field(default_factory=Counter)
    category_duration: Counter = field(default_factory=Counter)
    gap_hist: Histogram = field(default_factory=lambda: Histogram(0.0, 10.0, 200))
    negative_gap_pairs: int = 0
    zero_gap_pairs: int = 0
    single_segment_chunks: int = 0
    # Sampled embedding check, filled by the GPU pass.
    sim_checked_pairs: int = 0
    sim_pass_pairs: int = 0
    sim_hist: Histogram = field(default_factory=lambda: Histogram(-1.0, 1.0, 200))
    sim_failed_pairs: int = 0
    buckets_read: int = 0
    buckets_failed: int = 0

    def add_pair(self, category: str, pair_duration: float, gap: float) -> None:
        self.pairs += 1
        self.category_counts[category] += 1
        self.category_duration[category] += pair_duration
        if gap < 0:
            self.negative_gap_pairs += 1
            gap = 0.0
        elif gap == 0.0:
            self.zero_gap_pairs += 1
        self.gap_hist.add(gap)

    def merge(self, other: "MergeabilityStats") -> None:
        self.pairs += other.pairs
        self.chunks += other.chunks
        self.category_counts.update(other.category_counts)
        self.category_duration.update(other.category_duration)
        self.gap_hist.merge(other.gap_hist)
        self.negative_gap_pairs += other.negative_gap_pairs
        self.zero_gap_pairs += other.zero_gap_pairs
        self.single_segment_chunks += other.single_segment_chunks
        self.sim_checked_pairs += other.sim_checked_pairs
        self.sim_pass_pairs += other.sim_pass_pairs
        self.sim_hist.merge(other.sim_hist)
        self.sim_failed_pairs += other.sim_failed_pairs
        self.buckets_read += other.buckets_read
        self.buckets_failed += other.buckets_failed

    @property
    def structurally_mergeable(self) -> int:
        return self.category_counts.get("mergeable_by_structure", 0)

    def to_dict(self) -> dict:
        sim_rate = _safe_div(self.sim_pass_pairs, self.sim_checked_pairs)
        struct = self.structurally_mergeable
        return {
            "chunks": self.chunks,
            "single_segment_chunks": self.single_segment_chunks,
            "adjacent_pairs": self.pairs,
            "categories": {
                name: {
                    "pairs": self.category_counts.get(name, 0),
                    "pairs_pct": _pct(self.category_counts.get(name, 0), self.pairs),
                    "duration_hours": round(
                        self.category_duration.get(name, 0.0) / 3600.0, 4
                    ),
                }
                for name in PAIR_CATEGORIES
            },
            "gap_seconds": self.gap_hist.to_dict(),
            "negative_gap_pairs": self.negative_gap_pairs,
            "zero_gap_pairs": self.zero_gap_pairs,
            "embedding_check": {
                "checked_pairs": self.sim_checked_pairs,
                "similarity_pass_pairs": self.sim_pass_pairs,
                "similarity_pass_rate_pct": _pct(self.sim_pass_pairs, self.sim_checked_pairs),
                "failed_pairs": self.sim_failed_pairs,
                "similarity": self.sim_hist.to_dict(),
            },
            "suspected_missed_merges": {
                "structurally_mergeable_pairs": struct,
                "structurally_mergeable_pct_of_pairs": _pct(struct, self.pairs),
                "estimated_after_embedding_filter": (
                    int(round(struct * sim_rate)) if sim_rate is not None else None
                ),
                "estimated_pct_of_pairs": (
                    _pct(struct * sim_rate, self.pairs) if sim_rate is not None else None
                ),
            },
            "buckets_read": self.buckets_read,
            "buckets_failed": self.buckets_failed,
        }


# ---------------------------------------------------------------------------
# requirement 3: speaker re-check
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    """2x2 agreement between the embedding method and the diarization method."""

    both_multi: int = 0
    emb_only: int = 0
    dia_only: int = 0
    both_single: int = 0

    def add(self, emb_multi: bool, dia_multi: bool) -> None:
        if emb_multi and dia_multi:
            self.both_multi += 1
        elif emb_multi:
            self.emb_only += 1
        elif dia_multi:
            self.dia_only += 1
        else:
            self.both_single += 1

    def merge(self, other: "ConfusionMatrix") -> None:
        self.both_multi += other.both_multi
        self.emb_only += other.emb_only
        self.dia_only += other.dia_only
        self.both_single += other.both_single

    @property
    def total(self) -> int:
        return self.both_multi + self.emb_only + self.dia_only + self.both_single

    def to_dict(self) -> dict:
        agree = self.both_multi + self.both_single
        return {
            "compared": self.total,
            "both_multi_speaker": self.both_multi,
            "embedding_only_multi": self.emb_only,
            "diarization_only_multi": self.dia_only,
            "both_single_speaker": self.both_single,
            "agreement_pct": _pct(agree, self.total),
            "embedding_multi_pct": _pct(self.both_multi + self.emb_only, self.total),
            "diarization_multi_pct": _pct(self.both_multi + self.dia_only, self.total),
        }


@dataclass
class SpeakerRecheckStats:
    """Requirement 3: is each final segment really single-speaker?"""

    segments: int = 0
    emb_scored: int = 0
    emb_multi: int = 0
    emb_failed: int = 0
    dia_scored: int = 0
    dia_multi: int = 0
    dia_failed: int = 0
    dia_speaker_counts: Counter = field(default_factory=Counter)
    matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    matrix_by_band: dict = field(default_factory=dict)
    band_totals: Counter = field(default_factory=Counter)
    sim_hist: Histogram = field(default_factory=lambda: Histogram(-1.0, 1.0, 200))
    drift_hist: Histogram = field(default_factory=lambda: Histogram(-1.0, 1.0, 200))
    drift_compared: int = 0
    multi_duration: float = 0.0
    total_duration: float = 0.0

    def band_matrix(self, band: str) -> ConfusionMatrix:
        m = self.matrix_by_band.get(band)
        if m is None:
            m = ConfusionMatrix()
            self.matrix_by_band[band] = m
        return m

    def merge(self, other: "SpeakerRecheckStats") -> None:
        self.segments += other.segments
        self.emb_scored += other.emb_scored
        self.emb_multi += other.emb_multi
        self.emb_failed += other.emb_failed
        self.dia_scored += other.dia_scored
        self.dia_multi += other.dia_multi
        self.dia_failed += other.dia_failed
        self.dia_speaker_counts.update(other.dia_speaker_counts)
        self.matrix.merge(other.matrix)
        self.band_totals.update(other.band_totals)
        self.sim_hist.merge(other.sim_hist)
        self.drift_hist.merge(other.drift_hist)
        self.drift_compared += other.drift_compared
        self.multi_duration += other.multi_duration
        self.total_duration += other.total_duration
        for band, m in other.matrix_by_band.items():
            self.band_matrix(band).merge(m)

    def to_dict(self) -> dict:
        return {
            "segments_checked": self.segments,
            "total_hours": round(self.total_duration / 3600.0, 4),
            "embedding_method": {
                "scored": self.emb_scored,
                "multi_speaker": self.emb_multi,
                "multi_speaker_pct": _pct(self.emb_multi, self.emb_scored),
                "failed": self.emb_failed,
                "min_similarity": self.sim_hist.to_dict(),
            },
            "diarization_method": {
                "scored": self.dia_scored,
                "multi_speaker": self.dia_multi,
                "multi_speaker_pct": _pct(self.dia_multi, self.dia_scored),
                "failed": self.dia_failed,
                "speaker_count_distribution": {
                    str(k): v for k, v in sorted(self.dia_speaker_counts.items())
                },
            },
            "cross_check": self.matrix.to_dict(),
            "cross_check_by_length": {
                band: {
                    **self.band_matrix(band).to_dict(),
                    "segments": self.band_totals.get(band, 0),
                }
                for band in LENGTH_BAND_LABELS
                if self.band_totals.get(band, 0)
            },
            "recorded_vs_recheck_drift": {
                "compared": self.drift_compared,
                "delta_min_similarity": self.drift_hist.to_dict(),
            },
            "multi_speaker_duration_hours": round(self.multi_duration / 3600.0, 4),
            "multi_speaker_duration_pct": _pct(self.multi_duration, self.total_duration),
        }


# ---------------------------------------------------------------------------
# requirement 4: background-noise re-check
# ---------------------------------------------------------------------------

BAK_GRADES = ("clean", "mild_residual", "clear_residual")


@dataclass
class BackgroundRecheckStats:
    """Requirement 4: was the background actually removed?"""

    segments: int = 0
    graded: int = 0
    grade_counts: Counter = field(default_factory=Counter)
    grade_duration: Counter = field(default_factory=Counter)
    grade_by_band: dict = field(default_factory=dict)
    bak_hist: Histogram = field(default_factory=lambda: Histogram(1.0, 5.0, 160))
    sig_hist: Histogram = field(default_factory=lambda: Histogram(1.0, 5.0, 160))
    ovrl_hist: Histogram = field(default_factory=lambda: Histogram(1.0, 5.0, 160))
    snr_hist: Histogram = field(default_factory=lambda: Histogram(-10.0, 80.0, 180))
    c50_hist: Histogram = field(default_factory=lambda: Histogram(0.0, 80.0, 160))
    dnsmos_failed: int = 0
    brouhaha_failed: int = 0
    brouhaha_sentinel: int = 0
    dnsmos_drift: Histogram = field(default_factory=lambda: Histogram(-2.0, 2.0, 160))
    snr_drift: Histogram = field(default_factory=lambda: Histogram(-40.0, 40.0, 160))
    c50_drift: Histogram = field(default_factory=lambda: Histogram(-40.0, 40.0, 160))
    dnsmos_drift_compared: int = 0
    snr_drift_compared: int = 0
    c50_drift_compared: int = 0
    below_snr_threshold: int = 0
    below_c50_threshold: int = 0
    below_dnsmos_threshold: int = 0

    def band_grades(self, band: str) -> Counter:
        c = self.grade_by_band.get(band)
        if c is None:
            c = Counter()
            self.grade_by_band[band] = c
        return c

    def merge(self, other: "BackgroundRecheckStats") -> None:
        self.segments += other.segments
        self.graded += other.graded
        self.grade_counts.update(other.grade_counts)
        self.grade_duration.update(other.grade_duration)
        for band, counter in other.grade_by_band.items():
            self.band_grades(band).update(counter)
        for name in ("bak_hist", "sig_hist", "ovrl_hist", "snr_hist", "c50_hist",
                     "dnsmos_drift", "snr_drift", "c50_drift"):
            getattr(self, name).merge(getattr(other, name))
        self.dnsmos_failed += other.dnsmos_failed
        self.brouhaha_failed += other.brouhaha_failed
        self.brouhaha_sentinel += other.brouhaha_sentinel
        self.dnsmos_drift_compared += other.dnsmos_drift_compared
        self.snr_drift_compared += other.snr_drift_compared
        self.c50_drift_compared += other.c50_drift_compared
        self.below_snr_threshold += other.below_snr_threshold
        self.below_c50_threshold += other.below_c50_threshold
        self.below_dnsmos_threshold += other.below_dnsmos_threshold

    def to_dict(self) -> dict:
        total_grade_dur = sum(self.grade_duration.values())
        return {
            "segments_checked": self.segments,
            "graded": self.graded,
            "grades": {
                grade: {
                    "count": self.grade_counts.get(grade, 0),
                    "count_pct": _pct(self.grade_counts.get(grade, 0), self.graded),
                    "duration_hours": round(self.grade_duration.get(grade, 0.0) / 3600.0, 4),
                    "duration_pct": _pct(self.grade_duration.get(grade, 0.0), total_grade_dur),
                }
                for grade in BAK_GRADES
            },
            "grades_by_length": {
                band: {
                    "segments": sum(counter.values()),
                    **{
                        grade: {
                            "count": counter.get(grade, 0),
                            "count_pct": _pct(counter.get(grade, 0), sum(counter.values())),
                        }
                        for grade in BAK_GRADES
                    },
                }
                for band, counter in (
                    (b, self.grade_by_band[b])
                    for b in LENGTH_BAND_LABELS if b in self.grade_by_band
                )
            },
            "metrics": {
                "dnsmos_bak": self.bak_hist.to_dict(),
                "dnsmos_sig": self.sig_hist.to_dict(),
                "dnsmos_ovrl": self.ovrl_hist.to_dict(),
                "brouhaha_snr": self.snr_hist.to_dict(),
                "brouhaha_c50": self.c50_hist.to_dict(),
            },
            "scoring_failures": {
                "dnsmos_failed": self.dnsmos_failed,
                "brouhaha_failed": self.brouhaha_failed,
                "brouhaha_sentinel_values": self.brouhaha_sentinel,
            },
            "below_production_thresholds": {
                "dnsmos_ovrl": self.below_dnsmos_threshold,
                "brouhaha_snr": self.below_snr_threshold,
                "brouhaha_c50": self.below_c50_threshold,
            },
            "recorded_vs_recheck_drift": {
                "dnsmos": {
                    "compared": self.dnsmos_drift_compared,
                    **self.dnsmos_drift.to_dict(),
                },
                "snr": {
                    "compared": self.snr_drift_compared,
                    **self.snr_drift.to_dict(),
                },
                "c50": {
                    "compared": self.c50_drift_compared,
                    **self.c50_drift.to_dict(),
                },
            },
        }


@dataclass
class AnomalyCounter:
    """Things that were skipped, and why. A quiet skip is a silent lie."""

    counts: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)
    max_examples: int = 3

    def add(self, kind: str, detail: Optional[str] = None) -> None:
        self.counts[kind] += 1
        if detail:
            bucket = self.examples.setdefault(kind, [])
            if len(bucket) < self.max_examples:
                bucket.append(detail[:200])

    def merge(self, other: "AnomalyCounter") -> None:
        self.counts.update(other.counts)
        for kind, items in other.examples.items():
            bucket = self.examples.setdefault(kind, [])
            for item in items:
                if len(bucket) >= self.max_examples:
                    break
                bucket.append(item)

    def to_dict(self) -> dict:
        return {
            "counts": dict(self.counts.most_common()),
            "examples": self.examples,
        }
