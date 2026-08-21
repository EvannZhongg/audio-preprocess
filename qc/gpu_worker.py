"""The GPU worker: model-bearing subprocess for requirements 3, 4 and 5.

Work is dispatched as **chunk groups**, never as individual segments, and that
is the single most important thing about this module. A chunk wav holds dozens of
segments; decoding it costs far more than scoring one segment. Grouping means
each file is read and decoded exactly once, and every segment inside it is then
sliced from memory -- so a 40-segment chunk costs one decode instead of forty.

For the same reason all three re-checks share one visit to a segment: the
speaker-consistency embedding, the diarization count and the background metrics
are computed back-to-back off the same slice. Running them as three separate
passes would triple the decode cost to compute the same numbers.

Errors are contained at the segment level. One unreadable file or one segment
that makes onnxruntime throw records an error on that item and moves on -- the
same per-segment tolerance production's metrics step uses
(pipeline_v2/steps/metrics.py:60-98), for the same reason: one pathological
segment must not discard the rest of the batch.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SegmentTask:
    """One final-output segment to re-check."""

    utt_id: str
    shard: str
    chunk_audio_path: str          # relative to output_root; vetted before use
    start: float                   # seconds, within the chunk
    end: float
    seg_duration: float
    speaker_id: Optional[str] = None
    # Values production already recorded, kept so the report can show drift
    # between what the pipeline measured and what QC measures now.
    rec_min_similarity: Optional[float] = None
    rec_dnsmos: Optional[float] = None
    rec_c50: Optional[float] = None
    rec_snr: Optional[float] = None


@dataclass(frozen=True)
class PairTask:
    """One adjacent same-speaker pair whose embedding similarity decides
    whether production would really have merged it (requirement 5)."""

    pair_key: str
    shard: str
    chunk_audio_path: str
    prev_start: float
    prev_end: float
    cur_start: float
    cur_end: float


@dataclass
class ChunkGroup:
    """All work for one chunk wav -- the unit of dispatch."""

    chunk_audio_path: str
    shard: str
    segments: list = field(default_factory=list)
    pairs: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.segments) + len(self.pairs)


# ---------------------------------------------------------------------------
# worker state (one per process)
# ---------------------------------------------------------------------------

_STATE: dict = {}


def init_worker(cfg, worker_id: int, device: str,
                need_speaker: bool, need_background: bool,
                need_embedding: bool) -> None:
    """Load the models once for this process.

    `CUDA_VISIBLE_DEVICES` is pinned in addition to passing the torch device
    string because some of the models reach for the GPU outside torch --
    onnxruntime's CUDA provider and pyannote's internal device handling among
    them. Setting both is what actually keeps two workers off the same card.
    """
    import sys

    # Workers are spawned from the repo root by the parent, but a spawn-based
    # start method loses that; make the package importable regardless.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        # After masking, the only visible card is index 0.
        torch_device = "cuda:0"
    elif device == "cuda":
        torch_device = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch_device = "cpu"

    from qc.cache import VerdictWriter, worker_cache_path
    from qc.models_bundle import ModelBundle

    bundle = ModelBundle(
        cfg, torch_device,
        need_speaker=need_speaker,
        need_background=need_background,
        need_embedding=need_embedding,
    )
    _STATE.update({
        "cfg": cfg,
        "worker_id": worker_id,
        "device": torch_device,
        "bundle": bundle,
        "writer": VerdictWriter(worker_cache_path(cfg.work_dir, worker_id)),
        "need_speaker": need_speaker,
        "need_background": need_background,
        "need_embedding": need_embedding,
        "reported_errors": {},
    })
    if bundle.load_errors:
        _log_once("model_load_failed", "; ".join(bundle.load_errors))
    if bundle.constant_warnings:
        _log_once("production_constants_changed", "; ".join(bundle.constant_warnings))


def _log_once(kind: str, detail: str, limit: int = 3) -> None:
    """Rate-limited logging.

    Per-segment logging at this volume would write millions of lines into
    logs/app.log and slow the pass down measurably, so each error kind is logged
    a few times and then only counted.
    """
    import logger

    seen = _STATE.setdefault("reported_errors", {})
    n = seen.get(kind, 0) + 1
    seen[kind] = n
    if n <= limit:
        logger.warning(f"qc_worker_{kind} worker {_STATE.get('worker_id')} {detail}")
    elif n == limit + 1:
        logger.warning(
            f"qc_worker_{kind} worker {_STATE.get('worker_id')} "
            "further occurrences suppressed; see the run summary for totals"
        )


def worker_summary() -> dict:
    """What this worker wants the parent to know once it is done."""
    bundle = _STATE.get("bundle")
    return {
        "worker_id": _STATE.get("worker_id"),
        "device": _STATE.get("device"),
        "error_counts": dict(_STATE.get("reported_errors", {})),
        "load_errors": list(bundle.load_errors) if bundle else [],
        "constant_warnings": list(bundle.constant_warnings) if bundle else [],
    }


def close_worker() -> dict:
    summary = worker_summary()
    writer = _STATE.get("writer")
    if writer is not None:
        writer.close()
    return summary


# ---------------------------------------------------------------------------
# the actual work
# ---------------------------------------------------------------------------

def process_group(group: ChunkGroup) -> dict:
    """Score every segment and pair of one chunk, from a single decode."""
    from qc.layout import resolve_chunk_audio
    from qc.loaders import read_chunk_audio, slice_segment

    cfg = _STATE["cfg"]
    bundle = _STATE["bundle"]
    writer = _STATE["writer"]
    stats = {"segments": 0, "pairs": 0, "skipped": 0, "errors": {}}

    abs_path = resolve_chunk_audio(cfg.output_root, group.chunk_audio_path)
    if abs_path is None:
        # Either the wav is gone or the parquet held a path pointing outside the
        # output tree. Both are anomalies worth counting, neither is fatal.
        stats["skipped"] = group.size
        stats["errors"]["chunk_unresolvable"] = 1
        _log_once("chunk_unresolvable", group.chunk_audio_path)
        for task in group.segments:
            writer.write({"id": task.utt_id, "kind": "segment", "shard": task.shard,
                          "seg_duration": task.seg_duration,
                          "error": "chunk audio unresolvable"})
        for pair in group.pairs:
            writer.write({"id": pair.pair_key, "kind": "pair", "shard": pair.shard,
                          "error": "chunk audio unresolvable"})
        return stats

    try:
        waveform, sample_rate = read_chunk_audio(abs_path)
    except Exception as exc:  # noqa: BLE001
        stats["skipped"] = group.size
        stats["errors"]["chunk_decode_failed"] = 1
        _log_once("chunk_decode_failed", f"{group.chunk_audio_path}: {exc}")
        for task in group.segments:
            writer.write({"id": task.utt_id, "kind": "segment", "shard": task.shard,
                          "seg_duration": task.seg_duration,
                          "error": f"decode failed: {type(exc).__name__}"})
        for pair in group.pairs:
            writer.write({"id": pair.pair_key, "kind": "pair", "shard": pair.shard,
                          "error": f"decode failed: {type(exc).__name__}"})
        return stats

    need_speaker = _STATE["need_speaker"]
    need_background = _STATE["need_background"]

    for task in group.segments:
        record = {
            "id": task.utt_id,
            "kind": "segment",
            "shard": task.shard,
            "seg_duration": task.seg_duration,
            "speaker_id": task.speaker_id,
            "rec_min_similarity": task.rec_min_similarity,
            "rec_dnsmos": task.rec_dnsmos,
            "rec_c50": task.rec_c50,
            "rec_snr": task.rec_snr,
        }
        seg_wave = slice_segment(waveform, sample_rate, task.start, task.end)
        if seg_wave.size == 0:
            record["error"] = "empty slice"
            stats["errors"]["empty_slice"] = stats["errors"].get("empty_slice", 0) + 1
            writer.write(record)
            stats["segments"] += 1
            continue

        if need_speaker:
            min_sim, n_win, emb_err = bundle.embed_consistency(seg_wave, sample_rate)
            record["emb_min_similarity"] = min_sim
            record["emb_n_windows"] = n_win
            if emb_err:
                record["emb_error"] = emb_err
                _log_once("embed_failed", emb_err)

            n_spk, n_dia_seg, dia_err = bundle.diarize_count(seg_wave, sample_rate)
            record["dia_n_speakers"] = n_spk
            record["dia_n_segments"] = n_dia_seg
            if dia_err:
                record["dia_error"] = dia_err
                _log_once("diarize_failed", dia_err)

        if need_background:
            quality = bundle.audio_quality(seg_wave, sample_rate)
            record.update({
                "bak": quality["bak"], "sig": quality["sig"], "ovrl": quality["ovrl"],
                "c50": quality["c50"], "snr": quality["snr"],
                "brouhaha_sentinel": quality["brouhaha_sentinel"],
            })
            if quality["dnsmos_error"]:
                record["dnsmos_error"] = quality["dnsmos_error"]
                _log_once("dnsmos_failed", quality["dnsmos_error"])
            if quality["brouhaha_error"]:
                record["brouhaha_error"] = quality["brouhaha_error"]
                _log_once("brouhaha_failed", quality["brouhaha_error"])

        writer.write(record)
        stats["segments"] += 1

    for pair in group.pairs:
        record = {"id": pair.pair_key, "kind": "pair", "shard": pair.shard}
        prev_wave = slice_segment(waveform, sample_rate, pair.prev_start, pair.prev_end)
        cur_wave = slice_segment(waveform, sample_rate, pair.cur_start, pair.cur_end)
        if prev_wave.size == 0 or cur_wave.size == 0:
            record["error"] = "empty slice"
            writer.write(record)
            stats["pairs"] += 1
            continue
        emb_a, err_a = bundle.embed_reference(prev_wave, sample_rate)
        emb_b, err_b = bundle.embed_reference(cur_wave, sample_rate)
        if err_a or err_b:
            record["error"] = err_a or err_b
            _log_once("pair_embed_failed", record["error"])
        else:
            from qc.models_bundle import cosine_similarity_1d

            record["similarity"] = cosine_similarity_1d(emb_a, emb_b)
        writer.write(record)
        stats["pairs"] += 1

    return stats


def run_groups(groups: list) -> dict:
    """Consume a batch of chunk groups in this worker."""
    totals = {"segments": 0, "pairs": 0, "skipped": 0, "errors": {}}
    for group in groups:
        stats = process_group(group)
        totals["segments"] += stats["segments"]
        totals["pairs"] += stats["pairs"]
        totals["skipped"] += stats["skipped"]
        for kind, n in stats["errors"].items():
            totals["errors"][kind] = totals["errors"].get(kind, 0) + n
    return totals
