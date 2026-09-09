"""Requirement 3: does each final segment really contain one speaker?

Two independent methods, both applied to every sampled segment:

  * **embedding consistency** -- production's own test, re-run: slide 1.1s
    windows across the segment, embed each, and take the minimum cosine
    similarity against the segment's overall embedding. Below
    `inter_similarity_threshold` means some window does not sound like the rest,
    which is how the pipeline infers a second speaker
    (pipeline_v2/steps/embedding_refinement.py:196-205). Cheap, and by
    construction the same standard the data was filtered with.
  * **diarization** -- pyannote run on the isolated segment, counting speakers
    directly. Slower and genuinely independent, since it re-derives the answer
    from scratch rather than reusing stage 1's labels.

Reporting both, plus their 2x2 agreement, is the point: the embedding method's
false-negative rate is unknowable on its own, and diarization alone is too slow
to run at scale. Where they disagree is where the data needs a human.

Everything is broken out by segment length as well, because both methods have
length-dependent failure modes -- pyannote collapses to "one speaker" on very
short audio, and a segment under 1s gets no embedding at all. A single pooled
percentage would average those artefacts into the headline number.
"""
from __future__ import annotations

from typing import Optional

from qc.accumulators import (LENGTH_BAND_EDGES, LENGTH_BAND_LABELS, AnomalyCounter,
                             SpeakerRecheckStats, bucket_index)
from qc.config import QCConfig
from qc.sampling import SampleInfo


def reduce(cfg: QCConfig, sample_info: SampleInfo,
           anomalies: AnomalyCounter) -> dict:
    """Aggregate the cached per-segment verdicts into the report section."""
    from qc.cache import iter_records

    th = cfg.thresholds
    stats = SpeakerRecheckStats()

    for rec in iter_records(cfg.work_dir, kind="segment"):
        # Records with only background metrics (a --steps background run) carry
        # no speaker fields; skip them rather than counting them as failures.
        if "emb_min_similarity" not in rec and "dia_n_speakers" not in rec:
            continue
        stats.segments += 1
        duration = float(rec.get("seg_duration") or 0.0)
        stats.total_duration += duration
        band = LENGTH_BAND_LABELS[bucket_index(duration, LENGTH_BAND_EDGES)]
        stats.band_totals[band] += 1

        min_sim = rec.get("emb_min_similarity")
        emb_multi: Optional[bool] = None
        if min_sim is None:
            # Either scoring failed, or the segment is too short for production
            # to have embedded it either -- distinguished by the error field.
            if rec.get("emb_error") or rec.get("error"):
                stats.emb_failed += 1
        else:
            stats.emb_scored += 1
            stats.sim_hist.add(float(min_sim))
            emb_multi = float(min_sim) < th.inter_similarity_threshold
            if emb_multi:
                stats.emb_multi += 1
            recorded = rec.get("rec_min_similarity")
            if recorded is not None:
                stats.drift_compared += 1
                stats.drift_hist.add(float(min_sim) - float(recorded))

        n_spk = rec.get("dia_n_speakers")
        dia_multi: Optional[bool] = None
        if n_spk is None:
            if rec.get("dia_error") or rec.get("error"):
                stats.dia_failed += 1
        else:
            stats.dia_scored += 1
            stats.dia_speaker_counts[int(n_spk)] += 1
            dia_multi = int(n_spk) > 1
            if dia_multi:
                stats.dia_multi += 1

        if emb_multi is not None and dia_multi is not None:
            stats.matrix.add(emb_multi, dia_multi)
            stats.band_matrix(band).add(emb_multi, dia_multi)
        if emb_multi or dia_multi:
            stats.multi_duration += duration

    payload = stats.to_dict()
    payload["sampling"] = sample_info.to_dict()
    payload["thresholds"] = {
        "inter_similarity_threshold": th.inter_similarity_threshold,
        "diarization_reliable_above_seconds": cfg.dia_min_reliable_s,
    }
    payload["notes"] = _notes(cfg, stats)
    payload["anomalies"] = anomalies.to_dict()
    return payload


def _notes(cfg: QCConfig, stats: SpeakerRecheckStats) -> list[str]:
    th = cfg.thresholds
    notes = [
        "Two independent methods: embedding self-consistency (production's own test, "
        f"multi-speaker when min cosine < inter_similarity_threshold="
        f"{th.inter_similarity_threshold}) and pyannote diarization on the isolated "
        "segment (multi-speaker when it finds >1 speaker).",
        "'multi_speaker' here means the re-check disagrees with the pipeline's "
        "single-speaker assumption -- these segments are the ones worth listening to.",
    ]
    if stats.segments and cfg.sample_n > 0:
        notes.append(
            "Percentages are measured on a deterministic sample, so they carry sampling "
            "error; the same sample is reproduced on every run, and raising --sample-n "
            "reuses the cached verdicts instead of discarding them."
        )
    short_band = stats.matrix_by_band.get("<2s")
    if short_band is not None and short_band.total:
        notes.append(
            f"The <2s band ({short_band.total} compared segments) is NOT reliable for the "
            "diarization method: pyannote tends to report a single speaker on very short "
            "audio regardless of content. Treat the longer bands as the real signal."
        )
    if stats.emb_scored < stats.segments:
        notes.append(
            f"{stats.segments - stats.emb_scored} sampled segment(s) could not be scored by "
            "the embedding method. Segments under 1s are expected here: production does not "
            "embed them either (pipeline_v2/steps/embedding_refinement.py:85-90)."
        )
    if stats.dia_failed:
        notes.append(f"{stats.dia_failed} segment(s) failed diarization outright.")
    if stats.drift_compared:
        mean_drift = stats.drift_hist.mean
        if mean_drift is None:
            notes.append(
                "Drift vs. the recorded speaker_min_similarity was computed on "
                f"{stats.drift_compared} segment(s)."
            )
        else:
            notes.append(
                "Drift vs. the recorded speaker_min_similarity was computed on "
                f"{stats.drift_compared} segment(s); mean delta {mean_drift:.4f}."
            )
        notes.append(
            "A large drift does not automatically mean a bug: production computes its "
            "value on the pre-export waveform, while QC reads the exported wav, so small "
            "differences are expected."
        )
    matrix = stats.matrix
    if matrix.total and (matrix.emb_only or matrix.dia_only):
        notes.append(
            f"The methods disagree on {matrix.emb_only + matrix.dia_only} of "
            f"{matrix.total} segment(s) ({matrix.emb_only} embedding-only, "
            f"{matrix.dia_only} diarization-only). The diarization-only count is the more "
            "interesting one: those are segments production's own test passed but an "
            "independent model considers multi-speaker."
        )
    return notes
