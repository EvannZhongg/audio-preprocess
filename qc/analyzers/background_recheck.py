"""Requirement 4: was the background actually removed?

Stage 1 runs a separation/denoise step before segmenting
(pipeline_v2/pipeline.py:103-112), so the exported wavs are *supposed* to be
clean speech. This re-measures what actually came out, using two complementary
signals:

  * **DNSMOS BAK** -- the background component of the DNSMOS triplet, i.e. a
    direct subjective estimate of "how audible is the noise". This is the
    headline metric and it is new to QC: production calls the very same model but
    keeps only `OVRL` and discards SIG/BAK
    (pipeline_v2/steps/metrics.py:81-83). An overall score can stay respectable
    while the background is clearly audible, which is exactly the failure this
    section exists to surface.
  * **brouhaha SNR / C50** -- objective signal-to-noise and reverberation
    estimates, the same numbers stage 1 filters on when `use_brouhaha` is set,
    so residual noise can also be judged against the production bar.

Both are reported against the values already stored in the parquet, because the
interesting question is usually not the absolute score but whether it changed --
a large gap between what the pipeline recorded and what the exported audio
measures points at the export path, not the audio.

Grades are cut by segment length too, since DNSMOS pads clips shorter than 9.01s
by concatenating them with themselves (models/dnsmos.py:157-159); that is a real
systematic bias on short segments and it must not be hidden in a pooled average.
"""
from __future__ import annotations

from qc.accumulators import (BAK_GRADES, LENGTH_BAND_EDGES, LENGTH_BAND_LABELS,
                             AnomalyCounter, BackgroundRecheckStats, bucket_index)
from qc.config import DNSMOS_INPUT_LENGTH_S, QCConfig
from qc.sampling import SampleInfo


def _grade(bak: float, bak_pass: float, bak_warn: float) -> str:
    if bak >= bak_pass:
        return "clean"
    if bak >= bak_warn:
        return "mild_residual"
    return "clear_residual"


def reduce(cfg: QCConfig, sample_info: SampleInfo,
           anomalies: AnomalyCounter) -> dict:
    """Aggregate the cached background metrics into the report section."""
    from qc.cache import iter_records

    th = cfg.thresholds
    stats = BackgroundRecheckStats()

    for rec in iter_records(cfg.work_dir, kind="segment"):
        # Speaker-only records (a --steps speaker run) have no metrics at all.
        if not any(k in rec for k in ("bak", "sig", "ovrl", "c50", "snr")):
            continue
        stats.segments += 1
        duration = float(rec.get("seg_duration") or 0.0)
        band = LENGTH_BAND_LABELS[bucket_index(duration, LENGTH_BAND_EDGES)]

        if rec.get("dnsmos_error"):
            stats.dnsmos_failed += 1
        if rec.get("brouhaha_sentinel"):
            stats.brouhaha_sentinel += 1
        elif rec.get("brouhaha_error"):
            stats.brouhaha_failed += 1

        bak = rec.get("bak")
        if bak is not None:
            stats.bak_hist.add(float(bak))
            grade = _grade(float(bak), cfg.bak_pass, cfg.bak_warn)
            stats.graded += 1
            stats.grade_counts[grade] += 1
            stats.grade_duration[grade] += duration
            stats.band_grades(band)[grade] += 1

        sig = rec.get("sig")
        if sig is not None:
            stats.sig_hist.add(float(sig))
        ovrl = rec.get("ovrl")
        if ovrl is not None:
            stats.ovrl_hist.add(float(ovrl))
            if float(ovrl) < th.fixed_dnsmos_threshold:
                stats.below_dnsmos_threshold += 1
            recorded = rec.get("rec_dnsmos")
            if recorded is not None:
                stats.dnsmos_drift_compared += 1
                stats.dnsmos_drift.add(float(ovrl) - float(recorded))

        snr = rec.get("snr")
        if snr is not None:
            stats.snr_hist.add(float(snr))
            if float(snr) < th.fixed_snr_threshold:
                stats.below_snr_threshold += 1
            recorded = rec.get("rec_snr")
            if recorded is not None:
                stats.snr_drift_compared += 1
                stats.snr_drift.add(float(snr) - float(recorded))

        c50 = rec.get("c50")
        if c50 is not None:
            stats.c50_hist.add(float(c50))
            if float(c50) < th.fixed_c50_threshold:
                stats.below_c50_threshold += 1
            recorded = rec.get("rec_c50")
            if recorded is not None:
                stats.c50_drift_compared += 1
                stats.c50_drift.add(float(c50) - float(recorded))

    payload = stats.to_dict()
    payload["sampling"] = sample_info.to_dict()
    payload["thresholds"] = {
        "bak_pass": cfg.bak_pass,
        "bak_warn": cfg.bak_warn,
        "production_fixed_dnsmos_threshold": th.fixed_dnsmos_threshold,
        "production_fixed_snr_threshold": th.fixed_snr_threshold,
        "production_fixed_c50_threshold": th.fixed_c50_threshold,
        "production_metrics_strategy": th.metrics_strategy,
        "production_use_brouhaha": th.use_brouhaha,
    }
    payload["notes"] = _notes(cfg, stats)
    payload["anomalies"] = anomalies.to_dict()
    return payload


def _notes(cfg: QCConfig, stats: BackgroundRecheckStats) -> list[str]:
    th = cfg.thresholds
    notes = [
        f"Grades use DNSMOS BAK: clean >= {cfg.bak_pass}, mild_residual "
        f"[{cfg.bak_warn}, {cfg.bak_pass}), clear_residual < {cfg.bak_warn}. "
        "These cutoffs are QC's own (tunable via --bak-pass/--bak-warn); production has no "
        "BAK threshold because it only keeps DNSMOS OVRL "
        "(pipeline_v2/steps/metrics.py:81-83).",
        "BAK is the background component of DNSMOS, so it answers 'is noise audible' "
        "directly, whereas OVRL can stay acceptable despite obvious background.",
    ]
    if not th.use_brouhaha:
        notes.append(
            "The production config has use_brouhaha=false, which means stage 1 stored "
            f"placeholder c50/snr (the fixed thresholds {th.fixed_c50_threshold}/"
            f"{th.fixed_snr_threshold}) rather than measured values. QC's brouhaha numbers "
            "are therefore real measurements with nothing meaningful to compare against, "
            "and the c50/snr drift figures should be ignored for this run."
        )
    if stats.brouhaha_sentinel:
        notes.append(
            f"{stats.brouhaha_sentinel} segment(s) hit brouhaha's internal failure sentinel "
            "(-420.69, models/brouhaha_metrics.py:39-41). They are excluded from all SNR/C50 "
            "statistics -- including them would drag every average into nonsense."
        )
    if stats.dnsmos_failed:
        notes.append(f"{stats.dnsmos_failed} segment(s) could not be scored by DNSMOS.")
    short = stats.grade_by_band.get("<2s")
    if short and sum(short.values()):
        notes.append(
            f"The <2s band ({sum(short.values())} segments) carries a known bias: DNSMOS "
            f"requires {DNSMOS_INPUT_LENGTH_S}s of audio and self-concatenates shorter clips "
            "(models/dnsmos.py:157-159), so their BAK reflects a looped signal. Read the "
            "longer bands for the real picture."
        )
    if stats.dnsmos_drift_compared:
        mean = stats.dnsmos_drift.mean
        if mean is not None:
            notes.append(
                f"DNSMOS OVRL drift vs. the recorded value: mean {mean:.4f} over "
                f"{stats.dnsmos_drift_compared} segment(s). Production scores the "
                "pre-export waveform while QC scores the exported wav, so a small offset is "
                "expected; a large one points at the export path."
            )
    if stats.graded:
        clear = stats.grade_counts.get("clear_residual", 0)
        if clear:
            pct = clear / stats.graded * 100.0
            notes.append(
                f"{clear} segment(s) ({pct:.2f}%) show clear residual background. If that "
                "share is high, the separation step "
                "(pipeline_v2/steps/source_separation.py) is the place to look."
            )
    return notes
