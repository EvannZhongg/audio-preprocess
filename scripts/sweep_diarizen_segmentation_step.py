#!/usr/bin/env python3
"""Sweep DiariZen `segmentation_step` and report speed against quality.

Runs the resident worker once per step over one or more audio files, then
scores every step against `--baseline-step` (treated as ground truth) on the
metric that actually matters here: are the short second-speaker turns -- the
backchannels this model was chosen to catch -- still being detected?

Quality axis is deliberately the same notion `local_adapter_v2/analysis_outputs/
analyze_backchannel.py` uses (short turns by a non-dominant speaker), computed
directly off the diarization JSON so no ASR is needed.

Usage (GPU box):

    python scripts/sweep_diarizen_segmentation_step.py \
        --config configs/config_pipeline_v2_diarizen_tts_clean_v2.json \
        --audio "local_adapter_v2/test_audios/411338_日谈物语/*.mp3" \
        --steps 0.1,0.2,0.25,0.3,0.5 \
        --output sweep_results.json

Read-only with respect to the pipeline: it drives `Diarizer` directly and
writes only its own report.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHORT_MAX_S = 1.5      # matches analyze_backchannel.py's SHORT_MAX
COVERAGE_MIN = 0.5     # a turn counts as "still detected" at >=50% coverage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--audio", required=True,
        help="audio file, directory, or glob (quote it to avoid shell expansion)",
    )
    p.add_argument("--steps", default="0.1,0.2,0.25,0.3,0.5")
    p.add_argument("--baseline-step", type=float, default=0.1)
    p.add_argument("--output", default="sweep_results.json")
    p.add_argument(
        "--max-seconds", type=float, default=None,
        help="truncate each file, for a quick smoke run",
    )
    return p.parse_args()


def collect_audio(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            q for q in path.iterdir()
            if q.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
        )
    return sorted(Path(q) for q in glob.glob(pattern))


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def short_turns(segments: list[dict]) -> list[dict]:
    """Short turns by a non-dominant speaker -- i.e. candidate backchannels.

    The dominant speaker is the one with the most total speech; their short
    turns are ordinary sentence fragments, not backchannels.
    """
    if not segments:
        return []
    totals: dict[str, float] = {}
    for s in segments:
        totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + s["end"] - s["start"]
    main = max(totals, key=lambda k: totals[k])
    return [
        s for s in segments
        if s["speaker"] != main and s["end"] - s["start"] <= SHORT_MAX_S
    ]


def recall_against(baseline: list[dict], candidate: list[dict]) -> tuple[int, int]:
    """How many of the baseline's short non-main turns survive in candidate."""
    refs = short_turns(baseline)
    hits = 0
    for r in refs:
        span = r["end"] - r["start"]
        if span <= 0:
            continue
        best = max(
            (overlap(r["start"], r["end"], c["start"], c["end"]) / span
             for c in candidate),
            default=0.0,
        )
        if best >= COVERAGE_MIN:
            hits += 1
    return hits, len(refs)


def summarize(segments: list[dict]) -> dict:
    lengths = [s["end"] - s["start"] for s in segments]
    return {
        "n_segments": len(segments),
        "n_speakers": len({s["speaker"] for s in segments}),
        "speech_seconds": round(sum(lengths), 1),
        "n_short_non_main": len(short_turns(segments)),
        "median_seconds": round(statistics.median(lengths), 2) if lengths else 0.0,
        "min_seconds": round(min(lengths), 2) if lengths else 0.0,
    }


def main() -> int:
    args = parse_args()
    steps = [float(s) for s in args.steps.split(",") if s.strip()]
    files = collect_audio(args.audio)
    if not files:
        print(f"no audio matched {args.audio!r}", file=sys.stderr)
        return 1

    import soundfile as sf

    from pipeline_v2.params import PipelineParams
    from pipeline_v2.steps.speaker_diarization import Diarizer

    base_params = PipelineParams.from_config(args.config)
    print(f"{len(files)} file(s), steps={steps}, "
          f"device={base_params.device_name}\n")

    # step -> file -> {segments, infer_ms, ...}
    results: dict[float, dict[str, dict]] = {}

    for step in steps:
        diarization_params = base_params.diarization.model_copy(
            update={"diarizen_segmentation_step": step}
        )
        diarizer = Diarizer(diarization_params, base_params.device_name)
        results[step] = {}
        print(f"=== segmentation_step={step}")
        try:
            for audio in files:
                wave, sr = sf.read(audio, dtype="float32", always_2d=False)
                if wave.ndim > 1:
                    wave = wave.mean(axis=1)
                if args.max_seconds:
                    wave = wave[: int(args.max_seconds * sr)]
                duration = len(wave) / sr

                t0 = time.perf_counter()
                out = diarizer.run(np.ascontiguousarray(wave), sr)
                wall = time.perf_counter() - t0
                if out is None:
                    print(f"  {audio.name}: FAILED")
                    continue
                df, _ = out
                segments = [
                    {"start": float(r.start), "end": float(r.end),
                     "speaker": str(r.speaker)}
                    for r in df.itertuples()
                ]
                stats = summarize(segments)
                results[step][str(audio)] = {
                    "segments": segments, "wall_seconds": round(wall, 1),
                    "audio_seconds": round(duration, 1),
                    "rtf": round(wall / duration, 3) if duration else 0.0,
                    **stats,
                }
                print(f"  {audio.name}: wall={wall:6.1f}s rtf={wall/duration:5.3f} "
                      f"segs={stats['n_segments']:3} spk={stats['n_speakers']} "
                      f"short={stats['n_short_non_main']:3} "
                      f"speech={stats['speech_seconds']:7.1f}s")
        finally:
            diarizer.close()
        print()

    # ---- report -------------------------------------------------------
    baseline = results.get(args.baseline_step, {})
    print(f"{'step':>6} {'rtf':>7} {'speedup':>8} {'segs':>6} {'spk':>4} "
          f"{'short':>6} {'speech_s':>9} {'recall':>10}")
    base_rtf = statistics.mean(
        [v["rtf"] for v in baseline.values()]) if baseline else None

    report = {"baseline_step": args.baseline_step, "steps": {}}
    for step in steps:
        per_file = results[step]
        if not per_file:
            continue
        rtf = statistics.mean(v["rtf"] for v in per_file.values())
        hits = total = 0
        for path, value in per_file.items():
            if path in baseline:
                h, t = recall_against(baseline[path]["segments"], value["segments"])
                hits += h
                total += t
        row = {
            "rtf": round(rtf, 3),
            "speedup_vs_baseline": round(base_rtf / rtf, 2) if base_rtf and rtf else None,
            "n_segments": sum(v["n_segments"] for v in per_file.values()),
            "n_speakers_max": max(v["n_speakers"] for v in per_file.values()),
            "n_short_non_main": sum(v["n_short_non_main"] for v in per_file.values()),
            "speech_seconds": round(sum(v["speech_seconds"] for v in per_file.values()), 1),
            "short_turn_recall": f"{hits}/{total}" if total else "n/a",
            "short_turn_recall_pct": round(100.0 * hits / total, 1) if total else None,
        }
        report["steps"][str(step)] = row
        print(f"{step:>6} {row['rtf']:>7.3f} "
              f"{(str(row['speedup_vs_baseline']) + 'x'):>8} "
              f"{row['n_segments']:>6} {row['n_speakers_max']:>4} "
              f"{row['n_short_non_main']:>6} {row['speech_seconds']:>9.1f} "
              f"{row['short_turn_recall']:>10}")

    report["per_file"] = {
        str(step): {path: {k: v for k, v in value.items() if k != "segments"}
                    for path, value in per_file.items()}
        for step, per_file in results.items()
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    print(
        "\nReminder: judge a step by short_turn_recall AND n_speakers_max "
        "together.\nA coarser step suppressing spurious speakers is a win on a "
        "2-host podcast\nbut a real regression on a multi-party recording. "
        "Also watch the pipeline's\nown retain_seg_percent -- coarser "
        "segmentation finds fewer foreign overlaps,\nso the downstream guards "
        "drop less."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
