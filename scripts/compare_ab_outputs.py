#!/usr/bin/env python3
"""Compare two PipelineV2 output trees (baseline vs optimized).

Both modes go through the same `pipeline_v2.steps.export.Exporter`, so this
reads their JSON sidecars directly and reports yield, retention, quality and
per-stage timings side by side.

The headline number is not segment count -- the optimized flow deliberately
retains less audio in exchange for purity -- so this also reports the thing
the optimized flow exists to fix: cross-speaker contamination, measured
structurally as segments whose span overlaps another speaker's segment.

    python scripts/compare_ab_outputs.py ab_out/baseline ab_out/optimized
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("optimized")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load(tree: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (all sentences, sentences grouped by source chunk)."""
    sentences: list[dict] = []
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(tree.glob("jsons/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = f"{payload.get('source')}#{payload.get('chunk_index')}"
        for s in payload.get("sentences", []):
            s = dict(s)
            s["_chunk"] = key
            s["_chunk_duration"] = payload.get("duration")
            sentences.append(s)
            by_chunk[key].append(s)
    return sentences, by_chunk


def seg_span(s: dict) -> tuple[float, float]:
    tr = s.get("time_range", {})
    return float(tr.get("start", 0.0)), float(tr.get("end", 0.0))


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def contamination(by_chunk: dict[str, list[dict]]) -> dict:
    """Segments that overlap a DIFFERENT speaker's exported segment.

    Exported segments come from a single diarization pass, so a cross-speaker
    overlap means two speakers were given the same audio -- exactly the
    swallowed-backchannel failure the optimized flow targets. This is a
    structural check on timestamps, not a voiceprint test, so it is stable but
    it can only see contamination that diarization itself labelled.
    """
    contaminated = 0
    total = 0
    overlap_seconds = 0.0
    for segments in by_chunk.values():
        for i, s in enumerate(segments):
            total += 1
            span = seg_span(s)
            hit = 0.0
            for j, other in enumerate(segments):
                if i == j or other.get("speaker_id") == s.get("speaker_id"):
                    continue
                hit += overlap(span, seg_span(other))
            if hit > 0.0:
                contaminated += 1
                overlap_seconds += hit
    return {
        "segments_with_cross_speaker_overlap": contaminated,
        "percent": round(100.0 * contaminated / total, 2) if total else 0.0,
        "overlap_seconds": round(overlap_seconds, 1),
    }


def quality(sentences: list[dict], name: str) -> dict:
    values = [
        float(s["audio_quality_info"][name])
        for s in sentences
        if s.get("audio_quality_info", {}).get(name) is not None
        and float(s["audio_quality_info"][name]) > -100  # -1 / -420 = soft failure
    ]
    if not values:
        return {"median": None, "min": None, "n": 0}
    return {
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "n": len(values),
    }


_STAGE_RE = re.compile(
    r"(std|sep|dia|vad|emb|seg|metrics|export)_time_cost.*?total_ms (\d+)"
)
_CHUNK_RE = re.compile(r"chunk_done input_sec ([\d.]+) wall_ms (\d+)")
_RETAIN_RE = re.compile(r"retain_seg_percent ([\d.]+)%")


def timings(log_path: Path) -> dict:
    """Sum per-stage wall time out of a run log, if one was captured."""
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stages: dict[str, float] = defaultdict(float)
    for stage, ms in _STAGE_RE.findall(text):
        stages[stage] += int(ms) / 1000.0
    audio = sum(float(a) for a, _ in _CHUNK_RE.findall(text))
    wall = sum(int(w) for _, w in _CHUNK_RE.findall(text)) / 1000.0
    retain = [float(r) for r in _RETAIN_RE.findall(text)]
    out = {
        "stage_seconds": {k: round(v, 1) for k, v in sorted(
            stages.items(), key=lambda kv: -kv[1])},
        "chunk_audio_seconds": round(audio, 1),
        "chunk_wall_seconds": round(wall, 1),
        "throughput_x_realtime": round(audio / wall, 2) if wall else None,
        "median_retain_seg_percent": round(statistics.median(retain), 1) if retain else None,
    }
    return out


def summarize(tree: Path) -> dict:
    sentences, by_chunk = load(tree)
    durations = [
        float(s.get("time_range", {}).get("duration", 0.0)) for s in sentences
    ]
    speakers = defaultdict(float)
    for s, d in zip(sentences, durations):
        speakers[s.get("speaker_id")] += d
    chunk_seconds = sum(
        {k: v[0]["_chunk_duration"] or 0.0 for k, v in by_chunk.items()}.values()
    )
    return {
        "chunks": len(by_chunk),
        "segments": len(sentences),
        "retained_seconds": round(sum(durations), 1),
        "chunk_seconds": round(chunk_seconds, 1),
        "retention_percent": (
            round(100.0 * sum(durations) / chunk_seconds, 1) if chunk_seconds else None
        ),
        "segment_seconds": {
            "median": round(statistics.median(durations), 2) if durations else None,
            "min": round(min(durations), 2) if durations else None,
            "max": round(max(durations), 2) if durations else None,
        },
        "speakers": len(speakers),
        "per_speaker_seconds": {
            k: round(v, 1) for k, v in sorted(speakers.items(), key=lambda kv: -kv[1])
        },
        "dnsmos": quality(sentences, "dnsmos"),
        "c50": quality(sentences, "c50"),
        "snr": quality(sentences, "snr"),
        "contamination": contamination(by_chunk),
        "timings": timings(tree.parent / f"{tree.name}.run.log"),
    }


def main() -> int:
    args = parse_args()
    base = summarize(Path(args.baseline))
    opt = summarize(Path(args.optimized))

    def line(label: str, a, b, unit: str = "") -> None:
        print(f"  {label:34} {str(a) + unit:>18} {str(b) + unit:>18}")

    print(f"\n  {'':34} {'baseline':>18} {'optimized':>18}")
    print("  " + "-" * 72)
    line("chunks", base["chunks"], opt["chunks"])
    line("segments", base["segments"], opt["segments"])
    line("retained audio", base["retained_seconds"], opt["retained_seconds"], "s")
    line("retention", base["retention_percent"], opt["retention_percent"], "%")
    line("median segment", base["segment_seconds"]["median"],
         opt["segment_seconds"]["median"], "s")
    line("min segment", base["segment_seconds"]["min"],
         opt["segment_seconds"]["min"], "s")
    line("speakers found", base["speakers"], opt["speakers"])
    print("  " + "-" * 72)
    line("median dnsmos", base["dnsmos"]["median"], opt["dnsmos"]["median"])
    line("min dnsmos", base["dnsmos"]["min"], opt["dnsmos"]["min"])
    line("median c50", base["c50"]["median"], opt["c50"]["median"])
    line("median snr", base["snr"]["median"], opt["snr"]["median"])
    print("  " + "-" * 72)
    print("  cross-speaker contamination (the thing the optimized flow fixes)")
    line("contaminated segments",
         base["contamination"]["segments_with_cross_speaker_overlap"],
         opt["contamination"]["segments_with_cross_speaker_overlap"])
    line("  as percent", base["contamination"]["percent"],
         opt["contamination"]["percent"], "%")
    line("  overlapping audio", base["contamination"]["overlap_seconds"],
         opt["contamination"]["overlap_seconds"], "s")

    for label, data in (("baseline", base), ("optimized", opt)):
        t = data["timings"]
        if not t:
            continue
        print(f"\n  {label} timings")
        print(f"    throughput {t['throughput_x_realtime']}x realtime "
              f"({t['chunk_audio_seconds']}s audio in {t['chunk_wall_seconds']}s)")
        if t.get("median_retain_seg_percent") is not None:
            print(f"    median retain_seg_percent {t['median_retain_seg_percent']}%")
        for stage, seconds in t["stage_seconds"].items():
            share = (100.0 * seconds / t["chunk_wall_seconds"]
                     if t["chunk_wall_seconds"] else 0.0)
            print(f"    {stage:8} {seconds:9.1f}s  {share:5.1f}%")

    print("\n  Read this as a trade, not a win/loss: the optimized flow retains")
    print("  LESS audio on purpose. Judge it on contamination down vs retention")
    print("  lost, and on whether per-speaker minutes still cover your training")
    print("  need. Listen to a sample before trusting any of these numbers.")

    report = {"baseline": base, "optimized": opt}
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
