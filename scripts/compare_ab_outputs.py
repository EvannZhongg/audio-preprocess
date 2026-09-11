#!/usr/bin/env python3
"""Compare two or more PipelineV2 output trees.

Every mode goes through the same `pipeline_v2.steps.export.Exporter`, so this
reads their JSON sidecars directly and reports yield, retention, quality and
per-stage timings side by side.

The headline number is deliberately NOT segment count -- the optimized flow
retains less audio on purpose -- so this also reports what that flow exists to
fix: cross-speaker contamination, measured structurally as exported segments
whose span overlaps another speaker's exported segment.

    python scripts/compare_ab_outputs.py ab_out/baseline ab_out/optimized
    python scripts/compare_ab_outputs.py ab_out/baseline ab_out/modelswap ab_out/optimized
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
    p.add_argument("trees", nargs="+", help="output directories, in order")
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
    """Exported segments overlapping a DIFFERENT speaker's exported segment.

    All exported segments in a chunk come from one diarization pass, so a
    cross-speaker overlap means two speakers were handed the same audio --
    exactly the swallowed-backchannel failure this work targets. Structural
    check on timestamps, so it is stable and cheap, but it can only see
    contamination that diarization itself labelled: a backchannel the model
    never detected is invisible here. Compare against a finer-grained run (or
    listen) before concluding a mode is clean.
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


def swallowed(by_chunk: dict[str, list[dict]],
              reference: dict[str, list[dict]]) -> dict:
    """Short foreign turns from `reference` that land INSIDE this mode's segments.

    This exists because `contamination` has a blind spot that matters exactly
    here: it can only see contamination the mode's own diarizer labelled. A
    diarizer that never detects a backchannel at all reports zero overlap and
    looks perfectly clean, while in truth it silently swallowed the turn --
    which is the original failure this whole effort is about.

    So score every mode against a shared reference (the union of all modes'
    detections): a reference turn counted here is audio some diarizer says
    belongs to another speaker, sitting inside a segment this mode exported
    and attributed to someone else.
    """
    total = 0
    swallowed_turns = 0
    swallowed_seconds = 0.0
    for chunk, refs in reference.items():
        segments = by_chunk.get(chunk, [])
        for r in refs:
            total += 1
            r_span = seg_span(r)
            for s in segments:
                if s.get("speaker_id") == r.get("speaker_id"):
                    continue
                # >=90% contained means the foreign turn is essentially wholly
                # inside a segment credited to a different speaker.
                span = r_span[1] - r_span[0]
                if span > 0 and overlap(r_span, seg_span(s)) / span >= 0.9:
                    swallowed_turns += 1
                    swallowed_seconds += span
                    break
    return {
        "reference_turns": total,
        "swallowed_turns": swallowed_turns,
        "percent": round(100.0 * swallowed_turns / total, 1) if total else None,
        "swallowed_seconds": round(swallowed_seconds, 1),
    }


def build_reference(all_by_chunk: list[dict[str, list[dict]]]) -> dict[str, list[dict]]:
    """Union of every mode's short (<=1.5s) non-dominant-speaker turns.

    Using the union rather than any single mode avoids privileging one
    diarizer's recall when deciding what "should" have been detected.
    """
    reference: dict[str, list[dict]] = defaultdict(list)
    for by_chunk in all_by_chunk:
        for chunk, segments in by_chunk.items():
            if not segments:
                continue
            totals: dict[str, float] = defaultdict(float)
            for s in segments:
                a, b = seg_span(s)
                totals[s.get("speaker_id")] += b - a
            main = max(totals, key=lambda k: totals[k])
            for s in segments:
                a, b = seg_span(s)
                if s.get("speaker_id") != main and 0 < b - a <= 1.5:
                    # Deduplicate near-identical turns found by several modes.
                    if not any(
                        overlap((a, b), seg_span(e)) / (b - a) >= 0.5
                        for e in reference[chunk]
                    ):
                        reference[chunk].append(s)
    return reference


def quality(sentences: list[dict], name: str) -> dict:
    values = [
        float(s["audio_quality_info"][name])
        for s in sentences
        if s.get("audio_quality_info", {}).get(name) is not None
        # -1 (soft-degrade) and -420.69 (brouhaha failure) are sentinels, not
        # measurements; including them would poison the median.
        and float(s["audio_quality_info"][name]) > -100
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
_DIA_RE = re.compile(r"dia_time_cost .*?infer_ms (\d+).*?spawn_ms (\d+)")


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
    dia = _DIA_RE.findall(text)
    out = {
        "stage_seconds": {
            k: round(v, 1)
            for k, v in sorted(stages.items(), key=lambda kv: -kv[1])
        },
        "chunk_audio_seconds": round(audio, 1),
        "chunk_wall_seconds": round(wall, 1),
        "throughput_x_realtime": round(audio / wall, 2) if wall else None,
        "median_retain_seg_percent": (
            round(statistics.median(retain), 1) if retain else None
        ),
    }
    if dia:
        # Only the resident DiariZen path emits these; spawn_ms is non-zero on
        # the first chunk only, so its total is the one-off worker startup.
        out["diarizen_infer_seconds"] = round(sum(int(i) for i, _ in dia) / 1000.0, 1)
        out["diarizen_spawn_seconds"] = round(sum(int(s) for _, s in dia) / 1000.0, 1)
    return out


def summarize(tree: Path) -> dict:
    sentences, by_chunk = load(tree)
    durations = [
        float(s.get("time_range", {}).get("duration", 0.0)) for s in sentences
    ]
    speakers: dict[str, float] = defaultdict(float)
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
        "_by_chunk": by_chunk,
        "timings": timings(tree.parent / f"{tree.name}.run.log"),
    }


def main() -> int:
    args = parse_args()
    trees = [Path(t) for t in args.trees]
    missing = [t for t in trees if not t.is_dir()]
    if missing:
        print("missing output tree(s): " + ", ".join(map(str, missing)))
        return 1
    names = [t.name for t in trees]
    data = {n: summarize(t) for n, t in zip(names, trees)}

    # Score every mode against a shared reference so a diarizer that simply
    # never detected a backchannel cannot look clean by omission.
    reference = build_reference([data[n]["_by_chunk"] for n in names])
    for n in names:
        data[n]["swallowed"] = swallowed(data[n].pop("_by_chunk"), reference)

    width = max(16, max(len(n) for n in names) + 2)

    def line(label: str, values, unit: str = "") -> None:
        cells = "".join(
            f"{(str(v) + unit) if v is not None else '-':>{width}}" for v in values
        )
        print(f"  {label:32}{cells}")

    def pull(*path):
        out = []
        for n in names:
            cur = data[n]
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
                if cur is None:
                    break
            out.append(cur)
        return out

    header = "".join(f"{n:>{width}}" for n in names)
    print(f"\n  {'':32}{header}")
    print("  " + "-" * (32 + width * len(names)))
    line("chunks", pull("chunks"))
    line("segments", pull("segments"))
    line("retained audio", pull("retained_seconds"), "s")
    line("retention", pull("retention_percent"), "%")
    line("median segment", pull("segment_seconds", "median"), "s")
    line("min segment", pull("segment_seconds", "min"), "s")
    line("max segment", pull("segment_seconds", "max"), "s")
    line("speakers found", pull("speakers"))
    print("  " + "-" * (32 + width * len(names)))
    line("median dnsmos", pull("dnsmos", "median"))
    line("min dnsmos", pull("dnsmos", "min"))
    line("median c50", pull("c50", "median"))
    line("median snr", pull("snr", "median"))
    print("  " + "-" * (32 + width * len(names)))
    print("  cross-speaker contamination (lower is better)")
    line("contaminated segments",
         pull("contamination", "segments_with_cross_speaker_overlap"))
    line("  as percent", pull("contamination", "percent"), "%")
    line("  overlapping audio", pull("contamination", "overlap_seconds"), "s")
    print("  " + "-" * (32 + width * len(names)))
    ref_n = pull("swallowed", "reference_turns")
    print(f"  swallowed backchannels (vs {ref_n[0]} shared reference turns)")
    line("swallowed turns", pull("swallowed", "swallowed_turns"))
    line("  as percent", pull("swallowed", "percent"), "%")
    line("  swallowed audio", pull("swallowed", "swallowed_seconds"), "s")
    print("  " + "-" * (32 + width * len(names)))
    line("throughput", pull("timings", "throughput_x_realtime"), "x")
    line("diarization wall", pull("timings", "stage_seconds", "dia"), "s")
    line("separation wall", pull("timings", "stage_seconds", "sep"), "s")
    line("  diarizen infer", pull("timings", "diarizen_infer_seconds"), "s")
    line("  diarizen spawn", pull("timings", "diarizen_spawn_seconds"), "s")

    for n in names:
        t = data[n]["timings"]
        if not t:
            continue
        print(f"\n  {n} stage breakdown "
              f"({t['chunk_audio_seconds']}s audio in {t['chunk_wall_seconds']}s)")
        for stage, seconds in t["stage_seconds"].items():
            share = (
                100.0 * seconds / t["chunk_wall_seconds"]
                if t["chunk_wall_seconds"] else 0.0
            )
            print(f"    {stage:8} {seconds:9.1f}s  {share:5.1f}%")

    print("\n  per-speaker retained seconds")
    for n in names:
        top = list(data[n]["per_speaker_seconds"].items())[:4]
        print(f"    {n:14} " + "  ".join(f"{k}={v}s" for k, v in top))

    if len(names) == 3:
        print("\n  Attribution: the 3-way is designed so each step isolates one"
              "\n  thing. first->second is the diarization MODEL (flow identical);"
              "\n  second->third is the segmentation FLOW (model identical).")
    print("\n  Read this as a trade, not a win/loss: the optimized flow retains")
    print("  LESS audio on purpose. Judge it on contamination down vs retention")
    print("  lost, and on whether per-speaker minutes still cover your training")
    print("  need. Listen to a sample before trusting any of these numbers.")
    print("\n  'contaminated segments' only sees overlap a mode's OWN diarizer")
    print("  labelled, so a model that misses a backchannel scores 0 there while")
    print("  silently swallowing it. 'swallowed backchannels' is the honest")
    print("  cross-model number -- compare modes on that one.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
