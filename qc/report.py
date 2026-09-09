"""Rendering the QC result as text, JSON and Markdown.

All three come from the same `sections` dict, so they cannot disagree -- the
text and Markdown renderers only choose what to surface and how to lay it out.
JSON is the complete picture; text is for reading in a terminal right after the
run; Markdown is for pasting into a doc.

One security property is enforced here rather than trusted: nothing derived from
the production config reaches a report except through `Thresholds.to_dict()`,
which is an explicit numeric allow-list. The config files hold
`huggingface_token` and third-party API keys in plain text
(configs/config_for_a10.json:46, :114), and a QC report is exactly the kind of
artefact that gets pasted into a chat or committed to a wiki. `_assert_no_secrets`
re-checks the rendered payload as a backstop against a future edit that widens
what gets echoed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from qc.config import QCConfig

# Substrings that must never appear as a key anywhere in a report payload.
_FORBIDDEN_KEY_PARTS = ("token", "api_key", "apikey", "secret", "password", "credential")


@dataclass
class Report:
    config: QCConfig
    sections: dict
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _fmt_int(v: Any) -> str:
    if v is None:
        return "-"
    return f"{int(v):,}"


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{float(v):,.{digits}f}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v):.2f}%"


def _hours(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v):,.2f}h"


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    """Fixed-width text table, right-aligned except the first column."""
    if not rows:
        return [indent + "(no data)"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = []
    head = indent + "  ".join(
        h.ljust(widths[i]) if i == 0 else h.rjust(widths[i])
        for i, h in enumerate(headers)
    )
    out.append(head)
    out.append(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        out.append(indent + "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        ))
    return out


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(no data)_", ""]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def _wrap(text: str, width: int = 96, indent: str = "  ") -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width,
                         initial_indent=indent, subsequent_indent=indent + "  ")


# ---------------------------------------------------------------------------
# section renderers -- each returns (text_lines, md_lines)
# ---------------------------------------------------------------------------

def _render_header(report: Report) -> tuple[list[str], list[str]]:
    cfg = report.config
    th = cfg.thresholds.to_dict()
    rows = [
        ["output root", cfg.output_root],
        ["manifest", cfg.manifest or "(not supplied)"],
        ["steps", ", ".join(cfg.steps)],
        ["shards", ", ".join(cfg.shards) if cfg.shards else "(all)"],
        ["sample size", "all segments" if cfg.sample_n <= 0 else
         f"{cfg.sample_n} {'per shard' if cfg.sample_per_shard else 'overall'}"],
        ["thresholds from", th["source"]],
        ["elapsed", f"{report.elapsed_seconds:.1f}s"],
    ]
    text = ["=" * 100,
            "pipeline_v3 OUTPUT QUALITY CONTROL REPORT",
            f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 100, ""]
    text += _table(["setting", "value"], rows)
    if th["is_default"]:
        text.append("")
        text += _wrap(
            "WARNING: thresholds are built-in defaults, not the values this data was "
            f"produced with (reason: {th['fallback_reason']}). Pass --config "
            "configs/config_for_<gpu>.json for an accurate report."
        )
    text.append("")
    text += _table(
        ["production threshold", "value"],
        [[k, _fmt_num(v, 4) if isinstance(v, float) else str(v)]
         for k, v in th.items()
         if k not in ("source", "is_default", "fallback_reason")],
    )

    md = ["# pipeline_v3 Output QC Report", "",
          f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, "
          f"elapsed {report.elapsed_seconds:.1f}s_", "",
          "## Run settings", ""]
    md += _md_table(["Setting", "Value"], [[r[0], f"`{r[1]}`"] for r in rows])
    if th["is_default"]:
        md += [f"> **WARNING** thresholds are built-in defaults, not the values this data "
               f"was produced with (reason: {th['fallback_reason']}). "
               "Pass `--config` for an accurate report.", ""]
    md += ["### Production thresholds in effect", ""]
    md += _md_table(["Threshold", "Value"],
                    [[k, f"`{v}`"] for k, v in th.items()
                     if k not in ("source", "is_default", "fallback_reason")])
    return text, md


def _render_yield(payload: dict) -> tuple[list[str], list[str]]:
    o = payload["overall"]
    raw, s1, s2, rates = o["raw"], o["stage1"], o["stage2"], o["rates"]

    funnel_rows = [
        ["0. raw corpus",
         _fmt_int(raw["files"]), _hours(raw["duration_hours"]), "-", "-"],
        ["1. stage 1 segments",
         _fmt_int(s1["files_with_output"]), _hours(s1["valid_duration_hours"]),
         _fmt_pct(rates["stage1_files_pct_of_raw"]),
         _fmt_pct(rates["stage1_duration_pct_of_raw"])],
        ["2. stage 2 kept (final)",
         "-", _hours(s2["kept_duration_hours"]),
         "-", _fmt_pct(rates["stage2_duration_pct_of_raw"])],
    ]
    text = ["", "-" * 100,
            "[1] EFFECTIVE DATA RATE: raw -> stage 1 -> stage 2",
            "-" * 100, ""]
    text += _table(["level", "files", "duration", "files % of raw", "duration % of raw"],
                   funnel_rows)
    text += ["", "  stage detail:"]
    text += _table(
        ["metric", "value"],
        [
            ["raw files", _fmt_int(raw["files"])],
            ["raw files w/o probed duration", _fmt_int(raw["files_unknown_duration"])],
            ["stage1 rows", _fmt_int(s1["rows"])],
            ["stage1 valid segments", _fmt_int(s1["valid_segments"])],
            ["stage1 failed-file rows", _fmt_int(s1["failed_file_rows"])],
            ["stage1 chunk wavs", _fmt_int(s1["chunk_wavs"])],
            ["stage2 rows", _fmt_int(s2["rows"])],
            ["stage2 kept rows", _fmt_int(s2["kept_rows"])],
            ["stage2 keep rate (rows)", _fmt_pct(s2["keep_rate_by_rows_pct"])],
            ["stage2 keep rate (duration)", _fmt_pct(s2["keep_rate_by_duration_pct"])],
            ["stage2 error rows", _fmt_int(s2["error_rows"])],
            ["stage2 retriable (asr_access_failed)", _fmt_int(s2["retriable_error_rows"])],
            ["stage2 kept % of stage1 (duration)",
             _fmt_pct(rates["stage2_duration_pct_of_stage1"])],
        ],
        indent="    ",
    )

    drop_rows = [
        [name.replace("dropped_by_", ""), _fmt_int(d["rows"]),
         _hours(d["duration_hours"]), _fmt_pct(d["rows_pct_of_total"])]
        for name, d in sorted(s2["dropped_by"].items(),
                              key=lambda kv: -kv[1]["rows"])
    ]
    text += ["", "  stage 2 drop reasons (NOT mutually exclusive -- one segment can trip several):"]
    text += _table(["reason", "rows", "duration", "% of stage2 rows"], drop_rows,
                   indent="    ")

    if s1["error_types"]:
        text += ["", "  stage 1 failure types:"]
        text += _table(["type", "files"],
                       [[k, _fmt_int(v)] for k, v in s1["error_types"].items()],
                       indent="    ")

    per_shard = payload.get("per_shard") or {}
    if len(per_shard) > 1:
        rows = []
        for name, f in sorted(per_shard.items()):
            rows.append([
                name,
                _fmt_int(f["stage1"]["valid_segments"]),
                _hours(f["stage1"]["valid_duration_hours"]),
                _fmt_int(f["stage2"]["kept_rows"]),
                _hours(f["stage2"]["kept_duration_hours"]),
                _fmt_pct(f["stage2"]["keep_rate_by_duration_pct"]),
            ])
        text += ["", "  per shard:"]
        text += _table(["shard", "s1 segs", "s1 dur", "s2 kept", "s2 dur",
                        "s2 keep% (dur)"], rows, indent="    ")

    text += _render_notes_text(payload)

    md = ["## 1. Effective data rate (raw -> stage 1 -> stage 2)", ""]
    md += _md_table(["Level", "Files", "Duration", "Files % of raw", "Duration % of raw"],
                    funnel_rows)
    md += ["### Stage 2 drop reasons", "",
           "_Not mutually exclusive: one segment can trip several filters._", ""]
    md += _md_table(["Reason", "Rows", "Duration", "% of stage-2 rows"], drop_rows)
    md += _render_notes_md(payload)
    return text, md


def _render_duration(payload: dict) -> tuple[list[str], list[str]]:
    o = payload["overall"]
    summary = o["summary"]
    text = ["", "-" * 100,
            "[2] FINAL OUTPUT DURATION DISTRIBUTION",
            "-" * 100, ""]
    text += _table(["metric", "value"], [
        ["kept segments", _fmt_int(o["segments"])],
        ["total duration", _hours(o["total_hours"])],
        ["mean", f"{_fmt_num(summary['mean'])}s"],
        ["p50", f"{_fmt_num(summary['p50'])}s"],
        ["p90", f"{_fmt_num(summary['p90'])}s"],
        ["p99", f"{_fmt_num(summary['p99'])}s"],
        ["min", f"{_fmt_num(summary['min'])}s"],
        ["max", f"{_fmt_num(summary['max'])}s"],
    ])

    def bucket_rows(buckets: dict) -> list[list[str]]:
        return [
            [label, _fmt_int(b["count"]), _fmt_pct(b["count_pct"]),
             _hours(b["duration_hours"]), _fmt_pct(b["duration_pct"])]
            for label, b in buckets.items()
        ]

    coarse = bucket_rows(o["coarse_buckets"])
    fine = bucket_rows(o["fine_buckets"])
    headers = ["bucket", "segments", "% of segs", "duration", "% of dur"]
    text += ["", "  coarse buckets (same as tmp/stat_parquet.py):"]
    text += _table(headers, coarse, indent="    ")
    text += ["", "  fine buckets (same as misc/analyze_output.py):"]
    text += _table(headers, fine, indent="    ")

    langs = o.get("by_language") or {}
    if langs:
        text += ["", "  by language:"]
        text += _table(
            ["language", "segments", "% of segs", "duration", "% of dur"],
            [[lang, _fmt_int(v["count"]), _fmt_pct(v["count_pct"]),
              _hours(v["duration_hours"]), _fmt_pct(v["duration_pct"])]
             for lang, v in list(langs.items())[:12]],
            indent="    ",
        )

    bounds = o["outside_production_bounds"]
    text += ["", "  outside production bounds:"]
    text += _table(["check", "segments"], [
        ["shorter than min_segment_length", _fmt_int(bounds["below_min_segment_length"])],
        ["longer than max_segment_length", _fmt_int(bounds["above_max_segment_length"])],
    ], indent="    ")
    text += _render_notes_text(payload)

    md = ["## 2. Final output duration distribution", ""]
    md += _md_table(["Metric", "Value"], [
        ["Kept segments", _fmt_int(o["segments"])],
        ["Total duration", _hours(o["total_hours"])],
        ["Mean / P50 / P90 / P99",
         f"{_fmt_num(summary['mean'])}s / {_fmt_num(summary['p50'])}s / "
         f"{_fmt_num(summary['p90'])}s / {_fmt_num(summary['p99'])}s"],
    ])
    md += ["### Coarse buckets", ""]
    md += _md_table(headers, coarse)
    md += ["### Fine buckets", ""]
    md += _md_table(headers, fine)
    md += _render_notes_md(payload)
    return text, md


def _render_speaker(payload: dict) -> tuple[list[str], list[str]]:
    text = ["", "-" * 100,
            "[3] SINGLE-SPEAKER RE-CHECK (embedding vs diarization)",
            "-" * 100, ""]
    md = ["## 3. Single-speaker re-check (two independent methods)", ""]
    if payload.get("skipped"):
        text += _wrap(f"SKIPPED: {payload['reason']}")
        md += [f"> **Skipped**: {payload['reason']}", ""]
        return text, md

    emb, dia, mx = payload["embedding_method"], payload["diarization_method"], payload["cross_check"]
    sampling = payload.get("sampling") or {}
    method_rows = [
        ["embedding self-consistency", _fmt_int(emb["scored"]),
         _fmt_int(emb["multi_speaker"]), _fmt_pct(emb["multi_speaker_pct"]),
         _fmt_int(emb["failed"])],
        ["pyannote diarization", _fmt_int(dia["scored"]),
         _fmt_int(dia["multi_speaker"]), _fmt_pct(dia["multi_speaker_pct"]),
         _fmt_int(dia["failed"])],
    ]
    text += _table(["metric", "value"], [
        ["segments checked", _fmt_int(payload["segments_checked"])],
        ["duration checked", _hours(payload["total_hours"])],
        ["sample rate", _fmt_pct((sampling.get("rate") or 0) * 100)],
        ["multi-speaker duration", _hours(payload["multi_speaker_duration_hours"])],
        ["multi-speaker duration %", _fmt_pct(payload["multi_speaker_duration_pct"])],
    ])
    text += ["", "  per method:"]
    text += _table(["method", "scored", "multi-speaker", "multi %", "failed"],
                   method_rows, indent="    ")

    matrix_rows = [
        ["both say multi-speaker", _fmt_int(mx["both_multi_speaker"])],
        ["embedding only", _fmt_int(mx["embedding_only_multi"])],
        ["diarization only", _fmt_int(mx["diarization_only_multi"])],
        ["both say single-speaker", _fmt_int(mx["both_single_speaker"])],
        ["agreement", _fmt_pct(mx["agreement_pct"])],
    ]
    text += ["", f"  cross-check ({_fmt_int(mx['compared'])} segments with both verdicts):"]
    text += _table(["outcome", "segments"], matrix_rows, indent="    ")

    by_len = payload.get("cross_check_by_length") or {}
    if by_len:
        text += ["", "  by segment length:"]
        text += _table(
            ["band", "compared", "both multi", "emb only", "dia only", "agreement"],
            [[band, _fmt_int(v["compared"]), _fmt_int(v["both_multi_speaker"]),
              _fmt_int(v["embedding_only_multi"]), _fmt_int(v["diarization_only_multi"]),
              _fmt_pct(v["agreement_pct"])]
             for band, v in by_len.items()],
            indent="    ",
        )

    sim = emb["min_similarity"]
    text += ["", "  min-similarity distribution (embedding method):"]
    text += _table(["stat", "value"], [
        ["mean", _fmt_num(sim["mean"], 4)],
        ["p50", _fmt_num(sim["p50"], 4)],
        ["p90", _fmt_num(sim["p90"], 4)],
        ["min", _fmt_num(sim["min"], 4)],
    ], indent="    ")
    dist = dia.get("speaker_count_distribution") or {}
    if dist:
        text += ["", "  diarization speaker-count distribution:"]
        text += _table(["speakers", "segments"],
                       [[k, _fmt_int(v)] for k, v in dist.items()], indent="    ")
    text += _render_notes_text(payload)

    md += _md_table(["Method", "Scored", "Multi-speaker", "Multi %", "Failed"], method_rows)
    md += ["### Cross-check matrix", ""]
    md += _md_table(["Outcome", "Segments"], matrix_rows)
    if by_len:
        md += ["### By segment length", ""]
        md += _md_table(["Band", "Compared", "Both multi", "Emb only", "Dia only", "Agreement"],
                        [[band, _fmt_int(v["compared"]), _fmt_int(v["both_multi_speaker"]),
                          _fmt_int(v["embedding_only_multi"]),
                          _fmt_int(v["diarization_only_multi"]),
                          _fmt_pct(v["agreement_pct"])]
                         for band, v in by_len.items()])
    md += _render_notes_md(payload)
    return text, md


def _render_background(payload: dict) -> tuple[list[str], list[str]]:
    text = ["", "-" * 100,
            "[4] BACKGROUND-REMOVAL RE-CHECK",
            "-" * 100, ""]
    md = ["## 4. Background-removal re-check", ""]
    if payload.get("skipped"):
        text += _wrap(f"SKIPPED: {payload['reason']}")
        md += [f"> **Skipped**: {payload['reason']}", ""]
        return text, md

    grade_rows = [
        [grade, _fmt_int(g["count"]), _fmt_pct(g["count_pct"]),
         _hours(g["duration_hours"]), _fmt_pct(g["duration_pct"])]
        for grade, g in payload["grades"].items()
    ]
    sampling = payload.get("sampling") or {}
    text += _table(["metric", "value"], [
        ["segments checked", _fmt_int(payload["segments_checked"])],
        ["segments graded", _fmt_int(payload["graded"])],
        ["sample rate", _fmt_pct((sampling.get("rate") or 0) * 100)],
    ])
    text += ["", "  background grade (DNSMOS BAK):"]
    text += _table(["grade", "segments", "% of graded", "duration", "% of dur"],
                   grade_rows, indent="    ")

    metrics = payload["metrics"]
    text += ["", "  metric distributions:"]
    text += _table(
        ["metric", "scored", "mean", "p50", "p90", "min", "max"],
        [[name, _fmt_int(m["count"]), _fmt_num(m["mean"]), _fmt_num(m["p50"]),
          _fmt_num(m["p90"]), _fmt_num(m["min"]), _fmt_num(m["max"])]
         for name, m in metrics.items()],
        indent="    ",
    )

    by_len = payload.get("grades_by_length") or {}
    if by_len:
        text += ["", "  grade by segment length:"]
        text += _table(
            ["band", "segments", "clean", "mild", "clear"],
            [[band, _fmt_int(v["segments"]), _fmt_pct(v["clean"]["count_pct"]),
              _fmt_pct(v["mild_residual"]["count_pct"]),
              _fmt_pct(v["clear_residual"]["count_pct"])]
             for band, v in by_len.items()],
            indent="    ",
        )

    fails = payload["scoring_failures"]
    below = payload["below_production_thresholds"]
    text += ["", "  scoring failures / production-threshold violations:"]
    text += _table(["check", "segments"], [
        ["dnsmos failed", _fmt_int(fails["dnsmos_failed"])],
        ["brouhaha failed", _fmt_int(fails["brouhaha_failed"])],
        ["brouhaha sentinel (-420.69)", _fmt_int(fails["brouhaha_sentinel_values"])],
        ["below production dnsmos threshold", _fmt_int(below["dnsmos_ovrl"])],
        ["below production snr threshold", _fmt_int(below["brouhaha_snr"])],
        ["below production c50 threshold", _fmt_int(below["brouhaha_c50"])],
    ], indent="    ")

    drift = payload["recorded_vs_recheck_drift"]
    text += ["", "  drift vs. values recorded by the pipeline:"]
    text += _table(["metric", "compared", "mean delta", "p50 delta"],
                   [[name, _fmt_int(d["compared"]), _fmt_num(d["mean"], 4),
                     _fmt_num(d["p50"], 4)]
                    for name, d in drift.items()], indent="    ")
    text += _render_notes_text(payload)

    md += _md_table(["Grade", "Segments", "% of graded", "Duration", "% of duration"],
                    grade_rows)
    md += ["### Metric distributions", ""]
    md += _md_table(["Metric", "Scored", "Mean", "P50", "P90", "Min", "Max"],
                    [[name, _fmt_int(m["count"]), _fmt_num(m["mean"]),
                      _fmt_num(m["p50"]), _fmt_num(m["p90"]), _fmt_num(m["min"]),
                      _fmt_num(m["max"])] for name, m in metrics.items()])
    md += _render_notes_md(payload)
    return text, md


def _render_merge(payload: dict) -> tuple[list[str], list[str]]:
    o = payload["overall"]
    text = ["", "-" * 100,
            "[5] ADJACENT SAME-SPEAKER MERGEABILITY",
            "-" * 100, ""]
    cat_rows = [
        [name, _fmt_int(c["pairs"]), _fmt_pct(c["pairs_pct"]), _hours(c["duration_hours"])]
        for name, c in o["categories"].items()
    ]
    text += _table(["metric", "value"], [
        ["chunks analysed", _fmt_int(o["chunks"])],
        ["chunks with a single segment", _fmt_int(o["single_segment_chunks"])],
        ["adjacent pairs", _fmt_int(o["adjacent_pairs"])],
        ["pairs with negative gap (grace period)", _fmt_int(o["negative_gap_pairs"])],
        ["pairs with zero gap", _fmt_int(o["zero_gap_pairs"])],
    ])
    text += ["", "  classification by first blocking condition (partitions all pairs):"]
    text += _table(["category", "pairs", "% of pairs", "duration"], cat_rows, indent="    ")

    emb = o["embedding_check"]
    text += ["", "  embedding-similarity check on structurally-mergeable pairs (sampled):"]
    text += _table(["metric", "value"], [
        ["pairs checked", _fmt_int(emb["checked_pairs"])],
        ["similarity >= threshold", _fmt_int(emb["similarity_pass_pairs"])],
        ["pass rate", _fmt_pct(emb["similarity_pass_rate_pct"])],
        ["failed to score", _fmt_int(emb["failed_pairs"])],
        ["similarity mean", _fmt_num(emb["similarity"]["mean"], 4)],
        ["similarity p50", _fmt_num(emb["similarity"]["p50"], 4)],
    ], indent="    ")

    missed = o["suspected_missed_merges"]
    text += ["", "  suspected missed merges:"]
    text += _table(["metric", "value"], [
        ["structurally mergeable pairs", _fmt_int(missed["structurally_mergeable_pairs"])],
        ["as % of all pairs", _fmt_pct(missed["structurally_mergeable_pct_of_pairs"])],
        ["estimated after embedding filter",
         _fmt_int(missed["estimated_after_embedding_filter"])],
        ["estimated as % of all pairs", _fmt_pct(missed["estimated_pct_of_pairs"])],
    ], indent="    ")

    gap = o["gap_seconds"]
    text += ["", "  gap distribution (seconds between adjacent segments):"]
    text += _table(["stat", "value"], [
        ["mean", _fmt_num(gap["mean"], 3)],
        ["p50", _fmt_num(gap["p50"], 3)],
        ["p90", _fmt_num(gap["p90"], 3)],
        ["max", _fmt_num(gap["max"], 3)],
    ], indent="    ")
    text += _render_notes_text(payload)

    md = ["## 5. Adjacent same-speaker mergeability", ""]
    md += _md_table(["Category", "Pairs", "% of pairs", "Duration"], cat_rows)
    md += ["### Suspected missed merges", ""]
    md += _md_table(["Metric", "Value"], [
        ["Structurally mergeable pairs", _fmt_int(missed["structurally_mergeable_pairs"])],
        ["As % of all pairs", _fmt_pct(missed["structurally_mergeable_pct_of_pairs"])],
        ["Estimated after embedding filter",
         _fmt_int(missed["estimated_after_embedding_filter"])],
        ["Embedding pass rate (sampled)", _fmt_pct(emb["similarity_pass_rate_pct"])],
    ])
    md += _render_notes_md(payload)
    return text, md


def _render_notes_text(payload: dict) -> list[str]:
    import textwrap

    out: list[str] = []
    notes = payload.get("notes") or []
    if notes:
        out += ["", "  notes:"]
        for note in notes:
            out += textwrap.wrap(
                note, width=96, initial_indent="    - ", subsequent_indent="      "
            )
    anomalies = (payload.get("anomalies") or {}).get("counts") or {}
    if anomalies:
        out += ["", "  anomalies:"]
        out += _table(["kind", "count"],
                      [[k, _fmt_int(v)] for k, v in anomalies.items()], indent="    ")
    return out


def _render_notes_md(payload: dict) -> list[str]:
    out: list[str] = []
    notes = payload.get("notes") or []
    if notes:
        out += ["**Notes**", ""]
        out += [f"- {n}" for n in notes]
        out.append("")
    anomalies = (payload.get("anomalies") or {}).get("counts") or {}
    if anomalies:
        out += ["**Anomalies**", ""]
        out += _md_table(["Kind", "Count"], [[k, _fmt_int(v)] for k, v in anomalies.items()])
    return out


# ---------------------------------------------------------------------------
# secret guard
# ---------------------------------------------------------------------------

def _assert_no_secrets(payload: Any, path: str = "") -> None:
    """Fail loudly if a credential-looking key made it into a report.

    A backstop, not the primary defence -- `Thresholds.to_dict()` is an explicit
    allow-list. But QC reports get pasted around freely, so it is worth crashing
    the render rather than writing a token to disk if someone later widens what
    gets echoed.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                raise RuntimeError(
                    f"refusing to write a report containing a credential-like key: "
                    f"{path}.{key}"
                )
            _assert_no_secrets(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            _assert_no_secrets(item, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_SECTION_RENDERERS = (
    ("yield", _render_yield),
    ("duration", _render_duration),
    ("speaker", _render_speaker),
    ("background", _render_background),
    ("merge", _render_merge),
)


def render(report: Report, quiet: bool = False) -> list[str]:
    """Write txt/json/md next to each other; return the paths written."""
    cfg = report.config
    text_lines, md_lines = _render_header(report)

    shard_info = report.sections.get("shards") or {}
    if shard_info.get("warning"):
        text_lines += ["", f"WARNING: {shard_info['warning']}"]
        md_lines += [f"> **WARNING** {shard_info['warning']}", ""]

    for name, renderer in _SECTION_RENDERERS:
        payload = report.sections.get(name)
        if payload is None:
            continue
        t, m = renderer(payload)
        text_lines += t
        md_lines += m

    text_lines += ["", "=" * 100, "end of report", "=" * 100]

    json_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(report.elapsed_seconds, 3),
        "runtime": cfg.runtime_to_dict(),
        "thresholds": cfg.thresholds.to_dict(),
        "sections": report.sections,
    }
    _assert_no_secrets(json_payload)

    os.makedirs(cfg.report_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(cfg.report_dir, f"qc_report_{stamp}")
    text_body = "\n".join(text_lines) + "\n"
    md_body = "\n".join(md_lines) + "\n"

    paths = []
    for suffix, body in ((".txt", text_body), (".md", md_body)):
        path = base + suffix
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        paths.append(path)
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, ensure_ascii=False, indent=2, default=_json_default)
    paths.append(json_path)

    if not quiet:
        print(text_body)
    return paths


def _json_default(obj: Any):
    """Last-resort encoder for stray non-JSON types (e.g. a Counter's keys)."""
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)
