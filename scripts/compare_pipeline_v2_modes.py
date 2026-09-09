#!/usr/bin/env python3
"""Run PipelineV2 baseline and local_adapter_v2 optimized modes side by side.

Both runs use the unchanged PipelineV2 exporter/schema. Only their config and
output roots differ, which makes the generated JSON/WAV/parquet-compatible
records directly comparable.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PAYLOAD_KEYS = {
    "file_name",
    "source",
    "pipeline_version",
    "chunk_index",
    "audio_path",
    "sample_rate",
    "duration",
    "sentences",
}
EXPECTED_SENTENCE_KEYS = {
    "utt_id",
    "speaker_id",
    "speaker_min_similarity",
    "time_range",
    "audio_quality_info",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="audio file or directory")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--baseline-config",
        default="configs/config.json",
        help="existing PipelineV2 config",
    )
    parser.add_argument(
        "--optimized-config",
        default="configs/config_pipeline_v2_diarizen_tts_clean_v2.json",
        help="native PipelineV2 config carrying the local_adapter_v2 optimizations",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove existing baseline/optimized output directories first",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for the PipelineV2 parent process",
    )
    return parser.parse_args()


def run_mode(
    python: str,
    label: str,
    config: str,
    input_path: str,
    output_root: Path,
    num_workers: int,
) -> None:
    output = output_root / label
    command = [
        python,
        str(ROOT / "main_v2.py"),
        "--config",
        str((ROOT / config).resolve() if not Path(config).is_absolute() else config),
        "--input",
        input_path,
        "--output",
        str(output),
        "--num-workers",
        str(num_workers),
    ]
    print(f"[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(output: Path) -> dict:
    json_paths = sorted((output / "jsons").glob("*/*.json"))
    segments: list[dict] = []
    payload_keys: set[str] = set()
    sentence_keys: set[str] = set()
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_keys.update(payload)
        for segment in payload.get("sentences", []):
            segments.append(segment)
            sentence_keys.update(segment)

    durations = [
        float(segment["time_range"]["duration"])
        for segment in segments
    ]
    quality = [
        segment.get("audio_quality_info", {})
        for segment in segments
    ]

    def median(name: str):
        values = [
            float(item[name])
            for item in quality
            if item.get(name) is not None
        ]
        return statistics.median(values) if values else None

    return {
        "json_files": len(json_paths),
        "segments": len(segments),
        "retained_seconds": round(sum(durations), 5),
        "min_segment_seconds": min(durations, default=None),
        "max_segment_seconds": max(durations, default=None),
        "median_dnsmos": median("dnsmos"),
        "median_c50": median("c50"),
        "median_snr": median("snr"),
        "payload_keys": sorted(payload_keys),
        "sentence_keys": sorted(sentence_keys),
        # A mode that legitimately yields zero segments writes no sidecar, so
        # an empty observed key set is "not observed" rather than a mismatch.
        "schema_valid": (
            (not payload_keys or payload_keys == EXPECTED_PAYLOAD_KEYS)
            and (not sentence_keys or sentence_keys == EXPECTED_SENTENCE_KEYS)
        ),
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for label in ("baseline", "optimized"):
        mode_output = output_root / label
        if mode_output.exists():
            if args.overwrite:
                shutil.rmtree(mode_output)
            elif any(mode_output.iterdir()):
                raise FileExistsError(
                    f"{mode_output} is not empty; choose a new --output-root "
                    "or pass --overwrite"
                )
    input_path = str(Path(args.input).expanduser().resolve())
    run_mode(
        args.python,
        "baseline",
        args.baseline_config,
        input_path,
        output_root,
        args.num_workers,
    )
    run_mode(
        args.python,
        "optimized",
        args.optimized_config,
        input_path,
        output_root,
        args.num_workers,
    )
    baseline = summarize(output_root / "baseline")
    optimized = summarize(output_root / "optimized")
    schema_match = baseline["schema_valid"] and optimized["schema_valid"]
    report = {
        "baseline": baseline,
        "optimized": optimized,
        "schema_match": schema_match,
    }
    report_path = output_root / "comparison_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"comparison outputs: {output_root / 'baseline'}")
    print(f"comparison outputs: {output_root / 'optimized'}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not schema_match:
        raise RuntimeError(
            "baseline and optimized JSON schemas differ; see "
            f"{report_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
