"""CLI entry point: `python -m qc.main`.

Orchestration only -- every statistic lives in `qc/analyzers/`, every model call
behind `qc/models_bundle.py`. Two passes run per invocation:

  1. parquet passes (requirements 1, 2, and the structural half of 5), over a
     process pool, always on the full dataset -- they are IO bound and cheap.
  2. one GPU pass (requirements 3, 4, and the embedding half of 5), on a
     deterministic sample by default. All three share a single audio decode per
     chunk, because decoding dominates and doing it three times would triple
     the wall clock for nothing.

Anything that fails degrades the report rather than aborting the run: a missing
manifest drops level 0, a shard without stage-2 parquet is reported as "stage 2
not run", an unavailable GPU falls back to CPU with a warning.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from typing import Optional

import logger
from logger import make_extra_tags
from qc.config import ALL_STEPS, QCConfig, Thresholds


def _parse_steps(raw: Optional[str]) -> tuple[str, ...]:
    if not raw or raw.strip() in {"all", "*"}:
        return ALL_STEPS
    wanted = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(
            f"unknown --steps value(s) {unknown}; choices: {', '.join(ALL_STEPS)}"
        )
    # Keep the canonical order so the report sections are always in the same
    # sequence regardless of how the user listed them.
    return tuple(s for s in ALL_STEPS if s in wanted)


def _parse_devices(raw: str) -> tuple[str, ...]:
    devices = tuple(d.strip() for d in raw.split(",") if d.strip())
    return devices or ("cuda:0",)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m qc.main",
        description="Offline quality control for pipeline_v3 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # everything, sampling 2000 segments per shard for the GPU checks\n"
            "  python -m qc.main --output /path/out --manifest /path/manifest \\\n"
            "      --config configs/config_for_a10.json\n\n"
            "  # parquet-only, no GPU needed, full dataset\n"
            "  python -m qc.main --output /path/out --steps yield,duration,merge\n\n"
            "  # exhaustive re-check on two GPUs (slow)\n"
            "  python -m qc.main --output /path/out --steps speaker,background \\\n"
            "      --sample-n 0 --gpu-workers 2 --devices cuda:0,cuda:1\n"
        ),
    )
    p.add_argument("--output", required=True,
                   help="pipeline_v3 output root (the --output passed to main_v3_ray.py)")
    p.add_argument("--manifest",
                   help="raw-data manifest parquet (file or shard dir) from build_manifest.py; "
                        "without it the level-0 baseline is unavailable and stage-1 yield "
                        "can only be reported in absolute terms")
    p.add_argument("--config",
                   help="production config json (configs/config_for_*.json) the run used; "
                        "thresholds are read from it so QC verdicts match production")
    p.add_argument("--shards", help="comma-separated shard names to limit the run to")
    p.add_argument("--steps", default="all",
                   help=f"comma-separated subset of: {', '.join(ALL_STEPS)} (default: all)")

    g = p.add_argument_group("sampling (GPU re-checks only)")
    g.add_argument("--sample-n", type=int, default=2000,
                   help="segments to re-check; 0 means every segment (default: 2000)")
    g.add_argument("--sample-scope", choices=("per-shard", "global"), default="per-shard",
                   help="apply --sample-n per shard or across the whole run "
                        "(per-shard avoids the sample skewing to the largest shard)")
    g.add_argument("--pair-sample-n", type=int, default=2000,
                   help="adjacent pairs to score embeddings for in the mergeability "
                        "check; 0 means all (default: 2000)")

    g = p.add_argument_group("parallelism")
    g.add_argument("--workers", type=int, default=32,
                   help="processes for the parquet passes (default: 32)")
    g.add_argument("--gpu-workers", type=int, default=1,
                   help="processes for the model passes (default: 1)")
    g.add_argument("--devices", default="cuda:0",
                   help="comma-separated torch devices, round-robined over gpu workers "
                        "(default: cuda:0; use 'cpu' to force CPU)")
    g.add_argument("--merge-buckets", type=int, default=64,
                   help="hash buckets used to regroup segments by chunk (default: 64)")

    g = p.add_argument_group("QC-only thresholds")
    g.add_argument("--bak-pass", type=float, default=4.0,
                   help="DNSMOS BAK at or above which background counts as removed "
                        "(default: 4.0)")
    g.add_argument("--bak-warn", type=float, default=3.0,
                   help="DNSMOS BAK below which residual background counts as clear "
                        "(default: 3.0)")
    g.add_argument("--dia-min-reliable", type=float, default=2.0,
                   help="segments shorter than this are flagged as unreliable for "
                        "diarization-based verdicts (default: 2.0)")

    g = p.add_argument_group("output")
    g.add_argument("--report-dir", default=None,
                   help="where to write the report files (default: <output>/_qc_reports)")
    g.add_argument("--work-dir", default=None,
                   help="scratch dir for hash buckets and the resumable GPU cache "
                        "(default: <report-dir>/_work)")
    g.add_argument("--keep-work", action="store_true",
                   help="keep the scratch dir (needed to resume a GPU pass later)")
    g.add_argument("--quiet", action="store_true",
                   help="skip the stdout report; still writes the json/markdown files")

    args = p.parse_args(argv)
    if args.bak_warn > args.bak_pass:
        p.error("--bak-warn must be <= --bak-pass")
    if args.merge_buckets < 1:
        p.error("--merge-buckets must be >= 1")
    return args


def build_config(args: argparse.Namespace) -> QCConfig:
    output_root = os.path.abspath(args.output)
    report_dir = os.path.abspath(args.report_dir or os.path.join(output_root, "_qc_reports"))
    work_dir = os.path.abspath(args.work_dir or os.path.join(report_dir, "_work"))
    return QCConfig(
        output_root=output_root,
        report_dir=report_dir,
        work_dir=work_dir,
        thresholds=Thresholds.from_config(args.config),
        config_path=args.config,
        manifest=os.path.abspath(args.manifest) if args.manifest else None,
        shards=[s.strip() for s in args.shards.split(",") if s.strip()] if args.shards else None,
        steps=_parse_steps(args.steps),
        sample_n=args.sample_n,
        sample_per_shard=(args.sample_scope == "per-shard"),
        pair_sample_n=args.pair_sample_n,
        workers=max(1, args.workers),
        gpu_workers=max(1, args.gpu_workers),
        devices=_parse_devices(args.devices),
        merge_buckets=args.merge_buckets,
        bak_pass=args.bak_pass,
        bak_warn=args.bak_warn,
        dia_min_reliable_s=args.dia_min_reliable,
        keep_work=args.keep_work,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    tag = make_extra_tags(audio_file=cfg.output_root, version="qc")

    if not os.path.isdir(cfg.output_root):
        print(f"output root not found: {cfg.output_root}", file=sys.stderr)
        return 2

    os.makedirs(cfg.report_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Imported here, not at module scope, so `--help` and argument errors do not
    # pay for pyarrow/torch imports.
    from qc.analyzers import run_all
    from qc.report import Report, render

    logger.info(
        f"qc_start output {cfg.output_root} steps {','.join(cfg.steps)} "
        f"sample_n {cfg.sample_n} workers {cfg.workers} gpu_workers {cfg.gpu_workers} "
        f"thresholds_from {cfg.thresholds.source}",
        extra=tag,
    )
    if cfg.thresholds.is_default:
        logger.warning(
            f"qc_thresholds_default reason {cfg.thresholds.fallback_reason} -- "
            "verdicts are measured against built-in defaults, not the values this "
            "data was produced with; pass --config for an accurate report",
            extra=tag,
        )

    started = time.time()
    exit_code = 0
    try:
        sections = run_all(cfg)
        report = Report(
            config=cfg,
            sections=sections,
            elapsed_seconds=time.time() - started,
        )
        paths = render(report, quiet=args.quiet)
        logger.info(
            f"qc_done elapsed_s {report.elapsed_seconds:.1f} reports {' '.join(paths)}",
            extra=tag,
        )
        if not args.quiet:
            print("\nreports written:")
            for path in paths:
                print(f"  {path}")
    except KeyboardInterrupt:
        logger.warning("qc_interrupted", extra=tag)
        print("\ninterrupted", file=sys.stderr)
        exit_code = 130
    except Exception:
        logger.error(f"qc_failed {traceback.format_exc()}", extra=tag)
        print(traceback.format_exc(), file=sys.stderr)
        exit_code = 1
    finally:
        if cfg.keep_work:
            logger.info(f"qc_work_kept {cfg.work_dir}", extra=tag)
        else:
            # Interrupted runs keep the scratch dir regardless: throwing away a
            # half-finished GPU pass would make --keep-work the only way to ever
            # resume, which is a nasty surprise after an hour of scoring.
            if exit_code == 0:
                shutil.rmtree(cfg.work_dir, ignore_errors=True)
            else:
                logger.info(
                    f"qc_work_kept_after_failure {cfg.work_dir} "
                    "(rerun with the same --work-dir to resume)",
                    extra=tag,
                )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
