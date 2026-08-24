"""Resident progress monitor for a pipeline_v3 run.

    # typical: scan every 5 minutes, push to WeCom hourly
    nohup python status/monitor.py \
        --output   /path/to/pipeline_v3/output \
        --manifest /path/to/manifest \
        > /dev/null 2>&1 &

    # one-shot, for eyeballing the current numbers
    python status/monitor.py --output ... --manifest ... --once

Reports go to `status/status.log` (rotated) and to stdout. See status/README.md
for what each number means and where its definition comes from.

The loop is deliberately hard to kill: a scan that raises -- an unreadable
parquet, a vanished mount, a webhook timeout -- is logged and the loop sleeps
and tries again. The only things that stop it are SIGINT/SIGTERM.
"""
from __future__ import annotations

import os
import sys

# Make the project root importable before any project package is touched, so
# that `python status/monitor.py` works from any cwd, as does
# `python -m status.monitor`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import logging
import logging.handlers
import signal
import threading
import time

from status.estimator import estimate
from status.reporter import push_wecom, render_report, render_wecom
from status.scanner import OutputScanner
from status.state import StatusState

STATUS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(STATUS_DIR, "status.log")

logger = logging.getLogger("status")

# Set when SIGINT/SIGTERM arrives; the sleep waits on it so shutdown is prompt
# instead of taking up to a full interval.
_stop = threading.Event()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="status/monitor.py",
        description="Resident progress monitor for a pipeline_v3 run "
                    "(read-only; never writes into --output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True,
        help="the --output folder main_v3_ray.py was launched with; scanned for "
             "<shard>/segments_part-*.parquet and stage2_segments_part-*.parquet",
    )
    parser.add_argument(
        "--manifest",
        help="manifest from build_manifest.py (a shard directory or a single "
             "parquet). Without it, total/progress/ETA are unavailable -- the "
             "raw-audio duration of each file only exists in the manifest.",
    )
    parser.add_argument(
        "--interval", type=float, default=300.0,
        help="seconds between scans",
    )
    parser.add_argument(
        "--push-interval", type=float, default=3600.0,
        help="minimum seconds between WeCom pushes (0 disables pushing)",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=60.0,
        help="trailing window used for the headline rate; shorter reacts faster "
             "to cluster resizes, longer is smoother",
    )
    parser.add_argument(
        "--stale-minutes", type=float, default=15.0,
        help="treat the pipeline as stopped when no parquet has been written "
             "for this long (must exceed the driver's 300s flush interval)",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="threads used to read parquet parts",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="scan once, print the report, exit (no WeCom push unless --push-now)",
    )
    parser.add_argument(
        "--push-now", action="store_true",
        help="push to WeCom on the first scan instead of waiting out "
             "--push-interval; useful with --once to test the webhook",
    )
    parser.add_argument(
        "--log-file", default=DEFAULT_LOG,
        help="report log path",
    )
    parser.add_argument(
        "--log-max-bytes", type=int, default=32 * 1024 * 1024,
        help="rotate the log once it exceeds this size",
    )
    parser.add_argument(
        "--log-backups", type=int, default=5,
        help="how many rotated logs to keep",
    )
    parser.add_argument(
        "--state-dir", default=STATUS_DIR,
        help="where history.jsonl / scan_cache.json / state.json are kept",
    )
    return parser.parse_args(argv)


def setup_logging(log_file: str, max_bytes: int, backups: int) -> None:
    """Plain-text rotating log plus stdout.

    Uses the stdlib rather than the project's `logger/` package on purpose: that
    one emits single-line JSON for machine collection and applies global
    configuration, whereas this report is written to be read by a person and
    must not fight with the pipeline's own logging setup.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
    root = logging.getLogger("status")
    root.setLevel(logging.INFO)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)


def _install_signal_handlers() -> None:
    def handle(signum, _frame):
        logger.info("received signal %s, shutting down after this cycle", signum)
        _stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle)
        except (ValueError, OSError):
            # Not the main thread, or the platform lacks the signal; the loop
            # still works, it just cannot be stopped gracefully.
            pass


def run_cycle(scanner: OutputScanner, state: StatusState, args: argparse.Namespace,
              push_enabled: bool) -> None:
    """One scan -> report -> maybe push -> persist. Raises only on bugs; every
    expected failure is handled deeper down."""
    snapshot = scanner.scan()

    # History is read BEFORE the new sample is appended, so the estimator never
    # differentiates this scan against itself.
    history = state.read_history()
    est = estimate(
        snapshot, history,
        window_secs=args.window_minutes * 60.0,
        stale_after_secs=args.stale_minutes * 60.0,
    )

    for line in render_report(snapshot, est).splitlines():
        logger.info(line)

    persisted = state.load_state()
    if push_enabled:
        last_push = float(persisted.get("last_push_ts") or 0.0)
        # Time-based, not counter-based: a restart or a slow cycle must not
        # shift the hourly cadence, nor trigger an immediate duplicate push.
        due = args.push_now or (snapshot.ts - last_push >= args.push_interval)
        if due and push_wecom(render_wecom(snapshot, est)):
            persisted["last_push_ts"] = snapshot.ts

    state.append_history(snapshot.history_sample())
    persisted.setdefault("first_seen_ts", snapshot.ts)
    persisted["last_scan_ts"] = snapshot.ts
    state.save_state(persisted)
    scanner.persist_cache()


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.log_max_bytes, args.log_backups)

    if not os.path.isdir(args.output):
        logger.error("--output is not a directory: %s", args.output)
        return 2
    if args.manifest and not os.path.exists(args.manifest):
        logger.error("--manifest does not exist: %s", args.manifest)
        return 2
    if args.stale_minutes * 60.0 <= args.interval:
        logger.warning(
            "--stale-minutes (%.1fm) is not longer than --interval (%.0fs); "
            "the pipeline may be reported as stalled spuriously",
            args.stale_minutes, args.interval,
        )

    _install_signal_handlers()
    state = StatusState(args.state_dir)
    scanner = OutputScanner(args.output, args.manifest, args.workers, state)
    push_enabled = args.push_interval > 0 and (not args.once or args.push_now)

    logger.info(
        "status monitor starting: output=%s manifest=%s interval=%.0fs "
        "push=%s window=%.0fm stale=%.0fm",
        args.output, args.manifest or "<none>", args.interval,
        f"{args.push_interval:.0f}s" if push_enabled else "disabled",
        args.window_minutes, args.stale_minutes,
    )

    first = True
    while not _stop.is_set():
        started = time.time()
        try:
            run_cycle(scanner, state, args, push_enabled)
        except Exception:  # noqa: BLE001 - a resident monitor must outlive its bugs
            logger.exception("scan cycle failed; retrying next interval")

        if args.once:
            break
        if first:
            # --push-now is a one-off nudge for the first cycle only; after that
            # the normal cadence applies.
            args.push_now = False
            first = False
        # Subtract the work already done so the cadence is the interval, not
        # interval + scan time.
        remaining = args.interval - (time.time() - started)
        if remaining > 0:
            _stop.wait(remaining)

    logger.info("status monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
