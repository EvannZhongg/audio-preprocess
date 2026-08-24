"""Rendering, and the WeCom push.

Two renderings of the same `Snapshot` + `Estimate`, so they can never disagree:

  * `render_report` -- the full multi-section block written to `status.log` and
    stdout every cycle.
  * `render_wecom` -- a condensed version for the group chat, kept under
    WeCom's 2048-byte limit for a text message.

Anything the estimator could not compute honestly renders as `N/A` rather than
a zero, because a zero here reads as a real measurement.

`utils.msg_bot.send_msg` is imported lazily and wrapped: it has no timeout, no
exception handling and does not check the response code, so calling it naively
from a resident loop would let one unreachable webhook take the monitor down.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from qc.loaders import DROP_COLUMNS
from status.estimator import Estimate
from status.scanner import Snapshot

logger = logging.getLogger("status.reporter")

# WeCom rejects a text message whose content exceeds 2048 bytes.
WECOM_MAX_BYTES = 2048
_SEP = "=" * 72

# How many trailing path components identify an output tree in a report.
# `/root/jfs/itachi/huyuan_ko/out/5/` -> `huyuan_ko/out/5`: the leading mount
# path is identical for every job and only costs width, while the last three
# components are what actually distinguish concurrent runs.
_PATH_TAIL_PARTS = 3


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def short_path(path: Optional[str], parts: int = _PATH_TAIL_PARTS) -> str:
    if not path:
        return "N/A"
    segments = [s for s in str(path).replace("\\", "/").split("/") if s]
    if not segments:
        return "/"
    return "/".join(segments[-parts:])


def _hours(secs: Optional[float]) -> str:
    if secs is None:
        return "N/A"
    return f"{secs / 3600.0:,.2f}h"


def _duration(secs: Optional[float]) -> str:
    """Human-readable span, e.g. `1d 19h 36m`. Coarse on purpose: these are
    estimates, and second-level precision would imply accuracy that is not
    there."""
    if secs is None:
        return "N/A"
    secs = max(0.0, float(secs))
    days, rem = divmod(int(secs), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _rate(value: Optional[float]) -> str:
    """A rate as both a realtime multiple and audio-hours per day -- the same
    two framings `pipeline_v3/driver.py:_log_progress` uses."""
    if value is None:
        return "N/A"
    return f"{value:.2f}x ({value * 24.0:,.0f}h/day)"


def _ts(epoch: Optional[float]) -> str:
    if epoch is None:
        return "N/A"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _bar(pct: Optional[float], width: int = 30) -> str:
    if pct is None:
        return "[" + "?" * width + "]"
    filled = int(round(max(0.0, min(100.0, pct)) / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# headline block
# ---------------------------------------------------------------------------

def render_summary(snap: Snapshot, est: Estimate) -> str:
    """The six lines someone actually acts on.

    Rendered identically into `status.log` and into the WeCom message, so the
    chat and the log can never quote different numbers for the same scan.
    """
    return "\n".join([
        f"进度: {_pct(snap.progress_pct)}"
        f"  ({_hours(snap.stage1_done_secs)} / {_hours(snap.total_secs)})",
        f"stage2 有效音频: {_hours(snap.stage2_kept_secs)}"
        f" ， 有效率: {_pct(est.stage2_yield_pct)}",
        "",
        f"已运行: {_duration(est.pipeline_elapsed_secs)}",
        f"预计剩余: {_duration(est.eta_recent_secs)}",
        f"预计最终有效音频: {_hours(est.projected_final_kept_secs)}",
    ])


def _warnings(snap: Snapshot, est: Estimate) -> list[str]:
    """Conditions that make the numbers above untrustworthy. Rendered before
    them, because a reader who stops at the first line must still see that the
    pipeline has died."""
    lines: list[str] = []
    if est.is_stale:
        lines.append(
            f"⚠️ 已停止写入 {_duration(est.stale_for_secs)}，pipeline 可能已停！"
            "（此时不再给出预计剩余时间）"
        )
    if not snap.manifest_available:
        lines.append("⚠️ 未提供 manifest，进度与预计剩余不可用")
    elif snap.stage1_done_files and not snap.stage1_matched_files:
        # A real failure mode: if stage 1's `source` values do not match the
        # manifest's `relative_path` values, progress sits at 0% forever.
        lines.append("⚠️ stage1 已完成文件在 manifest 中一个都匹配不上，进度无意义")
    if snap.parts_failed:
        lines.append(f"⚠️ 有 {snap.parts_failed:,} 个 parquet 无法读取")
    return lines


# ---------------------------------------------------------------------------
# full report
# ---------------------------------------------------------------------------

def render_report(snap: Snapshot, est: Estimate) -> str:
    lines: list[str] = [
        _SEP,
        f"{short_path(snap.output_root)}    @ {_ts(snap.ts)}",
        _SEP,
    ]
    lines += _warnings(snap, est)
    lines += ["", render_summary(snap, est), ""]

    lines += [
        "-- PROGRESS " + "-" * 60,
        f"  {_bar(snap.progress_pct)}  {_pct(snap.progress_pct)}",
        f"  output               : {snap.output_root}",
        f"  raw audio total      : {_hours(snap.total_secs)}"
        f"   ({snap.total_files:,} files)",
        f"  stage1 processed     : {_hours(snap.stage1_done_secs)}"
        f"   ({snap.stage1_done_files:,} files)",
        f"  remaining            : {_hours(est.remaining_secs)}",
    ]
    if snap.files_unknown_duration:
        lines.append(
            f"  note                 : {snap.files_unknown_duration:,} manifest "
            "files have duration=0 (probe failed); they add no seconds to the total"
        )

    lines += [
        "",
        "-- STAGE 1 (segmentation + audio metrics) " + "-" * 30,
        f"  files done           : {snap.stage1_done_files:,}"
        f"   (failed {snap.stage1_failed_files:,})",
        f"  valid segments       : {snap.stage1_valid_segments:,}"
        f"   totalling {_hours(snap.stage1_valid_secs)}",
    ]
    if snap.stage1_error_types:
        lines.append("  failure types        :")
        for key, count in snap.stage1_error_types.items():
            lines.append(f"      {count:>8,}  {key}")

    kept_rate = (
        100.0 * snap.stage2_kept_secs / snap.stage2_total_secs
        if snap.stage2_total_secs > 0 else None
    )
    lines += [
        "",
        "-- STAGE 2 (ASR + text/quality filtering) " + "-" * 30,
        f"  VALID OUTPUT AUDIO   : {_hours(snap.stage2_kept_secs)}"
        f"   ({snap.stage2_kept_rows:,} segments)",
        f"  stage2 seen          : {_hours(snap.stage2_total_secs)}"
        f"   ({snap.stage2_total_rows:,} segments)",
        f"  keep rate (by dur)   : {_pct(kept_rate)}",
        f"  end-to-end yield     : {_pct(est.stage2_yield_pct)}"
        "   (valid output / raw audio processed)",
    ]
    if snap.stage2_total_rows:
        lines.append("  dropped by (reasons are not mutually exclusive):")
        for column in DROP_COLUMNS:
            rows = snap.stage2_drop_rows.get(column, 0)
            secs = snap.stage2_drop_secs.get(column, 0.0)
            lines.append(
                f"      {column:<28} {rows:>10,} segs  {_hours(secs):>12}"
            )
    if snap.stage2_error_rows:
        lines.append(
            f"  error rows           : {snap.stage2_error_rows:,}"
            f"   (of which retriable ASR failures: "
            f"{snap.stage2_retriable_error_rows:,}, reprocessed next run)"
        )

    # Label the recent-rate row with the window it was actually measured over,
    # so a reader can tell a 5-minute slope from an hour-long one.
    if est.rate_is_mtime_estimated:
        rate_label = "  rate (mtime est.)    "
        rate_note = "   [rough: inferred from file mtimes, no history yet]"
    elif est.rate_window_secs:
        rate_label = f"  rate (last {_duration(est.rate_window_secs):<9})"
        rate_note = ""
    else:
        rate_label = "  rate (recent)        "
        rate_note = ""

    lines += [
        "",
        "-- RATE & ETA " + "-" * 58,
        f"  pipeline running for : {_duration(est.pipeline_elapsed_secs)}"
        "   (estimated from earliest parquet mtime)",
        f"  monitor observing for: {_duration(est.observed_window_secs)}",
        f"{rate_label}: {_rate(est.rate_recent)}{rate_note}",
        f"  rate (whole run)     : {_rate(est.rate_overall)}",
        f"  ETA (recent rate)    : {_duration(est.eta_recent_secs)}",
        f"  ETA (whole-run rate) : {_duration(est.eta_overall_secs)}",
        f"  projected final valid: {_hours(est.projected_final_kept_secs)}",
    ]
    if est.projected_final_kept_secs is not None:
        lines.append(
            "  note                 : projection assumes the current yield holds; "
            "it understates while stage 2 still lags stage 1"
        )

    lines += [
        "",
        "-- SCAN " + "-" * 64,
        f"  shards               : {snap.shards_total:,}"
        f"   (stage1 {snap.shards_with_stage1:,}, stage2 {snap.shards_with_stage2:,})",
        f"  parquet parts        : {snap.parts_total:,}"
        f"   (read {snap.parts_read:,}, cached {snap.parts_cached:,},"
        f" unreadable {snap.parts_failed:,})",
        f"  newest part written  : {_ts(snap.max_part_mtime)}",
        f"  scan took            : {snap.scan_secs:.1f}s",
    ]
    if snap.failed_part_samples:
        lines.append("  unreadable parts (first few):")
        for sample in snap.failed_part_samples:
            lines.append(f"      {sample}")
        if snap.parts_failed > len(snap.failed_part_samples):
            lines.append(
                f"      ... and {snap.parts_failed - len(snap.failed_part_samples):,} more"
            )

    lines.append(_SEP)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WeCom message
# ---------------------------------------------------------------------------

def render_wecom(snap: Snapshot, est: Estimate, title: str = "TTS 数据处理进度") -> str:
    """Condensed report for the group chat.

    Header is the titled timestamp then the output tree, so a glance at the
    chat list shows what the message is; after that comes the exact same
    `render_summary` block the log carries. Deliberately shorter than the log
    report: a chat message that needs scrolling does not get read.
    """
    lines: list[str] = [
        f"[{title}] {_ts(snap.ts)}",
        f"文件: {short_path(snap.output_root)}",
    ]
    lines += _warnings(snap, est)
    lines += ["", render_summary(snap, est)]
    return _truncate(("\n".join(lines)).strip(), WECOM_MAX_BYTES)


def _truncate(text: str, limit: int) -> str:
    """Trim to `limit` BYTES (WeCom's limit is on bytes, and this text is mostly
    multi-byte Chinese), without splitting a character in half."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    marker = "\n...(truncated)"
    budget = limit - len(marker.encode("utf-8"))
    return raw[:budget].decode("utf-8", errors="ignore") + marker


def push_wecom(message: str) -> bool:
    """Send to the WeCom group, swallowing every failure.

    `utils.msg_bot.send_msg` is imported here rather than at module scope so
    that `requests` is only required when a push actually happens -- the
    statistics path stays importable on a box without it.

    Returns True when the send did not raise. `send_msg` ignores the HTTP
    response, so this is "handed off without error", not "delivered".
    """
    try:
        from utils.msg_bot import send_msg
    except Exception as exc:  # noqa: BLE001 - missing requests, bad module, ...
        logger.error("wecom push skipped: cannot import utils.msg_bot: %s", exc)
        return False
    try:
        send_msg(message)
        logger.info("wecom push sent (%d bytes)", len(message.encode("utf-8")))
        return True
    except Exception as exc:  # noqa: BLE001 - a dead webhook must not kill the loop
        logger.error("wecom push failed: %s: %s", type(exc).__name__, exc)
        return False
