"""How fast is it going, how long has it been going, and how much longer.

`pipeline_v3/driver.py:_log_progress` already computes throughput and an ETA,
but only *within one driver process*: the numbers reset when the pipeline
restarts, and they are reported per `(stage, shard)`, so they cannot answer
"how far along is the whole job". That is the gap this module fills, using two
independent sources of evidence.

**Elapsed time.** Two different quantities, always reported separately because
they can disagree wildly and each is misleading alone:

  * *pipeline elapsed* -- inferred from the earliest parquet mtime in the tree.
    Slightly low (the first flush happens after startup: the driver flushes on
    100k rows or every 300s, `pipeline_v3/driver.py:61`), and on a resumed run
    it is far too high, because the oldest part is from the previous attempt.
  * *observed window* -- since this monitor's first sample. Always honest about
    what it measured, but says nothing about what happened before it started.

**Rate.** The slope of processed-seconds against wall-clock, from
`history.jsonl`:

  * *recent* -- a trailing window (default 60 min). This is the headline number,
    because the cluster resizes while a job runs (`ActorPool.reconcile`,
    `retire_all` when stage 1's input drains), so a whole-run average reacts far
    too slowly to be actionable.
  * *overall* -- from the first sample to now, as a slower-moving reference.
    Reporting both lets a reader see acceleration or decay directly.

On the very first scan there is no history at all, so the rate falls back to
`done / (now - earliest_mtime)`. It is flagged `rate_is_mtime_estimated` and the
report labels it as approximate, since it inherits the mtime bias above.

**Staleness.** If nothing has been written for `stale_after_secs` (default 15
min, comfortably beyond the driver's 300s flush interval), the job is probably
stopped or wedged. An ETA extrapolated from a rate that has since gone to zero
is actively misleading, so ETAs are suppressed and the report leads with a
warning instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from status.scanner import Snapshot

# Ignore windows shorter than this when differentiating: over a few seconds the
# slope is dominated by scan jitter and flush timing, not by real throughput.
_MIN_WINDOW_SECS = 30.0
# A rate below this is treated as "not moving" rather than yielding an ETA of
# several millennia.
_MIN_RATE = 1e-9


@dataclass
class Estimate:
    """Derived timing. Any field that cannot be computed honestly is None, and
    the renderer prints N/A rather than inventing a number."""

    ts: float

    pipeline_elapsed_secs: Optional[float] = None   # from earliest part mtime
    observed_window_secs: Optional[float] = None    # since this monitor's 1st sample

    rate_recent: Optional[float] = None             # audio-secs processed per wall-sec
    rate_overall: Optional[float] = None
    rate_is_mtime_estimated: bool = False
    rate_window_secs: Optional[float] = None
    rate_samples: int = 0

    eta_recent_secs: Optional[float] = None
    eta_overall_secs: Optional[float] = None

    remaining_secs: Optional[float] = None          # of raw audio still to do
    progress_pct: Optional[float] = None

    stage2_yield_pct: Optional[float] = None        # kept stage-2 vs stage-1 raw processed
    projected_final_kept_secs: Optional[float] = None

    is_stale: bool = False
    stale_for_secs: Optional[float] = None

    @property
    def throughput_recent(self) -> Optional[float]:
        """Same number as `rate_recent`, named as the driver names it: audio
        seconds per wall second, i.e. an "x times realtime" figure."""
        return self.rate_recent

    @property
    def hours_per_day_recent(self) -> Optional[float]:
        if self.rate_recent is None:
            return None
        return self.rate_recent * 24.0


def _slope(samples: Sequence[Dict[str, Any]], field: str) -> Optional[float]:
    """Rate of change of `field` between the first and last sample.

    An endpoint difference, not a least-squares fit: the series is a
    monotonically increasing cumulative total, so the endpoint slope IS the mean
    rate over the window, and it cannot be skewed by a single outlier sample the
    way a fit can.
    """
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]
    span = float(last["ts"]) - float(first["ts"])
    if span < _MIN_WINDOW_SECS:
        return None
    delta = float(last.get(field, 0.0) or 0.0) - float(first.get(field, 0.0) or 0.0)
    if delta < 0:
        # The output tree shrank (a shard was deleted, or the job was restarted
        # against a fresh --output). A negative rate is meaningless; report none.
        return None
    return delta / span


def _eta(remaining: Optional[float], rate: Optional[float]) -> Optional[float]:
    if remaining is None or rate is None or rate <= _MIN_RATE:
        return None
    return max(0.0, remaining) / rate


def estimate(snap: Snapshot, history: List[Dict[str, Any]],
             window_secs: float = 3600.0,
             stale_after_secs: float = 900.0,
             now: Optional[float] = None) -> Estimate:
    """Combine this scan with the recorded history into timing figures.

    `history` must NOT yet include `snap`; the caller appends the new sample
    after rendering, so a scan is never differentiated against itself.
    """
    now = now if now is not None else time.time()
    est = Estimate(ts=now)
    est.progress_pct = snap.progress_pct

    # -- elapsed -----------------------------------------------------------
    if snap.min_part_mtime is not None:
        est.pipeline_elapsed_secs = max(0.0, now - snap.min_part_mtime)
    if history:
        est.observed_window_secs = max(0.0, now - float(history[0]["ts"]))

    # -- staleness ---------------------------------------------------------
    if snap.max_part_mtime is not None:
        idle = now - snap.max_part_mtime
        if idle > stale_after_secs:
            est.is_stale = True
            est.stale_for_secs = idle

    # -- rate --------------------------------------------------------------
    # `snap` is appended to the series only for the purposes of this
    # calculation, so the newest scan participates in the slope without being
    # persisted twice.
    series = list(history) + [snap.history_sample()]
    est.rate_samples = len(series)

    recent = [s for s in series if float(s["ts"]) >= now - window_secs]
    if len(recent) >= 2:
        est.rate_recent = _slope(recent, "stage1_done_secs")
        est.rate_window_secs = float(recent[-1]["ts"]) - float(recent[0]["ts"])
    est.rate_overall = _slope(series, "stage1_done_secs")

    if est.rate_recent is None and est.rate_overall is not None:
        # Window too short to differentiate yet (monitor just started); the
        # whole-series slope is the best available estimate of "recent".
        est.rate_recent = est.rate_overall
        est.rate_window_secs = est.observed_window_secs

    if est.rate_recent is None and est.pipeline_elapsed_secs:
        # Cold start: no usable history, so infer a rate from how much has been
        # done since the earliest flush. Flagged, because it inherits the mtime
        # bias (low on a fresh run, high on a resumed one).
        if snap.stage1_done_secs > 0 and est.pipeline_elapsed_secs > _MIN_WINDOW_SECS:
            est.rate_recent = snap.stage1_done_secs / est.pipeline_elapsed_secs
            est.rate_is_mtime_estimated = True

    # -- remaining and ETA -------------------------------------------------
    if snap.manifest_available and snap.total_secs > 0:
        est.remaining_secs = max(0.0, snap.total_secs - snap.stage1_done_secs)

    if not est.is_stale:
        est.eta_recent_secs = _eta(est.remaining_secs, est.rate_recent)
        est.eta_overall_secs = _eta(est.remaining_secs, est.rate_overall)

    # -- yield extrapolation ----------------------------------------------
    # Stage 2's kept seconds as a fraction of the RAW audio stage 1 has chewed
    # through -- the end-to-end "useful data rate". Extrapolating it over the
    # whole corpus answers "how much trainable audio will this job produce".
    #
    # Only meaningful once stage 2 has actually produced something: while
    # stage 2 lags behind stage 1 (there is no back-pressure on the hand-off
    # queue, `pipeline_v3/driver.py` docstring) this ratio starts near zero and
    # climbs, so early values understate the final yield.
    if snap.stage1_done_secs > 0 and snap.stage2_kept_secs > 0:
        est.stage2_yield_pct = 100.0 * snap.stage2_kept_secs / snap.stage1_done_secs
        if snap.manifest_available and snap.total_secs > 0:
            est.projected_final_kept_secs = (
                snap.total_secs * snap.stage2_kept_secs / snap.stage1_done_secs
            )
    return est
