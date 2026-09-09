"""SlowActorDetector: per-stage relative-throughput watchdog for the
pipeline_v3 driver.

Why this exists: an elastic Ray cluster can contain a machine that is *not*
broken -- its actor stays alive, returns correct results and never raises --
but runs an order of magnitude slower than its peers because the machine
itself is resource-starved (memory pressure/swap, cgroup CPU throttling, a
noisy neighbour, or a slow shared filesystem). Observed in production: two
tasks on the same GPU model and the same config, one doing 39h of audio while
the other did 5.3h, with `normalize` -- a pure-numpy O(n) pass -- 335x slower
on the sick node. The GPU there idles at ~3%, and the platform eventually
reclaims the "low load" task, destroying the evidence. This module makes the
cluster say out loud WHICH machine is the outlier, early enough to log in and
look, so the log line carries the node IP / hostname / pid, not just an
opaque actor id.

Metric: per-file RTF over a sliding window of that actor's most recent
completions,

    actor_rtf = sum(item.duration) / sum(t_done - t_submit)

i.e. source audio seconds processed per wall second of actual processing.
Deliberately NOT `_Progress.audio_secs / elapsed`: `_Progress` is keyed by
(stage, shard) and cannot attribute work to an actor, and its wall clock
includes time the actor spent idle for lack of queued work -- which would
report "has nothing to do" as "is slow". Submit->return spans only real
processing.

Comparability: every actor of one stage runs with the same `max_concurrency`
(one shared `defaults` block per stage), so this is "RTF per concurrency
slot" and actors of the same stage compare directly. Across stages it does
NOT compare (stage_2's `duration` is summed speech seconds, a different
magnitude), so medians are computed strictly per stage.

Two independent verdicts:
  * relative -- rtf * SLOW_RATIO < stage median, for SLOW_STRIKES consecutive
    checks. Needs peers, so it is skipped while the pool is too small.
  * stalled  -- files in flight but nothing completed for STALL_SECONDS. This
    is the one that catches the worst case: a node so slow that it finishes
    no file at all within a check period therefore contributes no window
    samples and cannot take part in the median comparison. It also works with
    a single actor, where a median is meaningless.

Log-only by design: nothing here touches `actor.draining` or the scheduler.
Draining a sick actor automatically would take the evidence offline, which is
the opposite of what this is for. Every public method swallows its own
exceptions -- monitoring must never be able to break the pipeline.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

import logger

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps runtime ray-free
    import ray

    from pipeline_v3.pool import Actor, ActorPool

__all__ = ["HEALTH_INTERVAL", "SlowActorDetector"]


def _env_num(name: str, default: float) -> float:
    """Read a positive numeric override from the environment.

    Thresholds are tunable per run without a code change or a config-schema
    change: the yaml `defaults` block is parsed into a frozen dataclass
    (pipeline_v3.config.Defaults) with strict keys, so adding knobs there
    would ripple through the config layer for something that is pure
    operational tuning. Bad/zero/negative values fall back to the default
    rather than disabling detection silently."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"ray_v3_health_bad_env {name}={raw!r} ignored, using {default}")
        return default
    if value <= 0:
        logger.warning(f"ray_v3_health_bad_env {name}={raw!r} not positive, using {default}")
        return default
    return value


HEALTH_INTERVAL = _env_num("AP_HEALTH_INTERVAL", 120.0)   # seconds between checks
RTF_WINDOW = int(_env_num("AP_RTF_WINDOW", 8))            # completions kept per actor
MIN_SAMPLES = int(_env_num("AP_MIN_SAMPLES", 2))          # completions before an actor is judged
MIN_PEERS = int(_env_num("AP_MIN_PEERS", 3))              # comparable actors before a median is trusted
SLOW_RATIO = _env_num("AP_SLOW_RATIO", 10.0)              # "an order of magnitude" below the median
SLOW_STRIKES = int(_env_num("AP_SLOW_STRIKES", 2))        # consecutive slow checks before alerting
ALERT_COOLDOWN = _env_num("AP_ALERT_COOLDOWN", 600.0)     # seconds between repeats of the same alert
STALL_SECONDS = _env_num("AP_STALL_SECONDS", 1800.0)      # in-flight but zero completions for this long
OVERVIEW_ACTORS = int(_env_num("AP_OVERVIEW_ACTORS", 8))  # slowest N actors listed in the overview line

# Appended to both alerts: what to actually run once you know the machine.
_HINT = ("check the node: free -g; vmstat 1 5; "
         "cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled; nvidia-smi")


@dataclass
class _ActorHealth:
    """One actor's rolling throughput record. Keyed by `Actor.uid` in
    SlowActorDetector, never by the Actor object itself (a mutable dataclass
    is unhashable)."""
    stage_key: str
    # (audio_secs, wall_secs) of recent completions; bounded, so memory is
    # flat for the whole run.
    window: deque = field(default_factory=lambda: deque(maxlen=RTF_WINDOW))
    files_done: int = 0
    audio_secs: float = 0.0
    # Seeded with the actor's FIRST submission so an actor that has never been
    # given work can never look "stalled"; afterwards, the last completion.
    last_done_at: float = 0.0
    strikes: int = 0
    last_slow_alert_at: float = 0.0
    last_stall_alert_at: float = 0.0

    def rtf(self) -> float:
        """Windowed source-audio seconds per wall second of processing."""
        wall = 0.0
        audio = 0.0
        for a, w in self.window:
            audio += a
            wall += w
        return (audio / wall) if wall > 0 else 0.0

    def comparable(self) -> bool:
        return len(self.window) >= MIN_SAMPLES and self.rtf() > 0


class SlowActorDetector:
    """Tracks every actor's throughput from the driver side only: submit
    timestamps plus `FileItem.duration`, no actor-side cooperation and no
    change to the FileResult contract."""

    def __init__(self, stage_keys: list[str]) -> None:
        self._stage_keys = list(stage_keys)
        self._health: dict[str, _ActorHealth] = {}          # Actor.uid -> record
        # in-flight submissions: ref -> (actor uid, audio secs, submitted at).
        # Bounded by (actors * max_concurrency); every exit path (done, task
        # error, actor crash) removes its entry, and sweep() collects anything
        # orphaned by an actor disappearing.
        self._pending: dict["ray.ObjectRef", tuple[str, float, float]] = {}

    # ------------------------------------------------------------------
    # instrumentation hooks (called from the driver's hot path: O(1) each)
    # ------------------------------------------------------------------
    def on_submit(self, stage_key: str, actor: "Actor", ref: "ray.ObjectRef",
                  duration: float) -> None:
        try:
            now = time.time()
            health = self._health.get(actor.uid)
            if health is None:
                health = _ActorHealth(stage_key=stage_key, last_done_at=now)
                self._health[actor.uid] = health
            self._pending[ref] = (actor.uid, duration, now)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_health_error on_submit {type(e).__name__}: {e}")

    def on_done(self, ref: "ray.ObjectRef", success: bool = True) -> None:
        """Record a returned file. A FAILED file still proves the actor is
        alive and moving (so it clears the stall clock) but is kept out of the
        RTF window: a file that fails fast -- e.g. rejected by a duration
        filter -- would otherwise look like enormous throughput."""
        try:
            entry = self._pending.pop(ref, None)
            if entry is None:
                return
            uid, duration, t_submit = entry
            health = self._health.get(uid)
            if health is None:
                return
            now = time.time()
            wall = now - t_submit
            health.files_done += 1
            health.last_done_at = now
            if not success:
                return
            health.audio_secs += duration
            # duration <= 0 happens in --input mode (no manifest durations):
            # it carries no throughput signal, so keep it out of the window
            # (the stall verdict still covers those runs).
            if duration > 0 and wall > 0:
                health.window.append((duration, wall))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_health_error on_done {type(e).__name__}: {e}")

    def on_drop(self, ref: "ray.ObjectRef") -> None:
        """Forget a submission that will never produce a timing sample
        (task error, or an actor crash taking its in-flight files down)."""
        try:
            self._pending.pop(ref, None)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_health_error on_drop {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # periodic work
    # ------------------------------------------------------------------
    def sweep(self, pools: dict[str, "ActorPool"]) -> None:
        """Drop state for actors that have left their pool (reconcile shrink,
        recycle, crash, retire_all). Keeps memory flat and -- more
        importantly -- keeps dead actors out of the median."""
        try:
            live = {a.uid for pool in pools.values() for a in pool.actors}
            for uid in [u for u in self._health if u not in live]:
                del self._health[uid]
            for ref in [r for r, (uid, _, _) in self._pending.items() if uid not in live]:
                del self._pending[ref]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_health_error sweep {type(e).__name__}: {e}")

    def check(self, pools: dict[str, "ActorPool"]) -> None:
        """One detection pass over every stage. Emits at most one overview
        line per stage plus (rate-limited) one alert per offending actor."""
        try:
            now = time.time()
            for key in self._stage_keys:
                pool = pools.get(key)
                if pool is None:
                    continue
                pairs = [
                    (actor, self._health[actor.uid])
                    for actor in pool.actors
                    if actor.uid in self._health
                ]
                if not pairs:
                    continue
                rtfs = [h.rtf() for _, h in pairs if h.comparable()]
                med = median(rtfs) if len(rtfs) >= MIN_PEERS else 0.0
                self._log_overview(key, pairs, med)
                for actor, health in pairs:
                    self._check_relative(key, actor, health, med, now)
                    self._check_stalled(key, actor, health, now)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ray_v3_health_error check {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # verdicts
    # ------------------------------------------------------------------
    def _check_relative(self, stage_key: str, actor: "Actor", health: _ActorHealth,
                        med: float, now: float) -> None:
        """Slow relative to this stage's median. Known limitation: if the
        WHOLE cluster degrades, the median sinks with it and this verdict goes
        quiet -- that case is covered by _check_stalled."""
        if med <= 0 or not health.comparable():
            return  # no trustworthy peer group yet; leave strikes untouched
        rtf = health.rtf()
        if rtf * SLOW_RATIO >= med:
            health.strikes = 0
            return
        health.strikes += 1
        if health.strikes < SLOW_STRIKES:
            return
        if now - health.last_slow_alert_at < ALERT_COOLDOWN:
            return
        health.last_slow_alert_at = now
        logger.error(
            f"ray_v3_slow_actor stage {stage_key} {actor.describe()} "
            f"rtf {rtf:.3f}x median {med:.3f}x ratio {med / rtf:.1f}x_below "
            f"samples {len(health.window)} files {health.files_done} "
            f"audio {health.audio_secs / 3600.0:.2f}h inflight {len(actor.inflight)} "
            f"age {int(now - actor.born_at)}s strikes {health.strikes} -- {_HINT}"
        )

    def _check_stalled(self, stage_key: str, actor: "Actor", health: _ActorHealth,
                       now: float) -> None:
        """Has work in flight but has completed nothing for a long time --
        the signature of a node so starved it cannot finish a single file
        within a check period, which is exactly the case the median-based
        verdict cannot see."""
        if not actor.inflight:
            return
        idle = now - health.last_done_at
        if health.last_done_at <= 0 or idle < STALL_SECONDS:
            return
        if now - health.last_stall_alert_at < ALERT_COOLDOWN:
            return
        health.last_stall_alert_at = now
        oldest, name = self._oldest_inflight(actor, now)
        logger.error(
            f"ray_v3_actor_stalled stage {stage_key} {actor.describe()} "
            f"no_completion_for {idle:.0f}s inflight {len(actor.inflight)} "
            f"oldest_inflight {oldest:.0f}s file {name} "
            f"files {health.files_done} age {int(now - actor.born_at)}s -- {_HINT}"
        )

    def _oldest_inflight(self, actor: "Actor", now: float) -> tuple[float, str]:
        """Longest-running in-flight file on this actor (elapsed seconds, its
        relative path) -- names the file to reproduce with."""
        oldest = 0.0
        name = "?"
        for ref, item in list(actor.inflight.items()):
            entry = self._pending.get(ref)
            if entry is None:
                continue
            elapsed = now - entry[2]
            if elapsed > oldest:
                oldest = elapsed
                name = getattr(item, "relative_path", "?")
        return oldest, name

    # ------------------------------------------------------------------
    @staticmethod
    def _log_overview(stage_key: str, pairs: list[tuple["Actor", _ActorHealth]],
                      med: float) -> None:
        """The line to watch continuously: this stage's throughput spread,
        slowest first. One line per stage per check period."""
        ranked = sorted(pairs, key=lambda p: p[1].rtf())
        shown = ranked[:OVERVIEW_ACTORS]
        parts = [
            f"{actor.describe()} rtf {h.rtf():.3f}x files {h.files_done} "
            f"inflight {len(actor.inflight)}"
            for actor, h in shown
        ]
        more = "" if len(ranked) <= len(shown) else f" (+{len(ranked) - len(shown)} more)"
        med_txt = f"{med:.3f}x" if med > 0 else "n/a"
        logger.info(
            f"ray_v3_actor_rtf stage {stage_key} actors {len(pairs)} median {med_txt}"
            f"{more} | " + " | ".join(parts)
        )
