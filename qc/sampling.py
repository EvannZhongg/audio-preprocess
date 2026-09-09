"""Deterministic sampling for the GPU re-check passes.

Requirements that ruled out the obvious approaches:

  * `random.sample` needs the population up front -- QC streams parquet parts
    and must not materialise millions of ids to pick 2000.
  * reservoir sampling is streaming but not reproducible across runs, so two
    QC runs on the same tree would disagree and you could not tell a real
    quality change from sampling noise.

So: hash every candidate with SHA-1 and keep the N smallest hashes. That is a
uniform random sample (the hash is a pseudo-random permutation), it streams in
O(N) memory, it is byte-identical across runs and machines, and -- usefully --
the sample for N is always a subset of the sample for N+k, so raising the
sample size reuses every cached verdict instead of invalidating it.
"""
from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional


def stable_key(value: str) -> int:
    """A stable pseudo-random 48-bit key for a string id.

    SHA-1 rather than `hash()` because Python's string hash is salted per
    process (PYTHONHASHSEED), which would make sampling non-reproducible.
    """
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class SampleInfo:
    """How a sample relates to its population, for reporting confidence."""

    population: int = 0
    sampled: int = 0
    mode: str = "full"

    @property
    def rate(self) -> float:
        return (self.sampled / self.population) if self.population else 0.0

    def merge(self, other: "SampleInfo") -> None:
        self.population += other.population
        self.sampled += other.sampled
        if other.mode != "full":
            self.mode = other.mode

    def to_dict(self) -> dict:
        return {
            "population": self.population,
            "sampled": self.sampled,
            "rate": round(self.rate, 6),
            "mode": self.mode,
        }


class SmallestNSampler:
    """Keeps the `n` items with the smallest stable keys seen so far.

    A max-heap of size n (negated keys), so each candidate costs O(log n) and
    memory never exceeds n regardless of population size. `n <= 0` disables
    sampling entirely and keeps everything.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.population = 0
        self._heap: list[tuple[int, int, Any]] = []
        self._all: list[Any] = []
        # Tie-breaker so heap comparison never falls through to the payload
        # (payloads are dataclasses/dicts and may not be orderable).
        self._counter = 0

    @property
    def unlimited(self) -> bool:
        return self.n <= 0

    def offer(self, key_source: str, payload: Any) -> None:
        self.population += 1
        if self.unlimited:
            self._all.append(payload)
            return
        key = stable_key(key_source)
        self._counter += 1
        entry = (-key, -self._counter, payload)
        if len(self._heap) < self.n:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            # entry has a *smaller* key than the current worst (keys negated).
            heapq.heapreplace(self._heap, entry)

    def result(self) -> list[Any]:
        if self.unlimited:
            return self._all
        return [payload for _, _, payload in sorted(self._heap, reverse=True)]

    def info(self) -> SampleInfo:
        items = len(self._all) if self.unlimited else len(self._heap)
        return SampleInfo(
            population=self.population,
            sampled=items,
            mode="full" if self.unlimited else f"smallest-{self.n}",
        )


def sample_iter(items: Iterable[Any], n: int, key_of) -> tuple[list[Any], SampleInfo]:
    """Convenience wrapper: sample `n` items out of an iterable."""
    sampler = SmallestNSampler(n)
    for item in items:
        sampler.offer(key_of(item), item)
    return sampler.result(), sampler.info()
