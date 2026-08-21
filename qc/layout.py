"""Discovering a pipeline_v3 output tree, and resolving paths out of it safely.

The layout QC reads (produced by pipeline_v2/steps/export.py and
pipeline_v2_ray/{segments,stage2_segments}.py):

    <output_root>/<shard>/segments_part-NNNNN.parquet          stage 1, per segment
    <output_root>/<shard>/stage2_segments_part-NNNNN.parquet    stage 2, per segment
    <output_root>/<shard>/audios/<base[:2]>/<base>_chunk<i>.wav chunk audio
    <output_root>/<shard>/jsons/<base[:2]>/<base>_chunk<i>.json chunk sidecar

`<shard>` mirrors a manifest shard name (e.g. `manifest_part-00000`), and
`base = sha1(relative_path)`. Note the audio granularity: one wav per *chunk*,
with each segment being a `[start, end)` window inside it -- so QC must slice,
never assume one file per segment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

STAGE1_PREFIX = "segments"
STAGE2_PREFIX = "stage2_segments"
MANIFEST_PREFIX = "manifest"


@dataclass(frozen=True)
class ShardLayout:
    """One shard subdirectory of the output root."""

    name: str
    path: str
    stage1_parts: list[str]
    stage2_parts: list[str]

    @property
    def has_stage1(self) -> bool:
        return bool(self.stage1_parts)

    @property
    def has_stage2(self) -> bool:
        return bool(self.stage2_parts)

    @property
    def status(self) -> str:
        if self.has_stage1 and self.has_stage2:
            return "stage1+stage2"
        if self.has_stage1:
            return "stage1_only"
        if self.has_stage2:
            return "stage2_only"
        return "empty"


def _list_parts(shard_dir: str, prefix: str) -> list[str]:
    """Sorted `<prefix>_part-*.parquet` paths in shard_dir.

    Reimplemented rather than calling source_scan.manifest.list_shards because
    that raises on a missing directory, and QC routinely points at trees where
    some shards have not been produced yet.
    """
    tag = f"{prefix}_part-"
    try:
        names = os.listdir(shard_dir)
    except OSError:
        return []
    return [
        os.path.join(shard_dir, n)
        for n in sorted(names)
        if n.startswith(tag) and n.endswith(".parquet")
    ]


def discover_shards(output_root: str, only: Optional[list[str]] = None) -> list[ShardLayout]:
    """Enumerate shard subdirectories of `output_root`, in name order.

    `only` restricts to the given shard names (silently skipping ones that do
    not exist, so `--shards` can be a superset while iterating a growing run).
    Shards with neither stage's parquet are returned too, so the report can
    distinguish "not started" from "produced nothing".
    """
    if not os.path.isdir(output_root):
        return []
    if only:
        names = [n for n in only if os.path.isdir(os.path.join(output_root, n))]
    else:
        names = sorted(
            n for n in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, n)) and not n.startswith("_")
        )
    shards = []
    for name in names:
        shard_dir = os.path.join(output_root, name)
        shards.append(ShardLayout(
            name=name,
            path=shard_dir,
            stage1_parts=_list_parts(shard_dir, STAGE1_PREFIX),
            stage2_parts=_list_parts(shard_dir, STAGE2_PREFIX),
        ))
    return shards


def discover_manifest_parts(manifest: Optional[str]) -> list[str]:
    """Resolve `--manifest` (a shard dir or a single parquet) to parquet paths."""
    if not manifest:
        return []
    if os.path.isfile(manifest):
        return [manifest]
    if not os.path.isdir(manifest):
        return []
    tag = f"{MANIFEST_PREFIX}_part-"
    names = sorted(
        n for n in os.listdir(manifest)
        if n.startswith(tag) and n.endswith(".parquet")
    )
    if names:
        return [os.path.join(manifest, n) for n in names]
    # Fall back to any parquet in the directory: manifests built by older
    # revisions of build_manifest.py did not always use the shard naming.
    return [
        os.path.join(manifest, n)
        for n in sorted(os.listdir(manifest))
        if n.endswith(".parquet")
    ]


def resolve_chunk_audio(output_root: str, chunk_audio_path: Optional[str]) -> Optional[str]:
    """Turn a parquet-stored relative chunk path into a vetted absolute path.

    `chunk_audio_path` is untrusted input (it came out of a parquet file that
    QC did not write), so an absolute path or a `..` component must not be
    allowed to make QC read outside the output tree. Returns None when the
    value is empty, escapes `output_root`, or does not exist -- the caller
    counts those as anomalies rather than crashing the run.
    """
    if not chunk_audio_path:
        return None
    if os.path.isabs(chunk_audio_path):
        return None
    root = os.path.realpath(output_root)
    candidate = os.path.realpath(os.path.join(root, chunk_audio_path))
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def sidecar_json_for(chunk_audio_abs: str) -> Optional[str]:
    """The per-chunk sidecar json matching a chunk wav.

    Mirrors the deterministic `audios/ -> jsons/`, `.wav -> .json` rewrite
    pipeline_v2_ray/actors/v2_stage_2.py:52-72 uses to write text back.
    """
    marker = os.sep + "audios" + os.sep
    if marker not in chunk_audio_abs or not chunk_audio_abs.endswith(".wav"):
        return None
    head, _, tail = chunk_audio_abs.rpartition(marker)
    path = head + os.sep + "jsons" + os.sep + tail[: -len(".wav")] + ".json"
    return path if os.path.isfile(path) else None
