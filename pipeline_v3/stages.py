"""Per-stage processing plugin registry.

Declares, per pipeline stage, which Params class configures it, which parquet
writer/resumer/error-record functions persist its output, and how to turn one
stage's finished result into the next stage's input. Everything is keyed by
stage_key and self-registered via `register_stage`, so adding a new stage
(e.g. stage_3) means adding one `register_stage(StageDef(...))` call here
(or in a new module imported alongside this one) plus a matching `stages:`
entry in configs/pipeline_v3.yaml -- no change to the generic scheduler in
pipeline_v3/driver.py or pool.py.

Reuses pipeline_v2_ray's actors (v2_stage_1 / v2_stage_2, registered into
ACTOR_REGISTRY simply by importing `pipeline_v2_ray.actors`), segment schemas
and resume/writer/error-record functions verbatim -- pipeline_v3 only adds
the multi-stage scheduling layer on top of pipeline_v2_ray's existing
per-file processing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from pipeline_v2.params import PipelineParams, Stage2Params
from pipeline_v2_ray.result import FileResult
from pipeline_v2_ray.segments import SEG_SHARD_SIZE as STAGE1_SEG_SHARD_SIZE
from pipeline_v2_ray.segments import error_record, resume_state, write_segments_shard
from pipeline_v2_ray.stage2_segments import SEG_SHARD_SIZE as STAGE2_SEG_SHARD_SIZE
from pipeline_v2_ray.stage2_segments import (error_record2, resume_state2,
                                              write_stage2_segments_shard)
from pipeline_v3.types import FileItem

__all__ = [
    "StageDef", "STAGE_REGISTRY", "register_stage",
    "stage1_result_to_stage2_items", "load_stage1_output_from_disk",
]


@dataclass(frozen=True)
class StageDef:
    key: str
    params_cls: type                    # PipelineParams / Stage2Params / ... -- drives yaml JSON parsing
    segment_writer: Callable            # (records, out_dir, part_index) -> path
    segment_resumer: Callable           # (shard_dir) -> (processed_keys, next_part_index)
    error_record_fn: Callable           # (source_key, shard, error) -> record
    seg_shard_size: int                 # row-count flush threshold for this stage's parquet
    next_stage: Optional[str] = None    # downstream stage key this stage feeds, if any (None = terminal)
    # (FileResult, output_root) -> list[FileItem] for `next_stage`. Applied to
    # one just-finished file's result (streaming hand-off). None for a
    # terminal stage.
    to_next_items: Optional[Callable[[FileResult, str], list]] = None
    # (output_root, shard_names) -> [(shard_name, [FileItem, ...]), ...].
    # Rebuilds THIS stage's own already-flushed parquet output as the
    # downstream stage's seed input -- used both when a run starts at this
    # stage's downstream neighbor without this stage in the same invocation,
    # and to seed a resumed multi-stage run with whatever this stage finished
    # before an earlier crash. None for a terminal stage.
    load_output_from_disk: Optional[Callable[[str, Optional[list]], list]] = None


STAGE_REGISTRY: dict[str, StageDef] = {}


def register_stage(d: StageDef) -> None:
    STAGE_REGISTRY[d.key] = d


# ---------------------------------------------------------------------------
# stage_1 -> stage_2 conversion. Mirrors main_v2_ray.collect_stage1_segments,
# generalized to work both on one live FileResult (streaming) and on a scan
# of already-flushed parquet (cold seed / disk-only mode).
# ---------------------------------------------------------------------------
def _group_stage1_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("error") is not None or not row.get("chunk_audio_path"):
            continue  # failed-file placeholder row; nothing to re-process
        groups.setdefault(row["chunk_audio_path"], []).append(row)
    return groups


def _stage1_rows_to_items(groups: dict[str, list[dict]], output_root: str) -> list[FileItem]:
    items: list[FileItem] = []
    for chunk_audio_path, rows in groups.items():
        rows.sort(key=lambda r: r.get("start") or 0.0)
        payload = [
            {
                "utt_id": r["utt_id"],
                "origin_source": r["source"],
                "chunk_index": r["chunk_index"],
                "speaker_id": r["speaker_id"],
                "start": r["start"],
                "end": r["end"],
            }
            for r in rows
        ]
        items.append(FileItem(
            audio_path=os.path.join(output_root, chunk_audio_path),
            relative_path=chunk_audio_path,
            duration=sum((r.get("end") or 0.0) - (r.get("start") or 0.0) for r in rows),
            payload=payload,
        ))
    return items


def stage1_result_to_stage2_items(fr: FileResult, output_root: str) -> list[FileItem]:
    """Streaming hand-off: one stage-1 file just finished -> its stage-2
    seed items, without touching disk."""
    if not fr.success or not fr.segments:
        return []
    return _stage1_rows_to_items(_group_stage1_rows(fr.segments), output_root)


def load_stage1_output_from_disk(output_root: str, shard_names: Optional[list] = None):
    """Cold path: scan stage-1's already-flushed segments_part-*.parquet
    under `output_root` and rebuild stage-2 seed items, grouped by shard.
    Used when a run selects only stage_2 (no stage_1 in this invocation) and,
    within a multi-stage run, to seed a shard's stage-2 queue with whatever
    stage-1 work already finished (and flushed) in an earlier, interrupted
    run before stage-2 got to it. Mirrors main_v2_ray.collect_stage1_segments.

    `shard_names`: restrict the scan to these shard subdirs (e.g. one shard,
    for the resume-seed use case); None scans every shard subdir of
    `output_root` (the "only run stage_2" use case).
    """
    import pyarrow.parquet as pq

    from source_scan.manifest import list_shards

    cols = ["utt_id", "source", "chunk_index", "chunk_audio_path",
            "speaker_id", "start", "end", "error"]
    if shard_names is None:
        if not os.path.isdir(output_root):
            return []
        shard_names = sorted(
            d for d in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, d))
        )
    groups: list[tuple[str, list[FileItem]]] = []
    for shard_name in shard_names:
        shard_dir = os.path.join(output_root, shard_name)
        if not os.path.isdir(shard_dir):
            continue
        parts = list_shards(shard_dir, "segments")
        if not parts:
            continue
        rows: list[dict] = []
        for part in parts:
            rows.extend(pq.read_table(part, columns=cols).to_pylist())
        items = _stage1_rows_to_items(_group_stage1_rows(rows), output_root)
        if items:
            groups.append((shard_name, items))
    return groups


register_stage(StageDef(
    key="stage_1",
    params_cls=PipelineParams,
    segment_writer=write_segments_shard,
    segment_resumer=resume_state,
    error_record_fn=error_record,
    seg_shard_size=STAGE1_SEG_SHARD_SIZE,
    next_stage="stage_2",
    to_next_items=stage1_result_to_stage2_items,
    load_output_from_disk=load_stage1_output_from_disk,
))

register_stage(StageDef(
    key="stage_2",
    params_cls=Stage2Params,
    segment_writer=write_stage2_segments_shard,
    segment_resumer=resume_state2,
    error_record_fn=error_record2,
    seg_shard_size=STAGE2_SEG_SHARD_SIZE,
    # Terminal stage today. To add stage_3: set next_stage="stage_3" here and
    # provide to_next_items/load_output_from_disk analogous to stage_1's,
    # converting a Stage2SegmentRecord group into stage_3's FileItems.
    next_stage=None,
    to_next_items=None,
    load_output_from_disk=None,
))
