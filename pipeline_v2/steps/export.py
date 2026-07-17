from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from typing import Optional

import numpy as np

import logger
from pipeline_v2.state import PIPELINE_VERSION, Segment, SegmentRecord
from utils.tool import write_wav


class Exporter:
    def run(
        self,
        segment_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        relative_path: str,
        shard: Optional[str],
        chunk_index: int,
        output_folder: str,
        log_tag: Optional[dict] = None,
    ) -> Optional[list[SegmentRecord]]:
        """Export one chunk's audio (WAV) + a sidecar JSON of its segments, and
        return the flat per-segment records for the driver to accumulate into
        segments_part parquet. Returns None on failure/empty.

        `relative_path` is the source path relative to the audio root (same key
        as the stage-0 manifest). Its SHA-1 is the file id `base`, so the id is
        stable across mount points and joins back to the manifest.

        `output_folder` is the per-manifest-shard directory. Within it, audio
        and json live in separate trees, each bucketed by the first 2 hex of
        `base` to keep any single directory small on JuiceFS:
            <output_folder>/audios/<base[:2]>/<base>_chunk<idx>.wav
            <output_folder>/jsons/<base[:2]>/<base>_chunk<idx>.json
        """
        if not segment_list:
            logger.error("export_empty_segment_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            base = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()  # 40 hex
            bucket = base[:2]
            file_name = f"{base}_chunk{chunk_index}"

            audio_dir = os.path.join(output_folder, "audios", bucket)
            json_dir = os.path.join(output_folder, "jsons", bucket)
            os.makedirs(audio_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            audio_path_out = os.path.join(audio_dir, f"{file_name}.wav")
            write_wav(audio_path_out, sample_rate, waveform)
            # Path stored in the record is RELATIVE to output_root and prefixed
            # with the shard (<shard>/audios/<bucket>/<file>.wav) so it's
            # self-contained and mount-independent. Falls back to shard-dir
            # relative when shard is unknown (non-ray --input mode).
            rel_parts = ["audios", bucket, f"{file_name}.wav"]
            audio_rel = os.path.join(shard, *rel_parts) if shard else os.path.join(*rel_parts)

            chunk_duration = round(len(waveform) / sample_rate, 5)
            records = [
                self._segment_record(s, file_name, relative_path, shard, chunk_index,
                                     audio_rel, sample_rate, chunk_duration)
                for s in segment_list
            ]

            # Keep the per-chunk JSON sidecar (nested, human-readable).
            payload = {
                "file_name": file_name,
                "source": relative_path,
                "pipeline_version": PIPELINE_VERSION,
                "chunk_index": chunk_index,
                "audio_path": audio_rel,
                "sample_rate": sample_rate,
                "duration": chunk_duration,
                "sentences": [self._segment_to_dict(s, file_name) for s in segment_list],
            }
            json_path = os.path.join(json_dir, f"{file_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.error(f"export_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"export_time_cost total_ms {total_ms} segments {len(segment_list)} "
            f"json {json_path} audio {audio_path_out}",
            extra=log_tag,
        )
        return records

    @staticmethod
    def _segment_record(seg: Segment, file_name: str, relative_path: str,
                        shard: Optional[str], chunk_index: int, audio_rel: str,
                        sample_rate: int, chunk_duration: float) -> SegmentRecord:
        """Flat row for segments_part parquet (mirrors SEGMENT_SCHEMA)."""
        return {
            "utt_id": f"{file_name}_{seg.index}",
            "source": relative_path,
            "shard": shard,
            "pipeline_version": PIPELINE_VERSION,
            "chunk_index": chunk_index,
            "chunk_audio_path": audio_rel,
            "sample_rate": sample_rate,
            "chunk_duration": chunk_duration,
            "speaker_id": seg.speaker,
            "speaker_min_similarity": round(seg.min_similarity, 4),
            "start": round(seg.start, 5),
            "end": round(seg.end, 5),
            "seg_duration": round(seg.end - seg.start, 5),
            "dnsmos": round(seg.dnsmos, 4) if seg.dnsmos is not None else None,
            "c50": round(seg.c50, 4) if seg.c50 is not None else None,
            "snr": round(seg.snr, 4) if seg.snr is not None else None,
            "error": None,   # a real segment; failed-file rows set this instead
        }

    @staticmethod
    def _segment_to_dict(seg: Segment, file_name: str) -> dict:
        return {
            "utt_id": f"{file_name}_{seg.index}",
            "speaker_id": seg.speaker,
            "speaker_min_similarity": round(seg.min_similarity, 4),
            "time_range": {
                "duration": round(seg.end - seg.start, 5),
                "start": round(seg.start, 5),
                "end": round(seg.end, 5),
            },
            "audio_quality_info": {
                "dnsmos": round(seg.dnsmos, 4) if seg.dnsmos is not None else None,
                "c50": round(seg.c50, 4) if seg.c50 is not None else None,
                "snr": round(seg.snr, 4) if seg.snr is not None else None,
            },
        }
