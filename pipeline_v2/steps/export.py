from __future__ import annotations

import json
import os
import time
import traceback
from typing import Optional

import logger
from pipeline_v2.state import Segment


class Exporter:
    def __init__(self, output_folder: str) -> None:
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def run(
        self,
        segment_list: list[Segment],
        audio_path: str,
        chunk_index: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[str]:
        if not segment_list:
            logger.error("export_empty_segment_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            base = os.path.splitext(os.path.basename(audio_path))[0]
            file_name = f"{base}_chunk{chunk_index}"
            payload = {
                "file_name": file_name,
                "source": audio_path,
                "chunk_index": chunk_index,
                "sentences": [self._segment_to_dict(s, file_name) for s in segment_list],
            }
            out_path = os.path.join(self.output_folder, f"{file_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.error(f"export_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"export_time_cost total_ms {total_ms} segments {len(segment_list)} "
            f"path {out_path}",
            extra=log_tag,
        )
        return out_path

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
