from __future__ import annotations

import json
import os
import time
import traceback
from typing import Optional

import numpy as np

import logger
from pipeline_v2.state import Segment
from utils.tool import write_mp3


class Exporter:
    def run(
        self,
        segment_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        audio_path: str,
        chunk_index: int,
        output_folder: str,
        log_tag: Optional[dict] = None,
    ) -> Optional[str]:
        if not segment_list:
            logger.error("export_empty_segment_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            os.makedirs(output_folder, exist_ok=True)
            base = os.path.splitext(os.path.basename(audio_path))[0]
            file_name = f"{base}_chunk{chunk_index}"

            audio_path_out = os.path.join(output_folder, f"{file_name}.mp3")
            write_mp3(audio_path_out, sample_rate, waveform)

            payload = {
                "file_name": file_name,
                "source": audio_path,
                "chunk_index": chunk_index,
                "audio_path": audio_path_out,
                "sample_rate": sample_rate,
                "duration": round(len(waveform) / sample_rate, 5),
                "sentences": [self._segment_to_dict(s, file_name) for s in segment_list],
            }
            json_path = os.path.join(output_folder, f"{file_name}.json")
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
        return json_path

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
