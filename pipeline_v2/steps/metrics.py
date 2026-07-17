from __future__ import annotations

import os
import time
import traceback
from typing import Optional

import librosa
import numpy as np

import logger
from models import brouhaha_metrics, dnsmos
from pipeline_v2.params import MetricsParams
from pipeline_v2.state import Segment


_FEAT_SR = 16000


class MetricsScorer:
    def __init__(self, params: MetricsParams, device: str) -> None:
        self.params = params
        self.device = device
        self.dnsmos_compute_score = dnsmos.ComputeScore(
            params.dnsmos_model_path, device
        )
        self.brouhaha_metric: Optional[brouhaha_metrics.ComputeScore] = None
        if params.use_brouhaha:
            model_ref = params.brouhaha_model
            cache = params.brouhaha_model_dir_cache
            if cache and os.path.exists(cache):
                model_ref = cache
            self.brouhaha_metric = brouhaha_metrics.ComputeScore(
                model_ref, token=params.huggingface_token, device=device
            )

    def run(
        self,
        segment_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[list[Segment]]:
        if not segment_list:
            logger.error("metrics_empty_input_segment_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            audio_16k = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=_FEAT_SR
            )

            fixed_c50 = self.params.fixed_c50_threshold
            fixed_snr = self.params.fixed_snr_threshold

            for seg in segment_list:
                start = int(seg.start * _FEAT_SR)
                end = int(seg.end * _FEAT_SR)
                chunk = audio_16k[start:end]

                seg.dnsmos = float(
                    self.dnsmos_compute_score(chunk, _FEAT_SR, False)["OVRL"]
                )
                if self.brouhaha_metric is not None:
                    c50, snr = self.brouhaha_metric(chunk, _FEAT_SR)
                    seg.c50 = float(c50)
                    seg.snr = float(snr)
                else:
                    seg.c50 = fixed_c50
                    seg.snr = fixed_snr
        except Exception:
            logger.error(f"metrics_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        avg_dnsmos = float(np.mean([s.dnsmos for s in segment_list]))
        avg_c50 = float(np.mean([s.c50 for s in segment_list]))
        avg_snr = float(np.mean([s.snr for s in segment_list]))

        strategy = self.params.strategy
        if strategy == "fixed":
            d_th = self.params.fixed_dnsmos_threshold
            c_th = self.params.fixed_c50_threshold
            s_th = self.params.fixed_snr_threshold
            filtered = [
                s for s in segment_list
                if s.dnsmos >= d_th and s.c50 >= c_th and s.snr >= s_th
            ]
        else:
            d_th = avg_dnsmos
            filtered = [s for s in segment_list if s.dnsmos >= d_th]

        n_dropped = len(segment_list) - len(filtered)
        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"metrics_time_cost total_ms {total_ms} "
            f"in {len(segment_list)} out {len(filtered)} dropped {n_dropped} "
            f"strategy {strategy} "
            f"avg_dnsmos {avg_dnsmos:.2f} avg_c50 {avg_c50:.2f} avg_snr {avg_snr:.2f} "
            f"use_brouhaha {self.brouhaha_metric is not None}",
            extra=log_tag,
        )
        return filtered
