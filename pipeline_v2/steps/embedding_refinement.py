"""Embedding-based refinement of VAD segments.

For each VAD segment >= 1s:
  - compute a reference ERes2NetV2 embedding over the whole segment
  - slide 1.1s / 0.4s windows across, embed each window, measure cosine
    similarity to the reference
  - if any window falls below `inter_similarity_threshold`, drop the
    whole segment (it likely contains multiple speakers or non-speech)
  - on segments that survive, attach `reference_embedding` and
    `min_similarity` to feed downstream same-speaker merge logic
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

import librosa
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence

import logger
from models.eres2net.ERes2NetV2 import ERes2NetV2
from models.eres2net.features import FBank
from pipeline_v2.params import EmbeddingRefinementParams
from pipeline_v2.state import Segment


_FEAT_SR = 16000
_MIN_SEGMENT_DURATION_S = 1.0
_MAX_SEGMENT_DURATION_S = 30
_WINDOW_SIZE_S = 1.1
_WINDOW_STEP_S = 0.4
_MIN_WAVEFORM_S = 0.1
_REF_BATCH_SIZE = 8


class EmbeddingRefiner:
    """Loads ERes2NetV2 once; reused per file."""

    def __init__(self, params: EmbeddingRefinementParams, device: str) -> None:
        self.params = params
        self.device = torch.device(device)

        model = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
        state_dict = torch.load(params.eres2net_model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model: ERes2NetV2 = model
        self.feature_extractor: FBank = FBank()

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        vad_list: list[Segment],
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[list[Segment]]:
        if not vad_list:
            logger.error("emb_empty_input_vad_list", extra=log_tag)
            return None

        t_total = time.perf_counter()
        try:
            refined: list[Segment] = []
            n_short = n_no_ref = n_no_window = n_dropped = 0
            n_too_long = 0  # [MAX_SEG_SKIP] 计数被跳过的超长片段，调试用，可整行删除
            default_sim = self.params.inter_similarity_threshold

            cand: list[tuple[Segment, np.ndarray, list[np.ndarray]]] = []
            for seg in vad_list:
                duration = seg.end - seg.start
                if duration > _MAX_SEGMENT_DURATION_S:
                    n_too_long += 1  # [MAX_SEG_SKIP] 调试用，可整行删除
                    continue
                if duration < _MIN_SEGMENT_DURATION_S:
                    seg.min_similarity = default_sim
                    seg.reference_embedding = None
                    refined.append(seg)
                    n_short += 1
                    continue

                seg_wave = waveform[
                    int(seg.start * sample_rate) : int(seg.end * sample_rate)
                ]
                if len(seg_wave) / sample_rate < _MIN_WAVEFORM_S:
                    seg.min_similarity = default_sim
                    seg.reference_embedding = None
                    refined.append(seg)
                    n_no_ref += 1
                    continue

                windows = self._collect_windows(seg_wave, sample_rate, duration)
                if not windows:
                    seg.min_similarity = default_sim
                    seg.reference_embedding = None
                    refined.append(seg)
                    n_no_window += 1
                    continue

                cand.append((seg, seg_wave, windows))

            if cand:
                ref_waves = [c[1] for c in cand]
                ref_embs = self._embed_batched(ref_waves, sample_rate, batch=_REF_BATCH_SIZE)

                win_counts = [len(c[2]) for c in cand]
                flat_windows: list[np.ndarray] = [w for c in cand for w in c[2]]
                flat_win_embs = self._embed_batched(flat_windows, sample_rate)

                offset = 0
                for (seg, _, _), ref_emb, n_win in zip(cand, ref_embs, win_counts):
                    win_embs = flat_win_embs[offset : offset + n_win]
                    offset += n_win

                    ref_emb_2d = ref_emb.reshape(1, -1)
                    consistent, min_sim = self._check_consistency(ref_emb_2d, win_embs)
                    if not consistent:
                        n_dropped += 1
                        continue

                    seg.min_similarity = float(min_sim)
                    seg.reference_embedding = ref_emb_2d
                    refined.append(seg)
        except Exception:
            logger.error(f"emb_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"emb_time_cost in {len(vad_list)} out {len(refined)} "
            f"short {n_short} no_ref {n_no_ref} no_window {n_no_window} "
            f"dropped {n_dropped} total_ms {total_ms}",
            extra=log_tag,
        )
        return refined

    # ------------------------------------------------------------------
    # internal: embedding
    # ------------------------------------------------------------------
    def _embed_single(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if len(wav) / sr < _MIN_WAVEFORM_S:
            return None
        wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=_FEAT_SR)
        feats = self.feature_extractor(
            torch.tensor(wav_16k, dtype=torch.float32).to(self.device)
        )
        with torch.no_grad():
            return self.model(feats.unsqueeze(0)).cpu().numpy()

    def _embed_batched(self, waves: list[np.ndarray], sr: int, batch: Optional[int] = None) -> np.ndarray:
        if batch is None:
            batch = self.params.refinement_batch_size
        all_emb: list[np.ndarray] = []
        for i in range(0, len(waves), batch):
            chunk = waves[i : i + batch]
            wavs_16k = [librosa.resample(w, orig_sr=sr, target_sr=_FEAT_SR) for w in chunk]
            feats = [
                self.feature_extractor(torch.tensor(w, dtype=torch.float32).to(self.device))
                for w in wavs_16k
            ]
            padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
            with torch.no_grad():
                all_emb.append(self.model(padded).cpu().numpy())
        return np.vstack(all_emb)

    # ------------------------------------------------------------------
    # internal: window selection / consistency
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_windows(
        seg_wave: np.ndarray, sr: int, duration: float
    ) -> list[np.ndarray]:
        windows: list[np.ndarray] = []
        win_start_s = 0.0
        while win_start_s + _WINDOW_SIZE_S <= duration:
            s = int(win_start_s * sr)
            e = int((win_start_s + _WINDOW_SIZE_S) * sr)
            chunk = seg_wave[s:e]
            if len(chunk) / sr >= _MIN_WAVEFORM_S:
                windows.append(chunk)
            win_start_s += _WINDOW_STEP_S
        return windows

    def _check_consistency(
        self, ref_emb: np.ndarray, win_embs: np.ndarray
    ) -> tuple[bool, float]:
        min_sim = float("inf")
        for win_emb in win_embs:
            sim = cosine_similarity(ref_emb, win_emb.reshape(1, -1))[0, 0]
            min_sim = min(min_sim, sim)
            if sim < self.params.inter_similarity_threshold:
                return False, min_sim
        return True, min_sim
