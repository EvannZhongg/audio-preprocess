"""The single seam between QC and the production models.

Every model call in this package goes through `ModelBundle`. Concentrating the
coupling in one file has a concrete payoff: the four analyzers stay pure
statistics, and when an upstream signature moves there is exactly one place to
fix instead of four.

Two design points worth stating explicitly.

**Reuse, do not reimplement.** The speaker-consistency check calls
`EmbeddingRefiner`'s own `_collect_windows` / `_embed_batched` /
`_check_consistency`. Those are underscore-prefixed, which is a real cost --
they carry no compatibility promise. The alternative was worse: copying the
ERes2NetV2 load, the FBank contract, the 1.1s/0.4s windowing and the cosine
loop into QC would guarantee the two drift apart, and a QC tool that measures
something subtly different from production is not merely useless, it is
misleading. `verify_production_constants()` guards the risk by asserting at load
time that the constants QC mirrors still match the module.

**Why `EmbeddingRefiner.run()` is not used directly.** It is a *filter*: it
returns the surviving segments and throws away the per-segment
`min_similarity` for everything it drops (embedding_refinement.py:128-133). QC
needs the score for every segment, including the failures -- that is the whole
measurement -- so it drives the primitives instead.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from qc.config import (BROUHAHA_FAILURE_SENTINEL, EMBED_MIN_SEGMENT_S,
                       EMBED_WINDOW_S, EMBED_WINDOW_STEP_S, QCConfig)

# Feature sample rate shared by ERes2NetV2/FBank (kaldi fbank is called with
# `sample_frequency=16000`, models/eres2net/features.py:31-35) and by DNSMOS
# and brouhaha. Production resamples to this in every one of those paths.
FEAT_SR = 16000
# Shortest slice worth handing to a model, mirroring embedding_refinement.py's
# `_MIN_WAVEFORM_S`. Below this kaldi fbank yields zero frames and throws.
MIN_WAVEFORM_S = 0.1


def verify_production_constants() -> list[str]:
    """Check the constants QC mirrors still match `embedding_refinement`.

    Returns a list of human-readable mismatches (empty when all good). Called
    once per worker so a future upstream retune surfaces as a loud warning in
    the report instead of silently shifting every verdict.
    """
    problems: list[str] = []
    try:
        from pipeline_v2.steps import embedding_refinement as er
    except Exception as exc:  # noqa: BLE001 - can't verify without the module
        return [f"could not import embedding_refinement to verify constants: {exc}"]

    for qc_name, qc_value, up_name in (
        ("EMBED_MIN_SEGMENT_S", EMBED_MIN_SEGMENT_S, "_MIN_SEGMENT_DURATION_S"),
        ("EMBED_WINDOW_S", EMBED_WINDOW_S, "_WINDOW_SIZE_S"),
        ("EMBED_WINDOW_STEP_S", EMBED_WINDOW_STEP_S, "_WINDOW_STEP_S"),
        ("FEAT_SR", float(FEAT_SR), "_FEAT_SR"),
        ("MIN_WAVEFORM_S", MIN_WAVEFORM_S, "_MIN_WAVEFORM_S"),
    ):
        upstream = getattr(er, up_name, None)
        if upstream is None:
            problems.append(f"{up_name} no longer exists in embedding_refinement")
        elif float(upstream) != float(qc_value):
            problems.append(
                f"{up_name} changed upstream to {upstream}, QC mirrors {qc_value} "
                f"as {qc_name}"
            )
    for method in ("_collect_windows", "_embed_batched", "_check_consistency"):
        if not hasattr(er.EmbeddingRefiner, method):
            problems.append(f"EmbeddingRefiner.{method} no longer exists")
    return problems


class ModelBundle:
    """All four models, loaded once per worker process.

    Loading costs seconds to tens of seconds, so it happens exactly once per
    worker and every segment reuses it. Any per-segment load would dominate the
    runtime by orders of magnitude.
    """

    def __init__(self, cfg: QCConfig, device: str,
                 need_speaker: bool, need_background: bool,
                 need_embedding: bool) -> None:
        self.cfg = cfg
        self.device = device
        self.constant_warnings: list[str] = []
        self.load_errors: list[str] = []

        self._refiner = None
        self._diarizer = None
        self._dnsmos = None
        self._brouhaha = None

        # The embedding model serves two requirements (speaker consistency and
        # pair similarity), so it loads if either needs it.
        if need_speaker or need_embedding:
            self.constant_warnings = verify_production_constants()
            self._refiner = self._load_refiner()
        if need_speaker:
            self._diarizer = self._load_diarizer()
        if need_background:
            self._dnsmos = self._load_dnsmos()
            self._brouhaha = self._load_brouhaha()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _production_params(self):
        """Production's parsed params, needed by Diarizer/EmbeddingRefiner.

        Unlike the threshold parsing in qc/config.py this genuinely needs the
        full object (model paths, hf token, cache dirs), so it goes through the
        real import -- by the time models are loaded torch is present anyway.
        """
        from pipeline_v2.params import PipelineParams

        if not self.cfg.config_path:
            raise RuntimeError(
                "model re-checks need --config: the model paths, pyannote cache and "
                "auth token all come from the production config json"
            )
        return PipelineParams.from_config(self.cfg.config_path)

    def _load_refiner(self):
        try:
            from pipeline_v2.steps.embedding_refinement import EmbeddingRefiner

            params = self._production_params()
            return EmbeddingRefiner(params.embedding_refinement, self.device)
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"eres2net: {type(exc).__name__}: {exc}")
            return None

    def _load_diarizer(self):
        try:
            from pipeline_v2.steps.speaker_diarization import Diarizer

            params = self._production_params()
            return Diarizer(params.diarization, self.device)
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"diarizer: {type(exc).__name__}: {exc}")
            return None

    def _load_dnsmos(self):
        try:
            from models import dnsmos

            params = self._production_params()
            return dnsmos.ComputeScore(params.metrics.dnsmos_model_path, self.device)
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"dnsmos: {type(exc).__name__}: {exc}")
            return None

    def _load_brouhaha(self):
        try:
            from models import brouhaha_metrics

            params = self._production_params()
            met = params.metrics
            model_ref = met.brouhaha_model
            cache = met.brouhaha_model_dir_cache
            if cache and os.path.exists(cache):
                model_ref = cache
            return brouhaha_metrics.ComputeScore(
                model_ref, token=met.huggingface_token, device=self.device
            )
        except Exception as exc:  # noqa: BLE001
            self.load_errors.append(f"brouhaha: {type(exc).__name__}: {exc}")
            return None

    @property
    def has_embedding(self) -> bool:
        return self._refiner is not None

    @property
    def has_diarizer(self) -> bool:
        return self._diarizer is not None

    @property
    def has_dnsmos(self) -> bool:
        return self._dnsmos is not None

    @property
    def has_brouhaha(self) -> bool:
        return self._brouhaha is not None

    # ------------------------------------------------------------------
    # requirement 3 / 5: speaker embedding
    # ------------------------------------------------------------------
    def embed_consistency(
        self, seg_wave: np.ndarray, sample_rate: int
    ) -> tuple[Optional[float], int, Optional[str]]:
        """Sliding-window self-consistency of one segment.

        Returns `(min_cosine_similarity, n_windows, error)`. A low minimum means
        some 1.1s window does not sound like the segment's own average, which is
        production's proxy for "more than one speaker in here"
        (embedding_refinement.py:196-205).

        The windowing and embedding come from `EmbeddingRefiner`, but the
        minimum is computed here instead of via `_check_consistency`, on purpose:
        that method returns as soon as one window falls below the threshold
        (embedding_refinement.py:203-204), so its `min_sim` is the minimum *so
        far*, not the global one. Production only keeps the value for segments
        that never short-circuit, so the two agree there -- but QC also reports
        the distribution for segments that DO fail, and a partial minimum would
        make that distribution depend on the threshold it is meant to be
        evaluated against.

        `(None, 0, None)` -- no error -- means the segment is legitimately
        unscoreable: production skips embedding under 1s, so QC reports it as
        not-checked rather than inventing a verdict.
        """
        if self._refiner is None:
            return None, 0, "embedding model unavailable"
        duration = len(seg_wave) / float(sample_rate)
        if duration < EMBED_MIN_SEGMENT_S or duration < MIN_WAVEFORM_S:
            return None, 0, None
        try:
            windows = self._refiner._collect_windows(seg_wave, sample_rate, duration)
            if not windows:
                return None, 0, None
            ref = self._refiner._embed_batched([seg_wave], sample_rate, batch=1)
            win_embs = self._refiner._embed_batched(windows, sample_rate)
            ref_1d = np.asarray(ref, dtype=np.float64).ravel()
            min_sim: Optional[float] = None
            for win in win_embs:
                sim = cosine_similarity_1d(ref_1d, win)
                if sim is None:
                    continue
                min_sim = sim if min_sim is None else min(min_sim, sim)
            if min_sim is None:
                return None, 0, None
            return float(min_sim), len(windows), None
        except Exception as exc:  # noqa: BLE001 - one bad segment must not sink the batch
            return None, 0, f"{type(exc).__name__}: {exc}"

    def embed_reference(
        self, seg_wave: np.ndarray, sample_rate: int
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """One segment's reference embedding, for pair similarity (req. 5)."""
        if self._refiner is None:
            return None, "embedding model unavailable"
        if len(seg_wave) / float(sample_rate) < MIN_WAVEFORM_S:
            return None, None
        try:
            emb = self._refiner._embed_batched([seg_wave], sample_rate, batch=1)
            return emb.reshape(1, -1), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # requirement 3: diarization
    # ------------------------------------------------------------------
    def diarize_count(
        self, seg_wave: np.ndarray, sample_rate: int
    ) -> tuple[Optional[int], int, Optional[str]]:
        """Independent speaker count for one segment.

        Returns `(n_speakers, n_diarization_segments, error)`. Runs pyannote on
        the isolated segment, so the answer is genuinely independent of the
        diarization stage 1 already did on the whole chunk -- which is the point:
        agreement between the two methods is evidence, a re-derivation of stage
        1's own labels would not be.
        """
        if self._diarizer is None:
            return None, 0, "diarizer unavailable"
        if len(seg_wave) / float(sample_rate) < MIN_WAVEFORM_S:
            return None, 0, None
        try:
            result = self._diarizer.run(
                np.ascontiguousarray(seg_wave, dtype=np.float32), sample_rate
            )
            if result is None:
                return None, 0, "diarization returned None"
            df, centroids = result
            n_speakers = int(df["speaker"].nunique()) if len(df) else 0
            return n_speakers, int(len(df)), None
        except Exception as exc:  # noqa: BLE001
            return None, 0, f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # requirement 4: background
    # ------------------------------------------------------------------
    def audio_quality(self, seg_wave: np.ndarray, sample_rate: int) -> dict:
        """DNSMOS (SIG/BAK/OVRL) plus brouhaha (c50/snr) for one segment.

        `BAK` is the headline number for "was the background removed" and is new
        to QC: production's metrics step computes the same DNSMOS call but keeps
        only `OVRL` (pipeline_v2/steps/metrics.py:81-83), discarding the
        background component.

        brouhaha's `(-420.69, -420.69)` failure sentinel
        (models/brouhaha_metrics.py:39-41) is translated to None here. Leaving it
        as a number would drag every mean and percentile into nonsense.
        """
        out: dict = {
            "sig": None, "bak": None, "ovrl": None, "c50": None, "snr": None,
            "dnsmos_error": None, "brouhaha_error": None, "brouhaha_sentinel": False,
        }
        if len(seg_wave) / float(sample_rate) < MIN_WAVEFORM_S:
            out["dnsmos_error"] = "segment too short to score"
            out["brouhaha_error"] = "segment too short to score"
            return out

        wave_16k = self._to_feat_sr(seg_wave, sample_rate)
        if wave_16k is None or wave_16k.size == 0:
            out["dnsmos_error"] = "resample produced no samples"
            out["brouhaha_error"] = "resample produced no samples"
            return out

        if self._dnsmos is not None:
            try:
                scores = self._dnsmos(wave_16k, FEAT_SR, False)
                out["sig"] = float(scores["SIG"])
                out["bak"] = float(scores["BAK"])
                out["ovrl"] = float(scores["OVRL"])
            except Exception as exc:  # noqa: BLE001
                out["dnsmos_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["dnsmos_error"] = "dnsmos unavailable"

        if self._brouhaha is not None:
            try:
                c50, snr = self._brouhaha(wave_16k, FEAT_SR)
                if (abs(c50 - BROUHAHA_FAILURE_SENTINEL) < 1e-3
                        and abs(snr - BROUHAHA_FAILURE_SENTINEL) < 1e-3):
                    out["brouhaha_sentinel"] = True
                    out["brouhaha_error"] = "brouhaha returned its failure sentinel"
                else:
                    out["c50"] = float(c50)
                    out["snr"] = float(snr)
            except Exception as exc:  # noqa: BLE001
                out["brouhaha_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["brouhaha_error"] = "brouhaha unavailable"
        return out

    @staticmethod
    def _to_feat_sr(wave: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Resample to 16k the same way production's metrics step does."""
        if sample_rate == FEAT_SR:
            return np.ascontiguousarray(wave, dtype=np.float32)
        try:
            import librosa

            return np.ascontiguousarray(
                librosa.resample(
                    np.ascontiguousarray(wave, dtype=np.float32),
                    orig_sr=sample_rate, target_sr=FEAT_SR,
                ),
                dtype=np.float32,
            )
        except Exception:  # noqa: BLE001
            return None


def cosine_similarity_1d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cosine similarity between two embeddings.

    Same measure Segmenter uses to decide a merge
    (pipeline_v2/steps/segment.py:127-130), computed directly to avoid pulling
    sklearn into a hot loop for a two-vector dot product.
    """
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return None
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(va, vb) / (na * nb))
