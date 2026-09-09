"""Speaker diarization via pyannote or DiariZen.

The existing pyannote backend stays in-process. DiariZen runs through a small
subprocess adapter because it requires its own patched pyannote environment.
Both backends expose the same DataFrame result consumed by VAD and Segmenter.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from pyannote.audio import Pipeline as PyannotePipeline

import logger
from pipeline_v2.params import DiarizationParams


class Diarizer:
    """Loads the selected backend once where possible; reused per file."""

    def __init__(self, params: DiarizationParams, device: str) -> None:
        self.params = params
        self.device = device
        self.dia_pipeline: Optional[PyannotePipeline] = (
            self._load_pyannote_pipeline()
            if params.provider == "pyannote"
            else None
        )

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def run(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict] = None,
    ) -> Optional[Tuple[pd.DataFrame, dict[str, np.ndarray]]]:
        if self.params.provider == "diarizen":
            return self._run_diarizen(waveform, sample_rate, log_tag)

        t_total = time.perf_counter()

        try:
            assert self.dia_pipeline is not None
            tensor = torch.from_numpy(waveform).to(self.dia_pipeline.device).unsqueeze(0)
        except Exception as e:
            logger.error(f"dia_input_convert_failed {e}", extra=log_tag)
            return None

        t0 = time.perf_counter()
        try:
            segments, embeddings = self.dia_pipeline(
                {"waveform": tensor, "sample_rate": sample_rate},
                return_embeddings=True,
            )
        except Exception:
            logger.error(f"dia_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None
        infer_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        diarize_df = self._segments_to_df(segments)
        centroids = {spk: embeddings[i] for i, spk in enumerate(segments.labels())}
        postprocess_ms = int((time.perf_counter() - t0) * 1000)

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"dia_time_cost provider {self.params.provider} "
            f"speakers {len(centroids)} segments {len(diarize_df)} "
            f"infer_ms {infer_ms} postprocess_ms {postprocess_ms} total_ms {total_ms}",
            extra=log_tag,
        )
        return diarize_df, centroids

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _load_pyannote_pipeline(self) -> PyannotePipeline:
        model_ref = self._resolve_model_ref()
        # pyannote.audio 3.3 predates PyTorch 2.6's weights_only=True
        # default. These are explicitly configured trusted checkpoints, so
        # retain pyannote's historical loading behavior for this scoped call.
        torch_load = torch.load

        def load_checkpoint(*args, **kwargs):
            if kwargs.get("weights_only") is None:
                kwargs["weights_only"] = False
            return torch_load(*args, **kwargs)

        with patch("torch.load", load_checkpoint):
            dia_pipeline = PyannotePipeline.from_pretrained(
                model_ref, use_auth_token=self.params.huggingface_token
            )
        dia_pipeline.to(torch.device(self.device))
        return dia_pipeline

    def _run_diarizen(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        log_tag: Optional[dict],
    ) -> Optional[Tuple[pd.DataFrame, dict[str, np.ndarray]]]:
        """Run DiariZen in its isolated interpreter and normalize its output."""
        t_total = time.perf_counter()
        duration = len(waveform) / sample_rate
        try:
            with tempfile.TemporaryDirectory(prefix="pipeline-v2-diarizen-") as temp_dir:
                input_path = Path(temp_dir) / "input.wav"
                output_path = Path(temp_dir) / "diarization.json"
                sf.write(
                    input_path,
                    np.asarray(waveform, dtype=np.float32),
                    sample_rate,
                    subtype="PCM_16",
                )
                command = self._diarizen_command(input_path, output_path)
                timeout = max(
                    900,
                    int(duration * self.params.timeout_per_audio_second)
                    + self.params.timeout_base_seconds,
                )
                proc = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    env=self._diarizen_env(),
                )
                if proc.returncode != 0:
                    details = (proc.stderr or proc.stdout).strip()
                    raise RuntimeError(
                        f"DiariZen exited with code {proc.returncode}: {details}"
                    )
                payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            logger.error(f"dia_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        rows = self._normalize_diarizen_segments(
            payload.get("segments", []), duration
        )
        diarize_df = pd.DataFrame(
            rows, columns=["segment", "label", "speaker", "start", "end"]
        )
        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"dia_time_cost provider diarizen "
            f"speakers {diarize_df['speaker'].nunique() if not diarize_df.empty else 0} "
            f"segments {len(diarize_df)} infer_ms {total_ms} "
            f"postprocess_ms 0 total_ms {total_ms}",
            extra=log_tag,
        )
        # PipelineV2 does not consume diarization centroids; returning an empty
        # dict preserves the established tuple/state shape.
        return diarize_df, {}

    def _diarizen_command(self, input_path: Path, output_path: Path) -> list[str]:
        p = self.params
        command = [
            self._resolve_existing_path(p.diarizen_python),
            self._resolve_existing_path(p.diarizen_runner),
            "--model",
            p.diarizen_model,
            "--embedding-model",
            p.diarizen_embedding_model,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cuda:0" if self.device.startswith("cuda") else self.device,
        ]
        if p.diarizen_model_dir_cache:
            command.extend(["--model-dir", p.diarizen_model_dir_cache])
        if p.diarizen_embedding_model_path:
            command.extend(
                ["--embedding-model-path", p.diarizen_embedding_model_path]
            )
        if p.diarizen_cache_dir:
            command.extend(["--cache-dir", p.diarizen_cache_dir])
        for option, value in (
            ("--num-speakers", p.num_speakers),
            ("--min-speakers", p.min_speakers),
            ("--max-speakers", p.max_speakers),
        ):
            if value is not None:
                command.extend([option, str(value)])
        return command

    def _diarizen_env(self) -> dict[str, str]:
        """Give the subprocess the same physical GPU selected by PipelineV2."""
        env = os.environ.copy()
        if not self.device.startswith("cuda"):
            return env

        index = 0
        if ":" in self.device:
            index = int(self.device.split(":", 1)[1])
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [item.strip() for item in visible.split(",") if item.strip()]
            if index >= len(devices):
                raise ValueError(
                    f"{self.device} is outside CUDA_VISIBLE_DEVICES={visible!r}"
                )
            env["CUDA_VISIBLE_DEVICES"] = devices[index]
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(index)
        return env

    @staticmethod
    def _resolve_existing_path(value: str) -> str:
        """Return an absolute path without dereferencing virtualenv symlinks."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return os.path.abspath(path)

    @staticmethod
    def _normalize_diarizen_segments(
        segments: list[dict], duration: float
    ) -> list[dict]:
        rows: list[dict] = []
        speaker_mapping: dict[str, str] = {}
        for item in segments:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
            if end <= start:
                continue
            source_speaker = str(item["speaker"])
            speaker = speaker_mapping.setdefault(
                source_speaker, f"SPEAKER_{len(speaker_mapping):02d}"
            )
            rows.append(
                {
                    "segment": None,
                    "label": None,
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                }
            )
        return sorted(rows, key=lambda row: (row["start"], row["end"], row["speaker"]))

    def _resolve_model_ref(self) -> str:
        """Prefer the local cache yaml if it points at existing weights;
        otherwise fall back to the HF model id."""
        cache = self.params.pyannote_model_dir_cache
        if cache and os.path.exists(cache):
            with open(cache, "r") as fp:
                cfg = yaml.safe_load(fp)
            params = cfg.get("pipeline", {}).get("params", {})
            seg = params.get("segmentation")
            emb = params.get("embedding")
            if seg and emb and os.path.exists(seg) and os.path.exists(emb):
                return cache
        return self.params.pyannote_model

    @staticmethod
    def _segments_to_df(segments) -> pd.DataFrame:
        df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        df["start"] = df["segment"].apply(lambda s: s.start)
        df["end"] = df["segment"].apply(lambda s: s.end)
        return df
