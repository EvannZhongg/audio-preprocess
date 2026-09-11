"""Speaker diarization via pyannote or DiariZen.

The existing pyannote backend stays in-process. DiariZen runs through a small
subprocess adapter because it requires its own patched pyannote environment.
Both backends expose the same DataFrame result consumed by VAD and Segmenter.

The DiariZen adapter defaults to a RESIDENT worker (`diarizen.resident`): one
long-lived child with the model loaded once, fed newline-delimited JSON
requests. This mirrors the pyannote backend, which has always loaded its model
once in `__init__` and kept it for the process's life. Set
`diarizen.resident: false` to fall back to one subprocess per chunk.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import threading
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

# How long to wait for the worker's "ready" frame. Covers interpreter start,
# `import torch`, HF path resolution and the model load. Generous because a
# cold HF cache lookup can be slow; bounded because this is spent under the
# caller's GPU lock.
_SPAWN_TIMEOUT_S = 600
# Bytes of the child's stderr log to quote when it dies unexpectedly.
_STDERR_TAIL_BYTES = 4000


class _DiarizenWorker:
    """A long-lived DiariZen child process. Model loaded once, one request at
    a time.

    Not thread-safe on its own; callers hold `Diarizer._lock`.

    The child terminates when its stdin closes, which the kernel guarantees
    whenever the parent dies -- by any means, including SIGKILL. That is the
    ONLY thing preventing a GPU-resident orphan, because the production
    teardown paths run no user code in the parent:
    `ray.kill(handle, no_restart=True)` (pipeline_v2_ray/driver.py) skips
    atexit/__del__/__ray_terminate__ entirely, and `mp.Pool.__exit__` sends
    SIGTERM. So there is deliberately no atexit hook or __del__ here: either
    would only create the illusion of cleanup.
    """

    def __init__(self, command: list[str], env: dict[str, str],
                 stderr_path: str) -> None:
        # Spawns nothing: construction must stay cheap and side-effect-free.
        self._command = command
        self._env = env
        self._stderr_path = stderr_path
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_handle = None
        self._next_id = 0
        self.startup: dict = {}
        self.spawn_ms = 0

    # ------------------------------------------------------------------
    def _spawn(self) -> None:
        os.makedirs(os.path.dirname(self._stderr_path) or ".", exist_ok=True)
        # stderr goes to a FILE, never a pipe: the child logs on every request
        # (DiariZen prints "Extracting segmentations." etc. unconditionally),
        # and an unread pipe deadlocks the child once the ~64KB kernel buffer
        # fills. A file is written by the kernel and can never fill.
        self._stderr_handle = open(self._stderr_path, "a", encoding="utf-8")
        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            env=self._env,
            text=True,
            bufsize=1,
        )
        line = self._readline(_SPAWN_TIMEOUT_S)
        if not line:
            detail = "timed out" if line is None else "exited"
            self._kill()
            raise RuntimeError(
                f"DiariZen worker {detail} before becoming ready; "
                f"stderr tail: {self._stderr_tail()}"
            )
        frame = json.loads(line)
        if frame.get("event") != "ready":
            self._kill()
            raise RuntimeError(f"unexpected first frame from worker: {frame!r}")
        self.startup = frame.get("startup", {})
        self.spawn_ms = int((time.perf_counter() - t0) * 1000)

    def ensure_started(self) -> None:
        if self._proc is None:
            self._spawn()
        else:
            self.spawn_ms = 0

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    def request(self, input_path: Path, output_path: Path, timeout: float) -> dict:
        """Send one diarization request and return the child's timings.

        Raises on failure. The caller decides whether the worker is still
        trustworthy: an `ok: false` reply is a per-chunk error and leaves the
        worker usable, while a timeout / EOF / desync means it must be
        discarded.
        """
        assert self._proc is not None and self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
        payload = json.dumps(
            {"id": request_id, "input": str(input_path), "output": str(output_path)}
        )
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise _WorkerBroken(
                f"worker stdin closed ({type(exc).__name__}); "
                f"stderr tail: {self._stderr_tail()}"
            ) from exc

        line = self._readline(timeout)
        if line is None:
            raise _WorkerBroken(
                f"worker timed out after {timeout:.0f}s; "
                f"stderr tail: {self._stderr_tail()}"
            )
        if line == "":
            raise _WorkerBroken(
                f"worker exited mid-request (code {self._proc.poll()}); "
                f"stderr tail: {self._stderr_tail()}"
            )
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _WorkerBroken(f"unparseable frame {line[:200]!r}") from exc
        if frame.get("id") != request_id:
            # Streams are out of sync; nothing this worker says can be trusted.
            raise _WorkerBroken(
                f"frame id {frame.get('id')!r} != expected {request_id}"
            )
        if not frame.get("ok"):
            raise RuntimeError(
                f"DiariZen inference failed: {frame.get('error')}\n"
                f"{frame.get('traceback', '')}"
            )
        return frame

    # ------------------------------------------------------------------
    def _readline(self, timeout: float) -> Optional[str]:
        """Read one frame, or None on timeout, or "" on EOF.

        This is how the per-request deadline is enforced now that
        `subprocess.run(timeout=)` no longer applies. `select` only promises
        that *some* bytes are ready, so a partial line could still block the
        readline() -- acceptable because the child writes each frame as a
        single small write plus flush, well under PIPE_BUF, so a readable pipe
        always holds a complete frame.
        """
        assert self._proc is not None and self._proc.stdout is not None
        with selectors.DefaultSelector() as sel:
            sel.register(self._proc.stdout, selectors.EVENT_READ)
            if not sel.select(timeout):
                return None
        return self._proc.stdout.readline()

    def _stderr_tail(self) -> str:
        try:
            if self._stderr_handle is not None:
                self._stderr_handle.flush()
            with open(self._stderr_path, "r", encoding="utf-8", errors="replace") as fp:
                return fp.read()[-_STDERR_TAIL_BYTES:].replace("\n", " | ")
        except Exception:  # noqa: BLE001 - diagnostics must not mask the error
            return "<unavailable>"

    def _kill(self) -> None:
        """Terminate and reap. Idempotent; never raises."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()   # EOF: lets the child exit cleanly
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            for stream in (proc.stdout, proc.stdin):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
        handle, self._stderr_handle = self._stderr_handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._kill()


class _WorkerBroken(RuntimeError):
    """The worker can no longer be trusted and must be replaced."""


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
        # The DiariZen worker is spawned lazily on the first run(), never here:
        # __init__ must stay side-effect-free so a Diarizer can be built for
        # config/command inspection on a machine with neither .venv-diarizen
        # nor CUDA (see tests/test_pipeline_v2_local_adapter_sync.py).
        self._worker: Optional[_DiarizenWorker] = None
        # Ray already serializes diarize() behind its per-actor GPU lock, but
        # qc/models_bundle.py builds its own Diarizer outside that discipline,
        # so own the invariant here rather than relying on every caller.
        self._lock = threading.Lock()

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
        timeout = max(
            900,
            int(duration * self.params.timeout_per_audio_second)
            + self.params.timeout_base_seconds,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="pipeline-v2-diarizen-") as temp_dir:
                input_path = Path(temp_dir) / "input.wav"
                output_path = Path(temp_dir) / "diarization.json"
                # The temp wav stays the payload channel even in resident mode:
                # measured at ~0.2s for a 1800s chunk, versus forking DiariZen's
                # vendored __call__ (which only accepts a path/BytesIO) to pass
                # an in-memory array. Keeping it also makes resident mode
                # byte-for-byte equivalent to the one-shot path.
                t0 = time.perf_counter()
                sf.write(
                    input_path,
                    np.asarray(waveform, dtype=np.float32),
                    sample_rate,
                    subtype="PCM_16",
                )
                write_wav_ms = int((time.perf_counter() - t0) * 1000)

                if self.params.diarizen_resident:
                    timings = self._request_resident(
                        input_path, output_path, timeout, log_tag
                    )
                else:
                    timings = self._run_diarizen_oneshot(
                        input_path, output_path, timeout
                    )
                payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            logger.error(f"dia_runtime_error {traceback.format_exc()}", extra=log_tag)
            return None

        t0 = time.perf_counter()
        rows = self._normalize_diarizen_segments(
            payload.get("segments", []), duration
        )
        diarize_df = pd.DataFrame(
            rows, columns=["segment", "label", "speaker", "start", "end"]
        )
        postprocess_ms = (
            int((time.perf_counter() - t0) * 1000) + timings.get("postprocess_ms", 0)
        )

        total_ms = int((time.perf_counter() - t_total) * 1000)
        infer_ms = timings.get("infer_ms", 0)
        spawn_ms = timings.get("spawn_ms", 0)
        # Everything unaccounted for: IPC, the child reading the wav back, and
        # subprocess teardown in one-shot mode. Large values mean the transport
        # is the problem, not the model.
        ipc_ms = max(0, total_ms - infer_ms - spawn_ms - write_wav_ms - postprocess_ms)
        logger.info(
            f"dia_time_cost provider diarizen "
            f"speakers {diarize_df['speaker'].nunique() if not diarize_df.empty else 0} "
            f"segments {len(diarize_df)} infer_ms {infer_ms} "
            f"postprocess_ms {postprocess_ms} total_ms {total_ms} "
            f"spawn_ms {spawn_ms} write_wav_ms {write_wav_ms} ipc_ms {ipc_ms} "
            f"resident {self.params.diarizen_resident}",
            extra=log_tag,
        )
        # PipelineV2 does not consume diarization centroids; returning an empty
        # dict preserves the established tuple/state shape.
        return diarize_df, {}

    def _request_resident(
        self,
        input_path: Path,
        output_path: Path,
        timeout: float,
        log_tag: Optional[dict],
    ) -> dict:
        """Send one request to the resident worker, spawning it on first use.

        A worker that times out, dies or desyncs is discarded so the NEXT call
        respawns it. There is deliberately no inline retry: the caller already
        turns a None result into a failed chunk, and retrying here would double
        an already-long timeout while holding the GPU lock.
        """
        with self._lock:
            if self._worker is None:
                self._worker = _DiarizenWorker(
                    self._diarizen_command(input_path, output_path, serve=True),
                    self._diarizen_env(),
                    os.path.join("logs", f"diarizen_worker.{os.getpid()}.log"),
                )
            worker = self._worker
            try:
                worker.ensure_started()
                if worker.spawn_ms:
                    logger.info(
                        f"dia_worker_ready spawn_ms {worker.spawn_ms} "
                        f"startup {json.dumps(worker.startup, ensure_ascii=False)}",
                        extra=log_tag,
                    )
                frame = worker.request(input_path, output_path, timeout)
            except _WorkerBroken:
                # Drop the handle BEFORE re-raising, so a later exception path
                # can never leave a dead worker installed forever. Any other
                # exception (notably an ok:false reply) is a per-chunk failure
                # and leaves this healthy worker in place for the next chunk.
                self._worker = None
                worker.close()
                raise
            return {**frame, "spawn_ms": worker.spawn_ms}

    def _run_diarizen_oneshot(
        self, input_path: Path, output_path: Path, timeout: float
    ) -> dict:
        """Original behavior: one fresh subprocess per chunk."""
        t0 = time.perf_counter()
        proc = subprocess.run(
            self._diarizen_command(input_path, output_path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=self._diarizen_env(),
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(
                f"DiariZen exited with code {proc.returncode}: {details}"
            )
        # One-shot cannot separate inference from process startup + model load,
        # so report the whole span as spawn_ms and leave infer_ms at 0 rather
        # than overstating it (which is what this log used to do).
        return {"spawn_ms": elapsed_ms}

    def _diarizen_command(
        self, input_path: Path, output_path: Path, serve: bool = False
    ) -> list[str]:
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
            ("--segmentation-step", p.diarizen_segmentation_step),
            ("--batch-size", p.diarizen_batch_size),
        ):
            if value is not None:
                command.extend([option, str(value)])
        # Tri-state: absent means "inherit the model's config.toml", so the
        # paired --no- form is needed to express an explicit False.
        if p.diarizen_apply_median_filtering is not None:
            command.append(
                "--apply-median-filtering"
                if p.diarizen_apply_median_filtering
                else "--no-apply-median-filtering"
            )
        if serve:
            # In serve mode --input/--output are ignored (paths arrive per
            # request), but they are left on the command line so the argv is
            # identical between modes and stays easy to reproduce by hand.
            command.append("--serve")
        return command

    def _diarizen_env(self) -> dict[str, str]:
        """Give the subprocess the same physical GPU selected by PipelineV2."""
        env = os.environ.copy()
        # Env vars are inherited for free, but a token that lives only in the
        # config json would otherwise never reach the child.
        if self.params.huggingface_token and not env.get("HF_TOKEN"):
            env["HF_TOKEN"] = self.params.huggingface_token
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

    def close(self) -> None:
        """Release the resident DiariZen worker, if any. Idempotent.

        Only useful where a cooperative shutdown exists. The production Ray
        path kills actors with ray.kill(no_restart=True), which runs no user
        code -- there the child's stdin-EOF self-exit is what cleans up.
        """
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()

    def reap_if_dead(self) -> bool:
        """Drop the worker handle if the child has exited, so the next call
        respawns instead of writing into a dead pipe. Returns True if a dead
        worker was reaped. Cheap enough for a per-file check."""
        with self._lock:
            worker = self._worker
            if worker is None or worker.is_alive():
                return False
            self._worker = None
        worker.close()
        return True

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
