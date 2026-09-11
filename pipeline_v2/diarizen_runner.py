#!/usr/bin/env python3
"""Isolated DiariZen inference entrypoint used by PipelineV2.

This file intentionally imports DiariZen only inside the separate interpreter
selected by ``diarization.diarizen_python``.

Two modes:

* one-shot (default) -- resolve models, build the pipeline, diarize one file,
  exit. The original behavior; kept as the fallback and for local debugging.
* ``--serve`` -- resolve and build once, then answer newline-delimited JSON
  requests on stdin until stdin closes. Loading the model once instead of once
  per chunk removes the fixed ~5-7s import cost (plus CUDA context setup on
  GPU) from every chunk, and keeps the parent's GPU lock from being held during
  process spawn.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Protocol channel setup. MUST run before importing torch / pyannote / DiariZen.
#
# DiariZen and pyannote print to stdout unconditionally, on every single call
# ("Extracting segmentations.", "Extracting Embeddings.", "Clustering.", plus
# "Loaded configuration: {...}" at construction), and we do not patch the
# vendored third-party tree. That noise would corrupt a line-delimited JSON
# protocol on fd 1.
#
# So: dup fd 1 to a private fd and use THAT as the protocol channel, then point
# fd 1 -- and sys.stdout -- at fd 2. Every library print now lands in the
# parent's stderr sink (a file, never a pipe; see _DiarizenWorker) while our
# frames travel on the private fd. os.dup2 is used rather than
# contextlib.redirect_stdout because it also captures writes from C/C++
# extension code (torch, onnxruntime), which never goes through sys.stdout.
# --------------------------------------------------------------------------
import os
import sys

_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_PROTOCOL = os.fdopen(_PROTOCOL_FD, "w", encoding="utf-8", buffering=1)

import argparse  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

_T_IMPORT0 = time.perf_counter()
import toml  # noqa: E402
import torch  # noqa: E402
from diarizen.pipelines.inference import DiariZenPipeline  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

IMPORT_MS = int((time.perf_counter() - _T_IMPORT0) * 1000)

# Exit code used by the parent-death watchdog, distinct from any exit status
# the interpreter itself produces, so it is identifiable in a stderr log.
_EXIT_ORPHANED = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    )
    parser.add_argument("--model-dir")
    parser.add_argument(
        "--embedding-model",
        default="pyannote/wespeaker-voxceleb-resnet34-LM",
    )
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--cache-dir")
    # Required only in one-shot mode; in --serve mode the paths arrive per
    # request on stdin.
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    # Inference knobs. All default to None meaning "leave the model's own
    # config.toml alone" -- see build_config_parse.
    parser.add_argument("--segmentation-step", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--apply-median-filtering",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="stay alive and answer JSON requests on stdin until EOF",
    )
    args = parser.parse_args()
    if not args.serve and (not args.input or not args.output):
        parser.error("--input and --output are required unless --serve is set")
    return args


def _emit(frame: dict) -> None:
    """Write one protocol frame. json.dumps never emits a raw newline (control
    characters are escaped), so one line is always exactly one frame."""
    _PROTOCOL.write(json.dumps(frame, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------
def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the model directory and embedding checkpoint, downloading from
    the Hub only when a local path was not supplied or does not exist."""
    model_dir = None
    if args.model_dir:
        candidate = Path(args.model_dir).expanduser()
        if candidate.is_dir():
            model_dir = candidate

    embedding_path = None
    if args.embedding_model_path:
        candidate = Path(args.embedding_model_path).expanduser()
        if candidate.is_file():
            embedding_path = candidate

    local_files_only = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    if model_dir is None:
        model_dir = Path(
            snapshot_download(
                repo_id=args.model,
                cache_dir=args.cache_dir,
                local_files_only=local_files_only,
            )
        )
    if embedding_path is None:
        embedding_path = Path(
            hf_hub_download(
                repo_id=args.embedding_model,
                filename="pytorch_model.bin",
                cache_dir=args.cache_dir,
                local_files_only=local_files_only,
            )
        )
    return model_dir.resolve(), embedding_path.resolve()


def preflight_embedding(embedding_path: Path) -> None:
    """Fail early, and legibly, on an embedding path that would be routed to
    the ONNX backend.

    pyannote's PretrainedSpeakerEmbedding factory dispatches on SUBSTRINGS of
    the path, testing "pyannote" before "wespeaker". The production path is an
    HF cache blob under `models--pyannote--wespeaker-voxceleb-resnet34-LM/`,
    which contains "pyannote" and therefore takes the torch backend. But a
    plausible-looking local filename such as `wespeaker-voxceleb-resnet34-LM.bin`
    matches only "wespeaker" and gets sent to the ONNX class, which cannot load
    a .bin and raises a bare ImportError about onnxruntime.

    That failure happens during construction, i.e. during worker spawn while
    the parent holds the GPU lock, so it is worth naming precisely.
    """
    path_str = str(embedding_path)
    if "pyannote" in path_str or "wespeaker" not in path_str:
        return
    raise RuntimeError(
        f"embedding path {path_str!r} contains 'wespeaker' but not 'pyannote', "
        "so pyannote.audio will route it to its ONNX backend, which cannot "
        "load a PyTorch .bin checkpoint. Point --embedding-model-path at the "
        "HuggingFace cache blob (its path contains 'models--pyannote--...') or "
        "omit it to let the Hub resolve it."
    )


def build_config_parse(
    model_dir: Path, args: argparse.Namespace
) -> Optional[dict]:
    """Return a `config_parse` override dict, or None to leave the model's own
    config.toml untouched.

    DiariZenPipeline.__init__ REPLACES config["inference"]["args"] and
    config["clustering"]["args"] wholesale when config_parse is supplied, so a
    partial dict raises KeyError on the first key it does not carry. We
    therefore load the model's own config.toml -- the single source of truth for
    all of these keys -- and mutate only what was explicitly overridden. A
    future model shipping different defaults is then picked up for free.

    Returning None when nothing was overridden is deliberate: it preserves the
    exact original code path (no config_parse at all) for every config that
    does not set a knob.
    """
    overrides = {
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return None

    config = toml.load((model_dir / "config.toml").as_posix())
    inference_args = dict(config["inference"]["args"])
    inference_args.update(overrides)
    return {
        "inference": {"args": inference_args},
        # Passed through unchanged, but it MUST be present: __init__ reads
        # config["clustering"]["args"] from config_parse once config_parse is
        # non-None, and the VBx branch needs all nine of its keys.
        "clustering": {"args": dict(config["clustering"]["args"])},
    }


def build_pipeline(
    args: argparse.Namespace, model_dir: Path, embedding_path: Path
) -> DiariZenPipeline:
    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir,
        embedding_model=str(embedding_path),
        config_parse=build_config_parse(model_dir, args),
    )

    # DiariZenPipeline hardcodes cuda:0 (or cpu) internally and takes no device
    # argument. That happens to be correct for us because the parent remaps
    # CUDA_VISIBLE_DEVICES so its chosen physical GPU is this process's index 0.
    # Move explicitly anyway: if the child unexpectedly landed on CPU, the
    # hardcoded ternary would silently degrade to CPU and run ~100x slower with
    # no error at all. The startup frame reports where we actually ended up.
    device = torch.device(args.device if args.device.startswith("cuda") else "cpu")
    try:
        pipeline.to(device)
    except Exception as exc:  # noqa: BLE001
        # Construction already succeeded, so the models are on whatever device
        # DiariZen picked; a failed move is worth a loud log, not a hard stop.
        print(f"diarizen_to_device_failed {type(exc).__name__}: {exc}", flush=True)

    if args.num_speakers is not None:
        pipeline.min_speakers = args.num_speakers
        pipeline.max_speakers = args.num_speakers
    else:
        if args.min_speakers is not None:
            pipeline.min_speakers = args.min_speakers
        if args.max_speakers is not None:
            pipeline.max_speakers = args.max_speakers

    return pipeline


def startup_info(pipeline: DiariZenPipeline, args: argparse.Namespace) -> dict:
    """Facts worth having in the log when a run turns out slow."""
    info: dict[str, Any] = {
        "import_ms": IMPORT_MS,
        "requested_device": args.device,
        "min_speakers": pipeline.min_speakers,
        "max_speakers": pipeline.max_speakers,
    }
    try:
        info["device"] = str(pipeline.device)
        # The ONLY direct confirmation that a segmentation_step override landed:
        # the pipeline stores it multiplied by seg_duration, so 0.25 * 16 = 4.0.
        info["effective_step_seconds"] = float(pipeline._segmentation.step)
        info["embedding_class"] = type(pipeline._embedding).__name__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        pass
    return info


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def run_once(
    pipeline: DiariZenPipeline, input_path: str, output_path: str
) -> dict:
    """Diarize one file and write the segments JSON. Returns timings."""
    t0 = time.perf_counter()
    annotation = pipeline(input_path, sess_name=Path(input_path).stem)
    # NOTE: this includes the torchaudio.load of the input wav, which happens
    # inside DiariZenPipeline.__call__ and cannot be separated without forking
    # the vendored implementation.
    infer_ms = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    segments = [
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        }
        for turn, _track, speaker in annotation.itertracks(yield_label=True)
    ]
    Path(output_path).write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    postprocess_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "infer_ms": infer_ms,
        "postprocess_ms": postprocess_ms,
        "n_segments": len(segments),
    }


# ----------------------------------------------------------------------
# serve mode
# ----------------------------------------------------------------------
def _start_orphan_watchdog() -> None:
    """Exit if our parent disappears.

    Belt-and-braces only: the real guarantee is that stdin hits EOF when the
    parent dies, because the kernel closes the parent's pipe fd regardless of
    HOW it died (ray.kill, SIGTERM from mp.Pool.terminate, SIGKILL). This
    thread covers the residual case where the write end somehow outlives the
    parent -- e.g. inherited by another process.

    Uses os.getppid() rather than psutil, which is not installed in either
    environment.
    """
    def watch() -> None:
        while True:
            time.sleep(5.0)
            if os.getppid() == 1:
                print("diarizen_worker_orphaned exiting", flush=True)
                os._exit(_EXIT_ORPHANED)

    threading.Thread(target=watch, daemon=True).start()


def serve(args: argparse.Namespace) -> int:
    """Build the pipeline once, then answer one request per stdin line.

    Terminates on stdin EOF. This is the entire orphan defense and it is
    sufficient: ray.kill(no_restart=True) and mp.Pool.terminate() both run no
    user code in the parent, so nothing on the parent side can be relied upon
    to clean us up. Iterating sys.stdin directly (rather than select-ing on
    other fds) is what makes EOF unmissable.
    """
    t0 = time.perf_counter()
    model_dir, embedding_path = resolve_paths(args)
    resolve_ms = int((time.perf_counter() - t0) * 1000)
    preflight_embedding(embedding_path)

    t0 = time.perf_counter()
    pipeline = build_pipeline(args, model_dir, embedding_path)
    load_ms = int((time.perf_counter() - t0) * 1000)

    _start_orphan_watchdog()
    info = startup_info(pipeline, args)
    info.update({"resolve_ms": resolve_ms, "load_ms": load_ms})
    _emit({"event": "ready", "startup": info})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request["id"]
        except Exception:  # noqa: BLE001
            # An unparseable request means the streams are out of sync; there
            # is no id to answer with, so let the parent's read time out and
            # respawn us rather than guessing.
            print(f"diarizen_bad_request {line[:200]!r}", flush=True)
            continue

        try:
            timings = run_once(pipeline, request["input"], request["output"])
            _emit({"id": request_id, "ok": True, **timings})
        except Exception as exc:  # noqa: BLE001
            # One bad chunk must not cost a respawn: report and stay alive.
            _emit({
                "id": request_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
    return 0


# ----------------------------------------------------------------------
# one-shot mode
# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DiariZen was configured for {args.device}, but CUDA is unavailable"
        )
    if args.device.startswith("cuda"):
        # Ray normally exposes exactly one device to the actor, making cuda:0
        # the correct target even when the parent config used another index.
        # This matters more in --serve mode: the pipeline outlives many
        # requests, so pinning the default device up front keeps every later
        # implicit allocation on the intended card.
        torch.cuda.set_device(0)

    if args.serve:
        return serve(args)

    model_dir, embedding_path = resolve_paths(args)
    preflight_embedding(embedding_path)
    pipeline = build_pipeline(args, model_dir, embedding_path)
    run_once(pipeline, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
