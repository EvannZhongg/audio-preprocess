#!/usr/bin/env python3
"""Isolated DiariZen inference entrypoint used by PipelineV2.

This file intentionally imports DiariZen only inside the separate interpreter
selected by ``diarization.diarizen_python``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diarizen.pipelines.inference import DiariZenPipeline
from huggingface_hub import hf_hub_download, snapshot_download


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"DiariZen was configured for {args.device}, but CUDA is unavailable"
        )
    if args.device.startswith("cuda"):
        # Ray normally exposes exactly one device to the actor, making cuda:0
        # the correct target even when the parent config used another index.
        torch.cuda.set_device(0)

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
    pipeline = DiariZenPipeline(
        diarizen_hub=model_dir.resolve(),
        embedding_model=str(embedding_path.resolve()),
    )

    if args.num_speakers is not None:
        pipeline.min_speakers = args.num_speakers
        pipeline.max_speakers = args.num_speakers
    else:
        if args.min_speakers is not None:
            pipeline.min_speakers = args.min_speakers
        if args.max_speakers is not None:
            pipeline.max_speakers = args.max_speakers

    annotation = pipeline(args.input, sess_name=Path(args.input).stem)
    segments = [
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        }
        for turn, _track, speaker in annotation.itertracks(yield_label=True)
    ]
    Path(args.output).write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
