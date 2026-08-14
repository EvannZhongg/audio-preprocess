"""Stage2Actor: one persistent Ray actor per pipe_slot, running remote ASR +
v1 post-processing on stage-1 output.

Unlike v2_stage_1 (decode -> VAD -> diarize -> segment -> export, from raw
audio), this actor:
  1. Receives one stage-1 chunk wav + its stage-1 segments (payload: a list
     of dicts with utt_id/origin_source/chunk_index/speaker_id/start/end,
     built by `main_v2_ray.collect_stage1_segments`).
  2. Runs `pipeline_v2.stage2.runner.run_stage2_asr` (remote ASR, unlocked,
     network I/O) followed by `run_stage2_postprocess` (v1 GPU-touching
     post-processing, cascading filters, write-back-not-delete semantics,
     serialized via `self._gpu_lock`).
  3. Writes the ASR text + quality fields back into the stage-1 sidecar JSON
     (found by deterministic path substitution -- audios/ -> jsons/, .wav ->
     .json -- so no extra path bookkeeping is needed anywhere).
  4. Returns flat Stage2SegmentRecord rows (one per stage-1 segment, 1:1, no
     deletions) for the driver to accumulate into stage2_segments_part
     parquet.
"""
from __future__ import annotations

import json
import os
import threading

import ray
import soundfile as sf

import logger
from logger import make_extra_tags
from pipeline_v2.params import Stage2Params
from pipeline_v2.stage2.models import load_stage2_models
from pipeline_v2.stage2.runner import run_stage2_asr, run_stage2_postprocess
from pipeline_v2.state import PIPELINE_VERSION
from pipeline_v2_ray.actors.base import PipelineActor, register_actor
from pipeline_v2_ray.config import RayConfig
from pipeline_v2_ray.result import FileResult


def _setup_env() -> None:
    """Redirect temp dirs onto the large scratch path. Mirrors v2_stage_1;
    needed here because Ray actor processes on a pre-started cluster do not
    inherit the driver's environment."""
    large_temp = f"{os.getcwd()}/TEMP"
    os.makedirs(large_temp, exist_ok=True)
    os.environ["LARGE_TEMP_DIR"] = large_temp
    os.environ["TMPDIR"] = large_temp
    os.environ["TEMP"] = large_temp
    os.environ["TMP"] = large_temp


def _json_rel_within_shard(chunk_audio_path: str, shard: str) -> str:
    """Derive the stage-1 sidecar json path (relative to the shard's own
    output dir, i.e. relative to the `output_folder` argument process_file
    receives -- which is already `<output_root>/<shard>`) from
    `chunk_audio_path` (`<shard>/audios/<bucket>/<file>.wav`, relative to
    OUTPUT ROOT), by deterministic substitution -- mirrors
    `pipeline_v2.steps.export.Exporter`'s audios/ <-> jsons/ layout, so no
    separate json path needs to be stored anywhere.
    """
    prefix = f"{shard}/"
    rel = chunk_audio_path[len(prefix):] if chunk_audio_path.startswith(prefix) else chunk_audio_path
    # After stripping the shard prefix, rel starts with "audios/..." (no leading
    # slash), so match without the leading slash too; otherwise the substitution
    # silently misses and we'd write back to the audio dir instead of jsons/.
    if rel.startswith("audios/"):
        rel = "jsons/" + rel[len("audios/"):]
    else:
        rel = rel.replace("/audios/", "/jsons/", 1)
    if rel.endswith(".wav"):
        rel = rel[: -len(".wav")] + ".json"
    return rel


@register_actor("v2_stage_2")
@ray.remote
class Stage2Actor(PipelineActor):
    def __init__(self, ray_config: RayConfig) -> None:
        _setup_env()

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Stage2Actor scheduled on a node without CUDA")

        # ray_config arrives fully resolved from the head node: every profile
        # already carries a parsed Stage2Params, so we read no config files.
        self._gpu_name = torch.cuda.get_device_name(0)
        profile, params = ray_config.resolve_params(self._gpu_name)
        self._profile_name = profile.name
        self._params: Stage2Params = params

        self._models = load_stage2_models(self._params)

        # Serialize only the GPU-touching post-processing stages
        # (alignment/ppl scorer/etc). The remote ASR call is plain HTTP I/O
        # and is deliberately run OUTSIDE this lock so multiple chunks'
        # ASR requests can be in flight concurrently within one actor
        # (see `max_concurrency` in the stage2 ray config).
        self._gpu_lock = threading.Lock()
        logger.info(
            f"ray_stage2_actor_ready gpu {self._gpu_name} profile {self._profile_name}"
        )

    def process_file(self, audio_path: str, output_folder: str,
                     relative_path: str, shard: str, payload=None) -> dict:
        """Process one stage-1 chunk wav; returns a serializable FileResult
        dict. `relative_path` is the chunk's `chunk_audio_path` (relative to
        the output root, includes the shard prefix); `payload` is the list of
        stage-1 segment dicts for that chunk. Never raises."""
        try:
            return self._process_chunk_inner(
                audio_path, output_folder, relative_path, shard, payload or []
            )
        except Exception as e:  # noqa: BLE001 - last-resort guard; keep the actor alive
            logger.error(f"ray_stage2_process_error file {audio_path} err {type(e).__name__}: {e}")
            return FileResult(
                audio_path, success=False, error=f"{type(e).__name__}: {e}"
            ).to_dict()

    def _process_chunk_inner(self, audio_path: str, output_folder: str,
                             relative_path: str, shard: str, stage1_segments: list) -> dict:
        log_tag = make_extra_tags(audio_file=relative_path, version=PIPELINE_VERSION)
        if not stage1_segments:
            return FileResult(audio_path, success=False, error="no stage1 segments").to_dict()

        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

        # Remote ASR is network I/O, not GPU work -- run it unlocked so
        # multiple in-flight chunks (up to max_concurrency) can all have
        # their ASR requests outstanding at the same time.
        asr_segments = run_stage2_asr(
            waveform, sample_rate, stage1_segments, self._params, self._models,
            logger=logger,
        )

        with self._gpu_lock:
            asr_segments = run_stage2_postprocess(
                waveform, sample_rate, asr_segments, self._params, self._models,
                logger=logger,
            )

        chunk_audio_path = relative_path  # for stage 2, relative_path IS the chunk's output-root-relative path
        origin_source = stage1_segments[0].get("origin_source")
        chunk_index = stage1_segments[0].get("chunk_index")

        records = [
            self._to_record(src, asr, chunk_audio_path, origin_source, shard, chunk_index)
            for src, asr in zip(stage1_segments, asr_segments)
        ]

        try:
            self._write_back_json(output_folder, chunk_audio_path, shard, stage1_segments, asr_segments)
        except Exception as e:  # noqa: BLE001 - json write-back is best-effort, must not fail the chunk
            logger.error(f"ray_stage2_writeback_failed {type(e).__name__}: {e}", extra=log_tag)

        return FileResult(
            audio_path, success=True, n_segments=len(records), segments=records,
        ).to_dict()

    @staticmethod
    def _to_record(src: dict, asr: dict, chunk_audio_path: str, origin_source, shard: str, chunk_index) -> dict:
        start = asr.get("start")
        end = asr.get("end")
        domain_info = asr.get("domain_info") if isinstance(asr.get("domain_info"), dict) else {}
        return {
            "utt_id": src.get("utt_id"),
            "source": chunk_audio_path,
            "origin_source": origin_source,
            "shard": shard,
            "pipeline_version": PIPELINE_VERSION,
            "chunk_index": chunk_index,
            "chunk_audio_path": chunk_audio_path,
            "start": start,
            "end": end,
            "seg_duration": (end - start) if (start is not None and end is not None) else None,
            "speaker_id": src.get("speaker_id"),
            "text": asr.get("text"),
            "language": asr.get("language"),
            "domain_text": domain_info.get("text_domain"),
            "domain_acoustic": domain_info.get("acoustic_domain"),
            "domain_speaker": domain_info.get("speaker_domain"),
            "speaking_rate": asr.get("speaking_rate"),
            "alignment_score": asr.get("alignment_score"),
            "ppl": asr.get("ppl"),
            "llm_text_score": asr.get("llm_quality"),
            "dropped_by_silence": bool(asr.get("dropped_by_silence", False)),
            "dropped_by_alignment": bool(asr.get("dropped_by_alignment", False)),
            "dropped_by_text_quality": bool(asr.get("dropped_by_text_quality", False)),
            "dropped_by_speaking_rate": bool(asr.get("dropped_by_speaking_rate", False)),
            "dropped_by_asr_validation": bool(asr.get("dropped_by_asr_validation", False)),
            "asr_wer": asr.get("asr_wer"),
            "asr_val_text": asr.get("asr_val_text"),
            "error": None,
        }

    @staticmethod
    def _write_back_json(output_folder: str, chunk_audio_path: str, shard: str,
                         stage1_segments: list, asr_segments: list) -> None:
        """Merge ASR text + quality fields into the stage-1 sidecar json's
        `sentences[i]`, matched by `utt_id`. Both lists are 1:1 with the
        stage-1 segment order, matched here by identity of source order."""
        json_rel = _json_rel_within_shard(chunk_audio_path, shard)
        json_path = os.path.join(output_folder, json_rel)
        if not os.path.exists(json_path):
            logger.error(f"ray_stage2_json_missing {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sentences = payload.get("sentences") or []
        by_utt = {s.get("utt_id"): s for s in sentences}
        for src, asr in zip(stage1_segments, asr_segments):
            sent = by_utt.get(src.get("utt_id"))
            if sent is None:
                continue
            domain_info = asr.get("domain_info") if isinstance(asr.get("domain_info"), dict) else asr.get("domain_info")
            sent["text"] = asr.get("text")
            sent["language"] = asr.get("language")
            sent["domain_info"] = domain_info
            sent["speaking_rate"] = asr.get("speaking_rate")
            sent["alignment_score"] = asr.get("alignment_score")
            sent["text_quality_info"] = {
                "ppl": asr.get("ppl"),
                "spell_score": asr.get("spell_score"),
                "llm_quality": asr.get("llm_quality"),
                "semantic_completeness": asr.get("semantic_completeness"),
                "tts_suitability": asr.get("tts_suitability"),
            }
            sent["dropped_by_speaking_rate"] = bool(asr.get("dropped_by_speaking_rate", False))
            sent["dropped_by_silence"] = bool(asr.get("dropped_by_silence", False))
            sent["dropped_by_alignment"] = bool(asr.get("dropped_by_alignment", False))
            sent["dropped_by_text_quality"] = bool(asr.get("dropped_by_text_quality", False))
            sent["dropped_by_asr_validation"] = bool(asr.get("dropped_by_asr_validation", False))
            sent["asr_wer"] = asr.get("asr_wer")
            sent["asr_val_text"] = asr.get("asr_val_text")
        payload["pipeline_version"] = PIPELINE_VERSION
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)
