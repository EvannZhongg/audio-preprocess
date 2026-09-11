from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid pipeline_v2/__init__.py importing the model-heavy full pipeline just to
# exercise the parameter/state/segmenter units.
package = types.ModuleType("pipeline_v2")
package.__path__ = [str(ROOT / "pipeline_v2")]
sys.modules.setdefault("pipeline_v2", package)
params_mod = load_module("pipeline_v2.params", "pipeline_v2/params.py")
state_mod = load_module("pipeline_v2.state", "pipeline_v2/state.py")

# Segmenter only needs this class for its constructor annotation in these tests.
vad_stub = types.ModuleType("models.vad")
vad_stub.SileroVAD = object
sys.modules.setdefault("models.vad", vad_stub)
segment_mod = load_module(
    "pipeline_v2.steps.segment", "pipeline_v2/steps/segment.py"
)

PipelineParams = params_mod.PipelineParams
SegmenterParams = params_mod.SegmenterParams
Segment = state_mod.Segment
SegmentRecord = state_mod.SegmentRecord
Segmenter = segment_mod.Segmenter


class _UnusedVad:
    def _get_speech_timestamps_wrapper(self, _audio, _sample_rate):
        raise AssertionError("long-segment splitting was not expected")


def make_segmenter(**overrides) -> Segmenter:
    values = {
        "merge_gap": 0.8,
        "min_segment_length": 2.0,
        "max_segment_length": 30.0,
        "intra_similarity_threshold": 0.70,
        "grace_period_start": 0.08,
        "grace_period_end": 0.02,
    }
    values.update(overrides)
    return Segmenter(SegmenterParams(**values), _UnusedVad())


def seg(start: float, end: float, speaker: str = "SPEAKER_00") -> Segment:
    return Segment(
        index="old",
        start=start,
        end=end,
        speaker=speaker,
        reference_embedding=np.asarray([[1.0, 0.0]], dtype=np.float32),
        min_similarity=0.9,
    )


def diarization(*rows: tuple[float, float, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": None,
                "label": None,
                "start": start,
                "end": end,
                "speaker": speaker,
            }
            for start, end, speaker in rows
        ]
    )


class ConfigMappingTests(unittest.TestCase):
    def test_existing_config_keeps_original_behavior(self):
        parsed = PipelineParams.from_config(str(ROOT / "configs/config_for_a10.json"))
        self.assertEqual(parsed.diarization.provider, "pyannote")
        self.assertAlmostEqual(parsed.standardization.min_audio_seconds, 600.0)
        self.assertFalse(parsed.segmenter.block_merge_across_foreign)
        self.assertFalse(parsed.segmenter.drop_segments_with_foreign_speech)
        self.assertFalse(parsed.segmenter.trim_foreign_at_boundary)
        self.assertFalse(parsed.segmenter.split_around_foreign)
        self.assertFalse(parsed.segmenter.enforce_min_after_grace)
        # None on every DiariZen knob == inherit the model's own config.toml, so
        # configs predating these knobs keep byte-identical behavior.
        self.assertIsNone(parsed.diarization.diarizen_segmentation_step)
        self.assertIsNone(parsed.diarization.diarizen_batch_size)
        self.assertIsNone(parsed.diarization.diarizen_apply_median_filtering)

    def test_segmentation_step_is_range_checked(self):
        # Out-of-range steps must fail at config-parse time. Upstream
        # Inference.__init__ would otherwise raise during resident-worker spawn,
        # i.e. while the caller holds the GPU lock.
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(Exception):
                params_mod.DiarizationParams(
                    provider="diarizen", diarizen_segmentation_step=bad
                )
        for good in (0.1, 0.25, 1.0):
            params_mod.DiarizationParams(
                provider="diarizen", diarizen_segmentation_step=good
            )

    def test_native_optimized_config_maps_to_diarizen_and_guards(self):
        parsed = PipelineParams.from_config(
            str(ROOT / "configs/config_pipeline_v2_diarizen_tts_clean_v2.json")
        )
        self.assertEqual(parsed.diarization.provider, "diarizen")
        self.assertAlmostEqual(parsed.standardization.min_audio_seconds, 0.5)
        self.assertEqual(
            parsed.diarization.diarizen_model,
            "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        )
        self.assertEqual(parsed.device_name, "cuda:0")
        self.assertIsNone(parsed.diarization.num_speakers)
        self.assertIsNone(parsed.diarization.min_speakers)
        self.assertIsNone(parsed.diarization.max_speakers)
        self.assertTrue(parsed.source_separation.enable)
        # SMRU is the recommended separator and what the original PipelineV2
        # configs use; the optimized config previously declared provider "uvr"
        # with an EMPTY smru block, so assert the block is actually populated.
        self.assertEqual(parsed.source_separation.provider, "smru")
        self.assertEqual(
            parsed.source_separation.smru_conf,
            {
                "conf": "ckpts/denoise_derev_48k_SFI_E128.yaml",
                "chunk_size": 12,
                "valid_size": 8,
                "overlap": 1,
                "batch_size": 4,
            },
        )
        # UVR stays configured so the provider is a one-word A/B switch.
        self.assertEqual(
            parsed.source_separation.uvr_conf["model_path"],
            "ckpts/UVR-MDX-NET-Inst_HQ_3.onnx",
        )
        # segmentation_step is the speed/accuracy dial; the other two knobs are
        # deliberately absent from the json so they inherit the model's own
        # config.toml (None == "don't override").
        self.assertAlmostEqual(parsed.diarization.diarizen_segmentation_step, 0.25)
        self.assertIsNone(parsed.diarization.diarizen_batch_size)
        self.assertIsNone(parsed.diarization.diarizen_apply_median_filtering)
        self.assertTrue(parsed.diarization.diarizen_resident)
        self.assertAlmostEqual(parsed.embedding_refinement.inter_similarity_threshold, 0.55)
        self.assertEqual(parsed.embedding_refinement.refinement_batch_size, 16)
        self.assertTrue(parsed.segmenter.block_merge_across_foreign)
        self.assertTrue(parsed.segmenter.drop_segments_with_foreign_speech)
        self.assertTrue(parsed.segmenter.trim_foreign_at_boundary)
        self.assertTrue(parsed.segmenter.split_around_foreign)
        self.assertTrue(parsed.segmenter.enforce_min_after_grace)
        self.assertAlmostEqual(parsed.segmenter.foreign_relabel_ratio, 0.8)
        self.assertAlmostEqual(parsed.metrics.fixed_dnsmos_threshold, 2.8)
        self.assertAlmostEqual(parsed.metrics.fixed_snr_threshold, 30.0)

    def test_diarizen_uses_pipeline_gpu_selection(self):
        # Stub pyannote.audio so the module can be loaded without constructing
        # either diarization backend.
        pyannote_pkg = types.ModuleType("pyannote")
        pyannote_audio = types.ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = object
        pyannote_pkg.audio = pyannote_audio
        old_pyannote = sys.modules.get("pyannote")
        old_pyannote_audio = sys.modules.get("pyannote.audio")
        sys.modules["pyannote"] = pyannote_pkg
        sys.modules["pyannote.audio"] = pyannote_audio
        try:
            diarization_mod = load_module(
                "_test_pipeline_v2_speaker_diarization",
                "pipeline_v2/steps/speaker_diarization.py",
            )
        finally:
            if old_pyannote is None:
                sys.modules.pop("pyannote", None)
            else:
                sys.modules["pyannote"] = old_pyannote
            if old_pyannote_audio is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = old_pyannote_audio

        parsed = PipelineParams.from_config(
            str(ROOT / "configs/config_pipeline_v2_diarizen_tts_clean_v2.json")
        )
        diarizer = diarization_mod.Diarizer(parsed.diarization, "cuda:2")
        # Constructing a Diarizer must remain completely side-effect-free: this
        # runs on machines with neither .venv-diarizen nor CUDA, so the resident
        # worker has to be spawned lazily on the first run().
        self.assertIsNone(diarizer._worker)

        command = diarizer._diarizen_command(Path("/tmp/in.wav"), Path("/tmp/out.json"))
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        # Knobs set in the json are forwarded; knobs left unset must NOT appear,
        # so the child inherits the model's own config.toml.
        self.assertEqual(
            command[command.index("--segmentation-step") + 1], "0.25"
        )
        self.assertNotIn("--batch-size", command)
        self.assertNotIn("--apply-median-filtering", command)
        self.assertNotIn("--no-apply-median-filtering", command)
        # One-shot argv must not carry --serve; serve mode must.
        self.assertNotIn("--serve", command)
        self.assertIn(
            "--serve",
            diarizer._diarizen_command(
                Path("/tmp/in.wav"), Path("/tmp/out.json"), serve=True
            ),
        )

        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,7,9"
        try:
            self.assertEqual(diarizer._diarizen_env()["CUDA_VISIBLE_DEVICES"], "9")
        finally:
            if old_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible

    def test_median_filter_disabled_uses_paired_negative_flag(self):
        # The knob is tri-state (None / True / False), so an explicit False has
        # to be expressible on the command line.
        pyannote_pkg = types.ModuleType("pyannote")
        pyannote_audio = types.ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = object
        pyannote_pkg.audio = pyannote_audio
        sys.modules["pyannote"] = pyannote_pkg
        sys.modules["pyannote.audio"] = pyannote_audio
        try:
            diarization_mod = load_module(
                "_test_pipeline_v2_speaker_diarization_mf",
                "pipeline_v2/steps/speaker_diarization.py",
            )
        finally:
            sys.modules.pop("pyannote", None)
            sys.modules.pop("pyannote.audio", None)

        params = params_mod.DiarizationParams(
            provider="diarizen", diarizen_apply_median_filtering=False
        )
        command = diarization_mod.Diarizer(params, "cpu")._diarizen_command(
            Path("/tmp/in.wav"), Path("/tmp/out.json")
        )
        self.assertIn("--no-apply-median-filtering", command)
        self.assertNotIn("--apply-median-filtering", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_local_adapter_config_schema_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "unsupported pipeline config schema"
        ):
            PipelineParams.from_config(
                str(ROOT / "local_adapter_v2/configs/tts_clean_v2.json")
            )


class SegmentGuardTests(unittest.TestCase):
    def test_original_mode_still_merges_across_foreign_speech(self):
        segmenter = make_segmenter(grace_period_start=0.0)
        result = segmenter.run(
            [seg(0.0, 3.0), seg(3.5, 6.0)],
            np.zeros(24000 * 8, dtype=np.float32),
            24000,
            diarize_df=diarization(
                (0.0, 3.0, "SPEAKER_00"),
                (3.1, 3.4, "SPEAKER_01"),
                (3.5, 6.0, "SPEAKER_00"),
            ),
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 6.02)

    def test_foreign_speech_blocks_merge(self):
        segmenter = make_segmenter(block_merge_across_foreign=True)
        result = segmenter.run(
            [seg(0.0, 3.0), seg(3.5, 6.0)],
            np.zeros(24000 * 8, dtype=np.float32),
            24000,
            diarize_df=diarization(
                (0.0, 3.0, "SPEAKER_00"),
                (3.1, 3.4, "SPEAKER_01"),
                (3.5, 6.0, "SPEAKER_00"),
            ),
        )
        self.assertEqual([(s.start, s.end) for s in result], [(0.0, 3.02), (3.42, 6.02)])

    def test_tolerance_allows_tiny_timestamp_overlap(self):
        segmenter = make_segmenter(block_merge_across_foreign=True)
        result = segmenter.run(
            [seg(0.0, 3.0), seg(3.5, 6.0)],
            np.zeros(24000 * 8, dtype=np.float32),
            24000,
            diarize_df=diarization((3.20, 3.25, "SPEAKER_01")),
        )
        self.assertEqual(len(result), 1)

    def test_local_pollution_is_split_and_clean_remainders_are_kept(self):
        segmenter = make_segmenter(
            block_merge_across_foreign=True,
            drop_segments_with_foreign_speech=True,
            trim_foreign_at_boundary=True,
            split_around_foreign=True,
            enforce_min_after_grace=True,
        )
        result = segmenter.run(
            [seg(0.0, 8.0)],
            np.zeros(24000 * 10, dtype=np.float32),
            24000,
            diarize_df=diarization((3.0, 4.0, "SPEAKER_01")),
        )
        self.assertEqual([(s.index, s.start, s.end) for s in result], [
            ("0", 0.0, 3.02),
            ("1", 3.92, 8.02),
        ])

    def test_dominant_foreign_overlap_is_kept_as_relabel_case(self):
        segmenter = make_segmenter(
            drop_segments_with_foreign_speech=True,
            foreign_relabel_ratio=0.8,
            enforce_min_after_grace=True,
        )
        result = segmenter.run(
            [seg(0.0, 5.0)],
            np.zeros(24000 * 6, dtype=np.float32),
            24000,
            diarize_df=diarization((0.0, 4.5, "SPEAKER_01")),
        )
        self.assertEqual(len(result), 1)

    def test_boundary_trim_happens_before_final_min_length_check(self):
        segmenter = make_segmenter(
            trim_foreign_at_boundary=True,
            enforce_min_after_grace=True,
        )
        result = segmenter.run(
            [seg(0.0, 2.2)],
            np.zeros(24000 * 4, dtype=np.float32),
            24000,
            diarize_df=diarization((0.0, 0.3, "SPEAKER_01")),
        )
        self.assertEqual(result, [])

    def test_segment_record_schema_is_unchanged(self):
        self.assertEqual(
            list(SegmentRecord.__annotations__),
            [
                "utt_id",
                "source",
                "shard",
                "pipeline_version",
                "chunk_index",
                "chunk_audio_path",
                "sample_rate",
                "chunk_duration",
                "speaker_id",
                "speaker_min_similarity",
                "start",
                "end",
                "seg_duration",
                "dnsmos",
                "c50",
                "snr",
                "error",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(Segment)],
            [
                "index",
                "start",
                "end",
                "speaker",
                "reference_embedding",
                "min_similarity",
                "dnsmos",
                "c50",
                "snr",
            ],
        )


if __name__ == "__main__":
    unittest.main()
