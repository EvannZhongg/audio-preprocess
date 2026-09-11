from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Array:
    def __init__(self, values):
        if isinstance(values, _Array):
            values = values.values
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    @property
    def ndim(self):
        return 1

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, column = item
            assert isinstance(rows, slice)
            return _Array([row[column] for row in self.values[rows]])
        if isinstance(item, slice):
            return _Array(self.values[item])
        return self.values[item]

    def astype(self, _dtype, copy=True):
        return self


class _Matrix(_Array):
    @property
    def shape(self):
        return (
            len(self.values),
            len(self.values[0]) if self.values else 0,
        )

    @property
    def ndim(self):
        return 2


class _Tensor:
    def __init__(self, data):
        self.data = data

    def unsqueeze(self, axis: int):
        assert axis == 0
        return self


numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = _Array
numpy_stub.float32 = "float32"
numpy_stub.asarray = lambda data, dtype=None: (
    data if isinstance(data, (_Array, _Matrix)) else _Array(data)
)
numpy_stub.empty = lambda size, dtype=None: _Array([0.0] * size)
numpy_stub.mean = lambda data: (
    sum(data.values) / len(data.values)
    if isinstance(data, _Array)
    else sum(data) / len(data)
)
numpy_stub.isfinite = lambda value: value == value
numpy_stub.poly1d = lambda coefficients: lambda value: value
numpy_stub.floor = lambda value: int(value // 1)
numpy_stub.append = lambda left, right: _Array(left.values + right.values)

local_package = types.ModuleType("local_adapter_v2")
local_package.__path__ = [str(ROOT / "local_adapter_v2")]
local_pipeline_package = types.ModuleType("local_adapter_v2.local_pipeline")
local_pipeline_package.__path__ = [str(ROOT / "local_adapter_v2/local_pipeline")]
librosa_stub = types.ModuleType("librosa")
librosa_stub.resample = lambda *args, **kwargs: None
onnxruntime_stub = types.ModuleType("onnxruntime")
onnxruntime_stub.InferenceSession = object
torch_stub = types.ModuleType("torch")
torch_stub.from_numpy = lambda data: _Tensor(data)
torch_stub.device = lambda name: name

with patch.dict(
    sys.modules,
    {
        "local_adapter_v2": local_package,
        "local_adapter_v2.local_pipeline": local_pipeline_package,
        "numpy": numpy_stub,
        "librosa": librosa_stub,
        "onnxruntime": onnxruntime_stub,
        "torch": torch_stub,
    },
):
    local_config = load_module(
        "local_adapter_v2.local_pipeline.config",
        "local_adapter_v2/local_pipeline/config.py",
    )
    local_state = load_module(
        "local_adapter_v2.local_pipeline.state",
        "local_adapter_v2/local_pipeline/state.py",
    )
    local_metrics = load_module(
        "local_adapter_v2.local_pipeline.metrics",
        "local_adapter_v2/local_pipeline/metrics.py",
    )

pipeline_package = types.ModuleType("pipeline_v2")
pipeline_package.__path__ = [str(ROOT / "pipeline_v2")]
steps_package = types.ModuleType("pipeline_v2.steps")
steps_package.__path__ = [str(ROOT / "pipeline_v2/steps")]
params_stub = types.ModuleType("pipeline_v2.params")
params_stub.MetricsParams = object
state_stub = types.ModuleType("pipeline_v2.state")
state_stub.Segment = object
logger_stub = types.ModuleType("logger")
logger_stub.error = lambda *args, **kwargs: None
logger_stub.info = lambda *args, **kwargs: None
models_stub = types.ModuleType("models")
models_stub.__path__ = [str(ROOT / "models")]
dnsmos_stub = types.ModuleType("models.dnsmos")
dnsmos_stub.ComputeScore = object
brouhaha_stub = types.ModuleType("models.brouhaha_metrics")
brouhaha_stub.ComputeScore = object
models_stub.dnsmos = dnsmos_stub
models_stub.brouhaha_metrics = brouhaha_stub

with patch.dict(
    sys.modules,
    {
        "pipeline_v2": pipeline_package,
        "pipeline_v2.steps": steps_package,
        "pipeline_v2.params": params_stub,
        "pipeline_v2.state": state_stub,
        "numpy": numpy_stub,
        "librosa": librosa_stub,
        "logger": logger_stub,
        "models": models_stub,
        "models.dnsmos": dnsmos_stub,
        "models.brouhaha_metrics": brouhaha_stub,
    },
):
    native_metrics = load_module(
        "pipeline_v2.steps.metrics",
        "pipeline_v2/steps/metrics.py",
    )


class _DnsRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, sample_rate: int) -> float:
        self.calls.append((audio, sample_rate))
        return 3.5


class _BrouhahaResult:
    data = _Matrix([[0.0, 35.0, 45.0]])


class _BrouhahaRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, audio: dict) -> _BrouhahaResult:
        self.calls.append(audio)
        return _BrouhahaResult()


class LocalMetricsResamplingTests(unittest.TestCase):
    def make_scorer(self):
        cfg = local_config.MetricsConfig(
            dnsmos_enabled=False,
            brouhaha_enabled=False,
            strategy="none",
        )
        scorer = local_metrics.MetricsScorer(cfg, None, logging.getLogger(__name__))
        scorer.dnsmos = _DnsRecorder()
        scorer.brouhaha = _BrouhahaRecorder()
        return scorer

    def test_resamples_whole_waveform_once_and_shares_16k_segment(self):
        scorer = self.make_scorer()
        segments = [
            local_state.Segment(index="0", start=0.25, end=0.75, speaker="A"),
            local_state.Segment(index="1", start=1.0, end=1.5, speaker="A"),
        ]
        waveform = _Array(range(48000))
        waveform_16k = _Array(range(32000))

        with patch.object(
            local_metrics.librosa,
            "resample",
            return_value=waveform_16k,
        ) as resample:
            result = scorer.run(segments, waveform, 24000)

        self.assertIs(result, segments)
        resample.assert_called_once_with(
            waveform,
            orig_sr=24000,
            target_sr=16000,
        )
        self.assertEqual(len(scorer.dnsmos.calls), 2)
        self.assertEqual(len(scorer.brouhaha.calls), 2)

        expected = [
            waveform_16k[4000:12000].values,
            waveform_16k[16000:24000].values,
        ]
        for idx, expected_values in enumerate(expected):
            dns_chunk, dns_sr = scorer.dnsmos.calls[idx]
            brouhaha_audio = scorer.brouhaha.calls[idx]
            self.assertEqual(dns_sr, 16000)
            self.assertEqual(brouhaha_audio["sample_rate"], 16000)
            self.assertEqual(dns_chunk.values, expected_values)
            self.assertEqual(
                brouhaha_audio["waveform"].data.values,
                expected_values,
            )

    def test_empty_segment_list_does_not_resample(self):
        scorer = self.make_scorer()
        with patch.object(local_metrics.librosa, "resample") as resample:
            self.assertEqual(scorer.run([], _Array(range(24000)), 24000), [])
        resample.assert_not_called()

    def test_disabled_metrics_do_not_resample(self):
        cfg = local_config.MetricsConfig(
            dnsmos_enabled=False,
            brouhaha_enabled=False,
            strategy="none",
        )
        scorer = local_metrics.MetricsScorer(
            cfg,
            None,
            logging.getLogger(__name__),
        )
        segments = [
            local_state.Segment(index="0", start=0.25, end=0.75, speaker="A")
        ]
        with patch.object(local_metrics.librosa, "resample") as resample:
            result = scorer.run(segments, _Array(range(24000)), 24000)
        resample.assert_not_called()
        self.assertIsNone(result[0].dnsmos)
        self.assertEqual(result[0].c50, cfg.fixed_c50_threshold)
        self.assertEqual(result[0].snr, cfg.fixed_snr_threshold)


class NativeMetricsResamplingTests(unittest.TestCase):
    def test_native_pipeline_resamples_once_and_shares_segment(self):
        dns = _DnsRecorder()
        brouhaha = _BrouhahaRecorder()
        scorer = object.__new__(native_metrics.MetricsScorer)
        scorer.params = SimpleNamespace(
            fixed_c50_threshold=40.0,
            fixed_snr_threshold=30.0,
            fixed_dnsmos_threshold=2.5,
            strategy="fixed",
        )
        scorer.dnsmos_compute_score = (
            lambda audio, sample_rate, _personalized: {
                "OVRL": dns(audio, sample_rate)
            }
        )
        scorer.brouhaha_metric = lambda audio, sample_rate: (
            brouhaha({"waveform": audio, "sample_rate": sample_rate}),
            None,
        )[1] or (45.0, 35.0)

        segments = [
            SimpleNamespace(start=0.25, end=0.75, dnsmos=None, c50=None, snr=None),
            SimpleNamespace(start=1.0, end=1.5, dnsmos=None, c50=None, snr=None),
        ]
        waveform = _Array(range(48000))
        waveform_16k = _Array(range(32000))

        with patch.object(
            native_metrics.librosa,
            "resample",
            return_value=waveform_16k,
        ) as resample:
            result = scorer.run(segments, waveform, 24000)

        self.assertIs(result[0], segments[0])
        self.assertEqual(len(result), 2)
        resample.assert_called_once_with(
            waveform,
            orig_sr=24000,
            target_sr=16000,
        )
        expected = [
            waveform_16k[4000:12000].values,
            waveform_16k[16000:24000].values,
        ]
        for idx, expected_values in enumerate(expected):
            self.assertEqual(dns.calls[idx][0].values, expected_values)
            self.assertEqual(dns.calls[idx][1], 16000)
            self.assertEqual(
                brouhaha.calls[idx]["waveform"].values,
                expected_values,
            )
            self.assertEqual(brouhaha.calls[idx]["sample_rate"], 16000)


class LocalPipelineOrderTests(unittest.TestCase):
    def test_metrics_runs_before_asr(self):
        tree = ast.parse(
            (ROOT / "local_adapter_v2/local_pipeline/pipeline.py").read_text(
                encoding="utf-8"
            )
        )
        local_pipeline = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LocalPipeline"
        )
        run_method = next(
            node
            for node in local_pipeline.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        calls = sorted(
            (
                node.lineno,
                ast.unparse(node),
            )
            for node in ast.walk(run_method)
            if isinstance(node, ast.Call)
        )
        metrics_index = next(
            index
            for index, (_line, call) in enumerate(calls)
            if call.startswith("self.metrics.run(")
        )
        asr_index = next(
            index
            for index, (_line, call) in enumerate(calls)
            if call.startswith("self.asr.run(")
        )
        self.assertLess(metrics_index, asr_index)

if __name__ == "__main__":
    unittest.main()
