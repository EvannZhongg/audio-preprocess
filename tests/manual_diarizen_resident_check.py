"""Local CPU verification of the resident DiariZen worker.

Drives the real `pipeline_v2.steps.speaker_diarization.Diarizer` (not a stub)
against the CPU .venv-diarizen, checking the properties that actually carry
risk: resident==oneshot output, model loaded once, per-request timeout, crash
recovery, and no orphan on abrupt parent death.

Run with the MAIN env python (needs pandas/pydantic/soundfile), while
`diarizen_python` points at the .venv-diarizen interpreter.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIARIZEN_PY = (
    "/System/Volumes/Data/Users/zhongjiarui/Tencent_projects/audio-preprocess"
    "/local_adapter_v2/.venv-diarizen/bin/python"
)
MODEL_DIR = str(ROOT / "local_adapter_v2/models/diarizen-wavlm-large-s80-md-v2")
EMB = (
    "/Users/zhongjiarui/Tencent_projects/audio-preprocess/local_adapter_v2/models/hf"
    "/models--pyannote--wespeaker-voxceleb-resnet34-LM/blobs"
    "/366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from pipeline_v2.params import DiarizationParams  # noqa: E402
from pipeline_v2.steps.speaker_diarization import Diarizer  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def make_params(**over) -> DiarizationParams:
    base = dict(
        provider="diarizen",
        diarizen_python=DIARIZEN_PY,
        diarizen_runner=str(ROOT / "pipeline_v2/diarizen_runner.py"),
        diarizen_model_dir_cache=MODEL_DIR,
        diarizen_embedding_model_path=EMB,
        diarizen_segmentation_step=0.25,
    )
    base.update(over)
    return DiarizationParams(**base)


def load_clip(seconds: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(ROOT / "local_adapter_v2/test_audios/1_first_3min.wav")
    return np.asarray(wav[: int(seconds * sr)], dtype=np.float32), sr


def main() -> int:
    wave, sr = load_clip(20)
    print(f"clip: {len(wave)/sr:.0f}s @ {sr}Hz\n")

    # --- 1. resident: 3 requests, one spawn, identical output -------------
    print("1. resident worker, 3 sequential requests")
    d = Diarizer(make_params(), "cpu")
    check("__init__ spawns nothing", d._worker is None)
    frames = []
    for i in range(3):
        t0 = time.perf_counter()
        result = d.run(wave, sr)
        assert result is not None, "diarization returned None"
        df, _ = result
        frames.append(df)
        spawn = d._worker.spawn_ms if d._worker else -1
        print(f"   req{i}: {time.perf_counter()-t0:5.1f}s  segments={len(df)}  "
              f"spawn_ms={spawn}  speakers={df['speaker'].nunique()}")
    check("model loaded once (spawn_ms==0 after first)",
          d._worker is not None and d._worker.spawn_ms == 0)
    check("startup reports effective_step_seconds==4.0",
          d._worker.startup.get("effective_step_seconds") == 4.0,
          str(d._worker.startup.get("effective_step_seconds")))
    same = all(frames[0].equals(f) for f in frames[1:])
    check("all 3 requests byte-identical", same)
    resident_df = frames[0]

    # --- 2. one-shot must produce the same answer ------------------------
    print("\n2. one-shot mode (resident=false)")
    d2 = Diarizer(make_params(diarizen_resident=False), "cpu")
    t0 = time.perf_counter()
    result = d2.run(wave, sr)
    assert result is not None
    oneshot_df, _ = result
    print(f"   oneshot: {time.perf_counter()-t0:5.1f}s  segments={len(oneshot_df)}")
    check("resident output == one-shot output", resident_df.equals(oneshot_df),
          f"resident={len(resident_df)} oneshot={len(oneshot_df)}")
    check("one-shot spawns no resident worker", d2._worker is None)

    # --- 3. close() is clean and idempotent ------------------------------
    print("\n3. lifecycle")
    proc = d._worker._proc
    child_pid = proc.pid
    d.close()
    time.sleep(1.0)
    check("close() reaps the child", proc.poll() is not None,
          f"pid {child_pid} rc={proc.poll()}")
    d.close()  # must not raise
    check("close() is idempotent", True)
    check("reap_if_dead() is false with no worker", d.reap_if_dead() is False)

    # --- 4. worker recovers after its child is killed --------------------
    print("\n4. crash recovery")
    d3 = Diarizer(make_params(), "cpu")
    d3.run(wave, sr)
    killed = d3._worker._proc.pid
    os.kill(killed, 9)
    time.sleep(1.0)
    check("reap_if_dead() detects the corpse", d3.reap_if_dead() is True)
    result = d3.run(wave, sr)   # must transparently respawn
    check("next run() respawns and succeeds", result is not None)
    check("respawned child is a new pid",
          d3._worker is not None and d3._worker._proc.pid != killed)
    d3.close()

    # --- 5. per-request timeout ------------------------------------------
    print("\n5. per-request timeout")
    d4 = Diarizer(
        make_params(timeout_base_seconds=0, timeout_per_audio_second=0.0), "cpu"
    )
    # timeout is max(900, ...) in _run_diarizen, so force a tiny one directly
    # on the worker to exercise _readline_with_timeout without a 15-min wait.
    d4.run(wave, sr)
    worker = d4._worker
    t0 = time.perf_counter()
    try:
        worker.request(Path("/tmp/clip_timeout.wav"), Path("/tmp/to.json"), 0.5)
        timed_out = False
    except Exception as exc:
        timed_out = "timed out" in str(exc)
    elapsed = time.perf_counter() - t0
    check("request() honors its deadline", timed_out and elapsed < 5.0,
          f"{elapsed:.2f}s")
    d4.close()

    # --- 6. no orphan when the parent is SIGKILLed -----------------------
    print("\n6. orphan safety (parent SIGKILL, worse than ray.kill)")
    helper = f'''
import os, sys, time
sys.path.insert(0, {str(ROOT)!r})
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, soundfile as sf
from pipeline_v2.params import DiarizationParams
from pipeline_v2.steps.speaker_diarization import Diarizer
wav, sr = sf.read({str(ROOT / "local_adapter_v2/test_audios/1_first_3min.wav")!r})
d = Diarizer(DiarizationParams(
    provider="diarizen", diarizen_python={DIARIZEN_PY!r},
    diarizen_runner={str(ROOT / "pipeline_v2/diarizen_runner.py")!r},
    diarizen_model_dir_cache={MODEL_DIR!r},
    diarizen_embedding_model_path={EMB!r},
    diarizen_segmentation_step=0.25), "cpu")
d.run(np.asarray(wav[:int(5*sr)], dtype=np.float32), sr)
print(d._worker._proc.pid, flush=True)
time.sleep(300)
'''
    parent = subprocess.Popen([sys.executable, "-c", helper], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True)
    grandchild = int(parent.stdout.readline().strip())
    parent.kill()
    parent.wait(timeout=30)
    alive = True
    for _ in range(30):
        time.sleep(1.0)
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            alive = False
            break
    check("child dies with its parent (no GPU-resident orphan)", not alive,
          f"grandchild pid {grandchild}")
    if alive:
        os.kill(grandchild, 9)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
