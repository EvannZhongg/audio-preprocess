#!/usr/bin/env bash
# Compare PipelineV2 modes on the same audio, three ways by default:
#
#   baseline   original flow, pyannote diarizer
#   modelswap  original flow, DiariZen diarizer        <- isolates the MODEL
#   optimized  local_adapter_v2 flow, DiariZen         <- isolates the FLOW
#
# The three configs are matched on everything not under test (chunking,
# separation provider, brouhaha path, thresholds), so baseline->modelswap
# attributes purely to the diarization model and modelswap->optimized purely
# to the segmentation guards.
#
# Run from the repo root on the GPU box:
#     bash scripts/run_ab_compare.sh [AUDIO_DIR] [OUTPUT_ROOT] [MODES]
#
#     bash scripts/run_ab_compare.sh audios_test ab_out
#     bash scripts/run_ab_compare.sh audios_test ab_out "baseline modelswap"
set -euo pipefail

AUDIO="${1:-audios_test}"
OUT="${2:-ab_out}"
MODES="${3:-baseline modelswap optimized}"

# Honor an explicit PYTHON, else prefer python3 (python may be absent outside
# an activated conda env).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY=python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else echo "no python interpreter found; set PYTHON=/path/to/python" >&2; exit 1
  fi
fi

config_for () {
  case "$1" in
    baseline)  echo "configs/config_pipeline_v2_baseline_ab.json" ;;
    modelswap) echo "configs/config_pipeline_v2_diarizen_swap_ab.json" ;;
    optimized) echo "configs/config_pipeline_v2_diarizen_tts_clean_v2.json" ;;
    *) echo "" ;;
  esac
}

# Offline mode is not optional. Every model here is already on disk, but the
# loaders still try to revalidate against the Hub, and a stalled revalidation
# is silent: TTS_PIPELINE_OPTIMIZATION.md records brouhaha hanging at zero CPU
# for 27 minutes this way. Also keeps the DiariZen worker from resolving HF
# paths over the network at spawn, which happens under the GPU lock.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ ! -d "$AUDIO" ]; then
  echo "audio directory not found: $AUDIO" >&2
  exit 1
fi

NEEDS_DIARIZEN=0
for mode in $MODES; do
  cfg="$(config_for "$mode")"
  if [ -z "$cfg" ]; then
    echo "unknown mode: $mode (want: baseline modelswap optimized)" >&2
    exit 1
  fi
  [ -f "$cfg" ] || { echo "missing config: $cfg" >&2; exit 1; }
  [ "$mode" = "baseline" ] || NEEDS_DIARIZEN=1
done

# The DiariZen worker needs its own interpreter. Fail loudly now rather than
# after the baseline run has already burned an hour.
if [ "$NEEDS_DIARIZEN" = "1" ]; then
  DIARIZEN_PY="$($PY - <<'PYEOF'
import json
cfg = json.load(open("configs/config_pipeline_v2_diarizen_tts_clean_v2.json"))
print(cfg.get("diarizen", {}).get("python_executable", ".venv-diarizen/bin/python"))
PYEOF
)"
  if [ ! -x "$DIARIZEN_PY" ]; then
    echo "DiariZen interpreter not executable: $DIARIZEN_PY" >&2
    echo "run scripts/install_pipeline_v2_diarizen.sh, or set" >&2
    echo "diarizen.python_executable in the diarizen configs" >&2
    exit 1
  fi
fi

mkdir -p logs "$OUT"
echo "audio=$AUDIO  output=$OUT  modes=$MODES"
echo "files: $(find "$AUDIO" -type f \( -name '*.wav' -o -name '*.mp3' -o -name '*.flac' -o -name '*.m4a' \) | wc -l)"
echo

run_mode () {
  local label="$1" config="$2"
  echo "=============================================================="
  echo " $label  ($config)"
  echo "=============================================================="
  # One worker per mode: with num-workers > 1 every worker builds its own
  # model set on the SAME card (no per-worker GPU split), which distorts
  # per-stage timings and risks OOM. Timing comparisons need 1.
  "$PY" main_v2.py \
      --config "$config" \
      --input "$AUDIO" \
      --output "$OUT/$label" \
      --num-workers 1 \
      2>&1 | tee "$OUT/$label.run.log" || {
        echo "$label FAILED -- see $OUT/$label.run.log" >&2
        return 1
      }
  echo
}

TREES=""
for mode in $MODES; do
  run_mode "$mode" "$(config_for "$mode")"
  TREES="$TREES $OUT/$mode"
done

# Any DiariZen child must be gone once main_v2.py has exited. A survivor is
# holding VRAM and is a bug worth knowing about immediately.
if pgrep -f diarizen_runner >/dev/null 2>&1; then
  echo "WARNING: diarizen_runner still running after exit:" >&2
  pgrep -af diarizen_runner >&2
fi

echo "=============================================================="
echo " comparison"
echo "=============================================================="
# shellcheck disable=SC2086
"$PY" scripts/compare_ab_outputs.py $TREES --output "$OUT/comparison.json"
