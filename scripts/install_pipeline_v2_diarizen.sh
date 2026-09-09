#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_DIR="${DIARIZEN_ENV_DIR:-$ROOT/.venv-diarizen}"
SOURCE_DIR="${DIARIZEN_SOURCE_DIR:-$ROOT/.third_party/DiariZen}"
BASE_PYTHON="${PYTHON:-python3.11}"
DIARIZEN_REV="${DIARIZEN_REV:-844f5555b0a98acd0931511fc641a8c5b8ba92c7}"

if ! command -v "$BASE_PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10+ is required. Install python3.11 or set PYTHON=/path/to/python." >&2
  exit 1
fi

"$BASE_PYTHON" -c \
  "import sys; assert sys.version_info >= (3, 10), 'DiariZen requires Python >= 3.10'"

if [ ! -d "$SOURCE_DIR/.git" ]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone https://github.com/BUTSpeechFIT/DiariZen.git "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch origin "$DIARIZEN_REV"
git -C "$SOURCE_DIR" checkout --detach "$DIARIZEN_REV"
git -C "$SOURCE_DIR" submodule update --init --recursive

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install \
  -c "$SOURCE_DIR/constraints.txt" \
  -r "$ROOT/requirements-diarizen.txt" \
  -e "$SOURCE_DIR" \
  -e "$SOURCE_DIR/pyannote-audio"

"$ENV_DIR/bin/python" -c \
  "import torch; from diarizen.pipelines.inference import DiariZenPipeline; print('DiariZen ready; CUDA:', torch.cuda.is_available())"
