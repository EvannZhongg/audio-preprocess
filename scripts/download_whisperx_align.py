#!/usr/bin/env python
"""
Pre-download WhisperX wav2vec2 alignment models for all supported languages.

WhisperX uses one wav2vec2-CTC model per language for forced alignment.
The pipeline pulls these on first use, but worker nodes have no
outbound network — so we cache them at image build time.

The default model IDs match `whisperx.alignment.DEFAULT_ALIGN_MODELS_HF`
for our 7 supported languages. Note `en` uses a torchaudio bundle
(WAV2VEC2_ASR_BASE_960H), not HuggingFace — torchaudio downloads it
on first use to ~/.cache/torch/hub/checkpoints/.

Total disk: ~7-8 GB (six XLSR-53 models @ ~1.2 GB each + EN bundle).

Usage:
    python scripts/download_whisperx_align.py
    python scripts/download_whisperx_align.py --langs zh,en,ru
    python scripts/download_whisperx_align.py --hf-mirror
    python scripts/download_whisperx_align.py --cache-dir /custom/path
"""
import argparse
import os
import sys
import time

# Mirror of whisperx.alignment.DEFAULT_ALIGN_MODELS_HF for our 7 languages.
# Note: we use facebook/wav2vec2-base-960h for `en` instead of the torchaudio
# bundle WAV2VEC2_ASR_BASE_960H — same underlying model but distributed via
# HF, which lets us manage all 7 languages in one cache and pin per-language
# local paths in config.
DEFAULT_HF_MODELS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "facebook/wav2vec2-base-960h",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
}


def _safe_repo_dir(cache_dir: str, repo: str) -> str:
    return os.path.join(cache_dir, "hub", "models--" + repo.replace("/", "--"))


def _is_hf_cached(cache_dir: str, repo: str) -> bool:
    p = _safe_repo_dir(cache_dir, repo)
    snap = os.path.join(p, "snapshots")
    if not os.path.isdir(snap):
        return False
    for d in os.listdir(snap):
        sd = os.path.join(snap, d)
        files = os.listdir(sd) if os.path.isdir(sd) else []
        has_cfg = any(f in files for f in ("config.json", "preprocessor_config.json"))
        has_w = any(f.endswith((".safetensors", ".bin")) for f in files)
        if has_cfg and has_w:
            return True
    return False


def download_hf_repo(repo: str, cache_dir: str) -> bool:
    print(f"\n=== {repo} ===")
    if _is_hf_cached(cache_dir, repo):
        print("  [SKIP] already cached")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ERROR: huggingface_hub not installed. pip install huggingface_hub")
        return False

    try:
        t0 = time.time()
        local = snapshot_download(
            repo_id=repo,
            cache_dir=cache_dir,
            ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.tflite"],
        )
        elapsed = time.time() - t0
        # snapshot dirs use symlinks pointing to ../../blobs/<hash>; follow
        # the symlinks so size reflects real downloaded weights.
        size = sum(
            os.path.getsize(os.path.realpath(os.path.join(dp, f)))
            for dp, _, files in os.walk(local)
            for f in files
        ) / 1024 / 1024
        print(f"  [OK] {size:.1f} MB in {elapsed:.1f}s → {local}")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir",
                    default=os.path.expanduser("~/.cache/huggingface/hub"),
                    help="HF hub cache root (default: ~/.cache/huggingface/hub). "
                         "Files land at <cache-dir>/models--<org>--<name>/...")
    ap.add_argument("--langs",
                    default="zh,en,fr,ja,ko,de,ru",
                    help="Comma-separated languages to download (default: all 7)")
    ap.add_argument("--hf-mirror", action="store_true",
                    help="Use https://hf-mirror.com (faster in CN)")
    args = ap.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF_ENDPOINT=https://hf-mirror.com")

    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"HF cache : {args.cache_dir}")

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    print(f"Languages: {langs}\n")

    failed = []
    for lang in langs:
        if lang in DEFAULT_HF_MODELS:
            repo = DEFAULT_HF_MODELS[lang]
            if not download_hf_repo(repo, args.cache_dir):
                failed.append(f"{lang} ({repo})")
        else:
            print(f"\n=== {lang} ===\n  [SKIP] no mapping in DEFAULT_HF_MODELS")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(langs)}):")
        for f in failed:
            print(f"  - {f}")
        print("\nCommon fixes:")
        print("  - Network: try `--hf-mirror` for hf-mirror.com")
        print("  - Proxy:   `unset HTTP_PROXY HTTPS_PROXY` if proxy blocks HF")
        print("  - Disk:    df -h ~/.cache  (~8 GB needed for full set)")
        sys.exit(1)
    else:
        print(f"SUCCESS: {len(langs)} alignment model(s) cached")


if __name__ == "__main__":
    main()
