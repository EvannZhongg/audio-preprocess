#!/usr/bin/env python
"""
Pre-download Qwen3-0.6B from HuggingFace to local cache.

Used by `pipeline.text_quality_filtering` PPL scorer. The pipeline
config points to:
    /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B

If `model_dir_cache` doesn't exist, `pipeline/global_var.py` falls back
to fetching `Qwen/Qwen3-0.6B` online — which fails on worker nodes
with no outbound network.

Why Qwen3-0.6B over Qwen2.5-0.5B:
  - 36T training tokens (vs 18T) → tighter PPL distribution
  - 119-language coverage → noticeably better ja / ko / ru scoring
  - Cleaner separation between hallucinated text and natural text PPL

Usage:
    python scripts/download_qwen3_06b.py
    python scripts/download_qwen3_06b.py --revision main
    python scripts/download_qwen3_06b.py --cache-dir /custom/path
    python scripts/download_qwen3_06b.py --hf-mirror   # use hf-mirror.com
"""
import argparse
import os
import sys
import time

MODEL_ID = "Qwen/Qwen3-0.6B"


def _expected_path(cache_dir: str) -> str:
    # HF caches under hub/models--<org>--<name>
    safe = MODEL_ID.replace("/", "--")
    return os.path.join(cache_dir, "hub", f"models--{safe}")


def _is_cached(cache_dir: str) -> bool:
    p = _expected_path(cache_dir)
    if not os.path.isdir(p):
        return False
    snap = os.path.join(p, "snapshots")
    if not os.path.isdir(snap):
        return False
    # Look for any snapshot with at least the config + a weights file
    for d in os.listdir(snap):
        sd = os.path.join(snap, d)
        files = os.listdir(sd) if os.path.isdir(sd) else []
        has_cfg = "config.json" in files
        has_w = any(f.endswith((".safetensors", ".bin")) for f in files)
        if has_cfg and has_w:
            return True
    return False


def download(cache_dir: str, revision: str, use_mirror: bool):
    print(f"Cache dir : {cache_dir}")
    print(f"Model     : {MODEL_ID}")
    print(f"Revision  : {revision}")
    print(f"HF mirror : {use_mirror}")
    print()

    if _is_cached(cache_dir):
        print("[SKIP] model already cached at:")
        print(f"  {_expected_path(cache_dir)}")
        return 0

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF_ENDPOINT=https://hf-mirror.com")

    os.environ["HF_HUB_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub")
        return 1

    try:
        t0 = time.time()
        local = snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            cache_dir=cache_dir,
            # Skip large optional artifacts to save bandwidth
            ignore_patterns=["*.gguf", "*.onnx", "*.msgpack"],
        )
        elapsed = time.time() - t0
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(local)
            for f in files if not os.path.islink(os.path.join(dp, f))
        ) / 1024 / 1024
        print(f"\n[OK] downloaded {size:.1f} MB in {elapsed:.1f}s")
        print(f"     snapshot path: {local}")
        print(f"     expected by pipeline: {_expected_path(cache_dir)}")
        return 0
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        print()
        print("Common fixes:")
        print("  - Network: try `--hf-mirror` to use hf-mirror.com")
        print("  - Token:   `huggingface-cli login` if model is gated")
        print("  - Proxy:   `unset HTTP_PROXY HTTPS_PROXY` if proxy blocks HF")
        print("  - Disk:    df -h ~/.cache")
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir",
                    default=os.path.expanduser("~/.cache/huggingface"),
                    help="HuggingFace cache root (default: ~/.cache/huggingface)")
    ap.add_argument("--revision", default="main",
                    help="Git revision/branch/tag to download (default: main)")
    ap.add_argument("--hf-mirror", action="store_true",
                    help="Use https://hf-mirror.com (faster in CN)")
    args = ap.parse_args()

    sys.exit(download(args.cache_dir, args.revision, args.hf_mirror))


if __name__ == "__main__":
    main()
