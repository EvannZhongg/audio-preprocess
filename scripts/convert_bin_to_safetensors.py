#!/usr/bin/env python
"""
Convert pytorch_model.bin → model.safetensors in WhisperX align model snapshots.

Why:
  transformers >=4.50 enforces CVE-2025-32434, refusing to call
  torch.load() (used to load .bin) unless torch >= 2.6. We're on
  torch 2.5.0 (dockerfile-pinned) for compatibility, so .bin loading
  fails — and the fallback ("download safetensors from HF") fails
  on offline workers.

  Solution: pre-convert each model's .bin to .safetensors. Once a
  snapshot dir contains both, transformers prefers safetensors and
  skips the torch.load path entirely.

What this script does:
  - Walks the HF cache hub dir.
  - For each models--<org>--<name>/snapshots/<commit>/, if
    pytorch_model.bin exists and model.safetensors doesn't, convert.
  - Uses torch.load(weights_only=False) directly — bypasses the
    transformers CVE check (we trust our own CFS-cached files).
  - Idempotent: skips conversion if model.safetensors already present.

Usage:
    python scripts/convert_bin_to_safetensors.py \
        --hub-dir /cfs/cfs-czb184s7/allenxzhang/models/huggingface/hub
    python scripts/convert_bin_to_safetensors.py \
        --hub-dir ~/.cache/huggingface/hub \
        --filter wav2vec2     # only convert repos with 'wav2vec2' in name
"""
import argparse
import os
import sys
import time


def find_snapshots(hub_dir: str, name_filter: str = ""):
    """Yield (repo_id, snapshot_dir) for every snapshot under hub_dir."""
    if not os.path.isdir(hub_dir):
        return
    for repo_marker in sorted(os.listdir(hub_dir)):
        if not repo_marker.startswith("models--"):
            continue
        if name_filter and name_filter.lower() not in repo_marker.lower():
            continue
        repo_id = repo_marker[len("models--"):].replace("--", "/", 1)
        snap_root = os.path.join(hub_dir, repo_marker, "snapshots")
        if not os.path.isdir(snap_root):
            continue
        for d in sorted(os.listdir(snap_root)):
            sd = os.path.join(snap_root, d)
            if os.path.isdir(sd):
                yield repo_id, sd


def convert_one(snapshot_dir: str) -> str:
    """Convert pytorch_model.bin to model.safetensors in this snapshot dir.

    Returns: 'ok' / 'skip' / 'no-bin' / 'fail:<reason>'.
    """
    bin_path = os.path.join(snapshot_dir, "pytorch_model.bin")
    st_path = os.path.join(snapshot_dir, "model.safetensors")

    if os.path.exists(st_path) or os.path.islink(st_path):
        return "skip"
    if not (os.path.exists(bin_path) or os.path.islink(bin_path)):
        return "no-bin"

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        return f"fail:import:{e}"

    try:
        # weights_only=False is safe here: we control these files (CFS-cached
        # from official HF repos). The CVE-2025-32434 attack vector requires
        # malicious .bin files, which is not our threat model.
        state_dict = torch.load(bin_path, weights_only=False, map_location="cpu")

        # Some HF .bin files are wrapped in a dict at top level
        # (e.g. {"model": state_dict}); we want the flat tensor map.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            return f"fail:not-dict:got {type(state_dict).__name__}"

        # safetensors requires non-shared, contiguous tensors. Detach + clone
        # to be safe; also strip non-tensor entries (sometimes there are ints
        # mixed in like global_step).
        clean = {}
        for k, v in state_dict.items():
            if hasattr(v, "detach") and hasattr(v, "contiguous"):
                clean[k] = v.detach().contiguous()
        save_file(clean, st_path)
        return "ok"
    except Exception as e:
        return f"fail:{type(e).__name__}:{e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub-dir",
                    default=os.path.expanduser("~/.cache/huggingface/hub"),
                    help="HF hub cache root (the dir containing models--*/)")
    ap.add_argument("--filter", default="",
                    help="Only process repos whose name contains this substring "
                         "(case-insensitive). Empty = all repos.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be converted; don't actually write.")
    args = ap.parse_args()

    print(f"Hub dir: {args.hub_dir}")
    print(f"Filter:  {args.filter or '(none)'}")
    print(f"Mode:    {'DRY-RUN' if args.dry_run else 'CONVERT'}\n")

    counts = {"ok": 0, "skip": 0, "no-bin": 0, "fail": 0}
    for repo_id, snap in find_snapshots(args.hub_dir, args.filter):
        bin_path = os.path.join(snap, "pytorch_model.bin")
        st_path = os.path.join(snap, "model.safetensors")
        rel = os.path.relpath(snap, args.hub_dir)
        if args.dry_run:
            need = (
                (os.path.exists(bin_path) or os.path.islink(bin_path))
                and not (os.path.exists(st_path) or os.path.islink(st_path))
            )
            print(f"  [{'WOULD' if need else 'skip '}] {repo_id} ({rel})")
            if need:
                counts["ok"] += 1
            continue

        t0 = time.time()
        result = convert_one(snap)
        elapsed = time.time() - t0
        print(f"  [{result.split(':')[0]:7s}] {repo_id}  ({elapsed:.1f}s)"
              + (f"  reason: {result}" if result.startswith("fail") else ""))
        bucket = result.split(":")[0] if ":" in result else result
        counts[bucket if bucket in counts else "fail"] += 1

    print("\n" + "=" * 60)
    print(f"converted={counts['ok']}  skipped={counts['skip']}  "
          f"no-bin={counts['no-bin']}  failed={counts['fail']}")
    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
