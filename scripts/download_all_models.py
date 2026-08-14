#!/usr/bin/env python
"""Pre-download every remote model referenced by audio-preprocess pipeline configs.

Why this exists
----------------
`pipeline/global_var.py` / `pipeline_v2/stage2/models.py` all follow the same
pattern: if a configured `*_model_dir_cache` path exists on disk, load from
it; otherwise fall back to the bare model id and let the underlying library
(modelscope / huggingface_hub / funasr) download it *on first use*. On a
freshly provisioned Ray worker, several actor processes on the same machine
can hit that "not cached yet" branch concurrently, racing to download the
same model. We already fixed the resulting FunASR lock-key bug, but the
right long-term fix is to simply never let a worker hit a cold cache during
a real job: run this script once (per machine image / per new node) so
every model referenced by *any* config under `configs/` is already warm.

How it works
------------
Rather than hardcoding a model list that can silently drift from the actual
configs, this script:
  1. Scans every `configs/*.json` (or an explicit `--configs` list).
  2. For each provider section (funasr / funasr_nano / paraformer / whisper /
     pyannote / brouhaha / text_quality.ppl / alignment), reads the
     `*_dir_cache` path already declared there and reverse-parses it into
     (backend, repo_id, cache_root):
       - HuggingFace cache layout:  .../models--<org>--<name>/snapshots/...
       - ModelScope cache layout:   .../models/<namespace>/<name>
  3. Also parses `ckpts/*.yaml` (pyannote pipeline configs) for the
     `segmentation` / `embedding` sub-model paths, since those aren't listed
     directly in the JSON configs.
  4. Dedupes and downloads each unique repo into exactly the cache_root the
     pipeline code expects, so `os.path.exists(model_dir_cache)` is true on
     the next real run.

Usage
-----
    python scripts/download_all_models.py                  # scan configs/*.json, download everything
    python scripts/download_all_models.py --dry-run         # just print what would be downloaded
    python scripts/download_all_models.py --hf-mirror        # use hf-mirror.com for HuggingFace repos
    python scripts/download_all_models.py --only whisper,pyannote
    python scripts/download_all_models.py --jobs 4           # parallel downloads
    python scripts/download_all_models.py --configs configs/config_for_a100.json
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DEFAULT_CKPTS_DIR = os.path.join(PROJECT_ROOT, "ckpts")


@dataclass
class Task:
    backend: str  # "huggingface" | "modelscope"
    repo_id: str
    cache_root: Optional[str] = None
    revision: Optional[str] = None
    label: str = ""
    sources: Set[str] = field(default_factory=set)

    def key(self) -> Tuple[str, str, Optional[str], Optional[str]]:
        return (self.backend, self.repo_id, self.cache_root, self.revision)


# ----------------------------------------------------------------------
# Cache-path reverse parsing
# ----------------------------------------------------------------------
def parse_hf_cache_path(path: str):
    """`.../models--<org>--<name>/snapshots/...` -> (cache_root, repo_id)."""
    if not path:
        return None, None
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, seg in enumerate(parts):
        if seg.startswith("models--"):
            rest = seg[len("models--"):]
            if "--" not in rest:
                continue
            org, name = rest.split("--", 1)
            cache_root = "/".join(parts[:i]) or "/"
            return cache_root, f"{org}/{name}"
    return None, None


def parse_ms_cache_path(path: str):
    """`.../hub/models/<namespace>/<name>[...]` -> (cache_root, repo_id).

    Anchored on the literal `hub/models/` pair (ModelScope's on-disk layout
    is `<cache_dir ending in "hub">/models/<namespace>/<name>`) rather than
    a bare `models` segment, because some deployments additionally nest
    everything under a generic top-level `models/` directory (e.g.
    `.../models/modelscope/hub/models/iic/SenseVoiceSmall`), which would
    otherwise match the wrong occurrence.
    """
    if not path:
        return None, None
    norm = path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p != ""]
    for i in range(len(parts) - 1):
        if parts[i] == "hub" and parts[i + 1] == "models" and len(parts) >= i + 4:
            repo_id = "/".join(parts[i + 2:i + 4])
            cache_root = "/" + "/".join(parts[:i + 1])
            return cache_root, repo_id
    return None, None


def derive_task(dir_cache: Optional[str], fallback_model: Optional[str],
                 backend: str, revision: Optional[str], label: str) -> Optional[Task]:
    repo_id, cache_root = None, None
    if dir_cache and not dir_cache.endswith((".yaml", ".yml")):
        parser = parse_ms_cache_path if backend == "modelscope" else parse_hf_cache_path
        cache_root, repo_id = parser(dir_cache)
    if not repo_id:
        # Fall back to the plain model id (e.g. funasr shorthand like
        # "fsmn-vad" won't parse -- but by that point model_dir_cache was
        # already unset/unparseable, so there's nothing better to try).
        repo_id = fallback_model
        cache_root = None
    if not repo_id or "/" not in repo_id:
        # Bare shorthand names (e.g. "fsmn-vad") aren't real hub ids we can
        # download directly; skip rather than guess wrong.
        return None
    return Task(backend=backend, repo_id=repo_id, cache_root=cache_root,
                revision=revision, label=label)


# ----------------------------------------------------------------------
# Config / yaml scanning
# ----------------------------------------------------------------------
def collect_tasks_from_configs(config_paths):
    tasks = {}
    hf_token = None

    def add(cfg_name, backend, dir_cache, fallback, revision=None, label=""):
        if not dir_cache and not fallback:
            return
        t = derive_task(dir_cache, fallback, backend, revision, label)
        if not t:
            return
        key = t.key()
        if key not in tasks:
            tasks[key] = t
        tasks[key].sources.add(cfg_name)

    for cfg_path in config_paths:
        cfg_name = os.path.basename(cfg_path)
        try:
            with open(cfg_path, "r") as fp:
                cfg = json.load(fp)
        except Exception as e:
            print(f"[WARN] failed to parse {cfg_path}: {e}")
            continue

        token = cfg.get("huggingface_token", "")
        if not hf_token and isinstance(token, str) and token.startswith("hf"):
            hf_token = token

        fa = cfg.get("funasr", {})
        add(cfg_name, "modelscope", fa.get("model_dir_cache"), fa.get("model"),
            label="funasr.model")
        add(cfg_name, "modelscope", fa.get("vad_model_dir_cache"), fa.get("vad_model"),
            label="funasr.vad_model")

        nano = cfg.get("funasr_nano", {})
        add(cfg_name, "modelscope", nano.get("model_dir_cache"), nano.get("model"),
            label="funasr_nano.model")
        add(cfg_name, "modelscope", nano.get("vad_model_dir_cache"), nano.get("vad_model"),
            label="funasr_nano.vad_model")

        pf = cfg.get("paraformer", {})
        add(cfg_name, "modelscope", pf.get("model_dir_cache"), pf.get("model"),
            revision=pf.get("model_revision"), label="paraformer.model")
        add(cfg_name, "modelscope", pf.get("vad_model_dir_cache"), pf.get("vad_model"),
            label="paraformer.vad_model")
        add(cfg_name, "modelscope", pf.get("punc_model_dir_cache"), pf.get("punc_model"),
            label="paraformer.punc_model")

        wh = cfg.get("whisper", {})
        add(cfg_name, "huggingface", wh.get("model_dir_cache"), wh.get("model"),
            label="whisper.model")

        pa = cfg.get("pyannote", {})
        add(cfg_name, "huggingface", pa.get("model_dir_cache"), pa.get("model"),
            label="pyannote.model")

        br = cfg.get("metrics", {}).get("brouhaha", {})
        add(cfg_name, "huggingface", br.get("model_dir_cache"), br.get("model"),
            label="brouhaha.model")

        ppl = cfg.get("text_quality", {}).get("ppl", {})
        add(cfg_name, "huggingface", ppl.get("model_dir_cache"), ppl.get("model"),
            label="text_quality.ppl.model")

        al = cfg.get("alignment", {})
        for lang, p in al.get("language_models", {}).items():
            add(cfg_name, "huggingface", p, None, label=f"alignment.{lang}")

    return list(tasks.values()), hf_token


def collect_tasks_from_pyannote_yaml(ckpts_dir):
    """`ckpts/*.yaml` pyannote pipeline configs list segmentation/embedding
    sub-model absolute paths that never appear directly in configs/*.json."""
    tasks = {}
    if not yaml or not os.path.isdir(ckpts_dir):
        return []
    for fn in sorted(os.listdir(ckpts_dir)):
        if not fn.endswith((".yaml", ".yml")):
            continue
        fpath = os.path.join(ckpts_dir, fn)
        try:
            with open(fpath, "r") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception as e:
            print(f"[WARN] failed to parse {fpath}: {e}")
            continue
        params = (data.get("pipeline") or {}).get("params") or {}
        for sub_key in ("segmentation", "embedding"):
            p = params.get(sub_key)
            if isinstance(p, str) and "models--" in p:
                cache_root, repo_id = parse_hf_cache_path(p)
                if repo_id:
                    t = Task("huggingface", repo_id, cache_root, None,
                             label=f"pyannote_yaml.{sub_key}")
                    key = t.key()
                    if key not in tasks:
                        tasks[key] = t
                    tasks[key].sources.add(fn)
    return list(tasks.values())


# ----------------------------------------------------------------------
# Cache-hit checks (mirrors the pipeline's own os.path.exists() gate)
# ----------------------------------------------------------------------
_WEIGHT_EXTS = (".bin", ".safetensors", ".onnx", ".pt", ".pth")


def is_hf_cached(cache_root, repo_id):
    root = cache_root or os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(root, "models--" + repo_id.replace("/", "--"))
    snap = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snap):
        return False
    for d in os.listdir(snap):
        sd = os.path.join(snap, d)
        if not os.path.isdir(sd):
            continue
        files = os.listdir(sd)
        if any(f.endswith(_WEIGHT_EXTS) for f in files):
            return True
    return False


def is_ms_cached(cache_root, repo_id):
    root = cache_root or os.path.expanduser("~/.cache/modelscope/hub")
    p = os.path.join(root, "models", repo_id)
    if not os.path.isdir(p):
        return False
    for _, _, files in os.walk(p):
        if any(f.endswith(_WEIGHT_EXTS) for f in files):
            return True
    return False


def _dir_size_mb(path):
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dp, f)
            try:
                total += os.path.getsize(os.path.realpath(fp))
            except OSError:
                pass
    return total / 1024 / 1024


# ----------------------------------------------------------------------
# Download backends
# ----------------------------------------------------------------------
def download_hf(task: Task, token=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [FAIL] huggingface_hub not installed. pip install huggingface_hub")
        return False

    cache_root = task.cache_root or os.environ.get("HF_HUB_CACHE") or \
        os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_root, exist_ok=True)
    try:
        t0 = time.time()
        local = snapshot_download(
            repo_id=task.repo_id,
            revision=task.revision,
            cache_dir=cache_root,
            token=token,
            ignore_patterns=["*.gguf", "*.msgpack", "*.h5", "*.tflite"],
        )
        elapsed = time.time() - t0
        print(f"  [OK] {_dir_size_mb(local):.1f} MB in {elapsed:.1f}s -> {local}")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def download_ms(task: Task):
    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        except ImportError:
            print("  [FAIL] modelscope not installed. pip install 'modelscope[audio]'")
            return False

    cache_root = task.cache_root or os.path.expanduser("~/.cache/modelscope/hub")
    os.makedirs(cache_root, exist_ok=True)
    try:
        t0 = time.time()
        kwargs = dict(cache_dir=cache_root)
        if task.revision:
            kwargs["revision"] = task.revision
        local = ms_snapshot_download(task.repo_id, **kwargs)
        elapsed = time.time() - t0
        print(f"  [OK] {_dir_size_mb(local):.1f} MB in {elapsed:.1f}s -> {local}")
        return True
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=None,
                     help="Comma-separated config json paths (default: all configs/*.json)")
    ap.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR,
                     help=f"Directory to glob *.json from (default: {DEFAULT_CONFIGS_DIR})")
    ap.add_argument("--ckpts-dir", default=DEFAULT_CKPTS_DIR,
                     help=f"Directory to scan pyannote *.yaml from (default: {DEFAULT_CKPTS_DIR})")
    ap.add_argument("--hf-mirror", action="store_true",
                     help="Use https://hf-mirror.com for HuggingFace downloads (faster in CN)")
    ap.add_argument("--hf-token", default=None,
                     help="Override huggingface_token discovered in configs (needed for gated "
                          "pyannote repos)")
    ap.add_argument("--only", default=None,
                     help="Comma-separated label prefixes to filter, e.g. "
                          "funasr,paraformer,whisper,pyannote,brouhaha,text_quality,alignment")
    ap.add_argument("--jobs", type=int, default=1,
                     help="Parallel download workers (default: 1, sequential)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Only print what would be downloaded, don't download")
    args = ap.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF_ENDPOINT=https://hf-mirror.com")

    if args.configs:
        config_paths = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        config_paths = sorted(glob.glob(os.path.join(args.configs_dir, "*.json")))

    if not config_paths:
        print(f"No config files found under {args.configs_dir}")
        sys.exit(1)

    print(f"Scanning {len(config_paths)} config file(s) under {args.configs_dir}...")
    tasks, hf_token = collect_tasks_from_configs(config_paths)
    hf_token = args.hf_token or hf_token

    yaml_tasks = collect_tasks_from_pyannote_yaml(args.ckpts_dir)
    dedup = {t.key(): t for t in tasks}
    for t in yaml_tasks:
        key = t.key()
        if key in dedup:
            dedup[key].sources |= t.sources
        else:
            dedup[key] = t
    tasks = list(dedup.values())

    if args.only:
        prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
        tasks = [t for t in tasks if any(t.label.startswith(p) for p in prefixes)]

    tasks.sort(key=lambda t: (t.backend, t.repo_id))

    print(f"\nFound {len(tasks)} unique model repo(s):\n")
    for t in tasks:
        print(f"  [{t.backend:11s}] {t.repo_id:60s} rev={t.revision or 'default':8s} "
              f"cache={t.cache_root or '(default)'}")
    print()

    if not tasks:
        print("Nothing to do.")
        return
    if args.dry_run:
        print("(dry-run: nothing downloaded)")
        return

    def run_one(t: Task):
        print(f"\n=== [{t.backend}] {t.repo_id} (rev={t.revision or 'default'}) ===")
        print(f"    used by: {', '.join(sorted(t.sources))}")
        if t.backend == "huggingface":
            if is_hf_cached(t.cache_root, t.repo_id):
                print("  [SKIP] already cached")
                return t, True
            return t, download_hf(t, token=hf_token)
        else:
            if is_ms_cached(t.cache_root, t.repo_id):
                print("  [SKIP] already cached")
                return t, True
            return t, download_ms(t)

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(run_one, tasks))
    else:
        results = [run_one(t) for t in tasks]

    failed = [t for t, ok in results if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED ({len(failed)}/{len(tasks)}):")
        for t in failed:
            print(f"  - [{t.backend}] {t.repo_id}")
        print("\nCommon fixes:")
        print("  - Network: try --hf-mirror for hf-mirror.com (HuggingFace only; "
              "ModelScope is already CN-hosted)")
        print("  - Gated repos: pyannote/segmentation-3.0, pyannote/wespeaker-*, "
              "pyannote/speaker-diarization-3.1 and pyannote/brouhaha require accepting the "
              "model's user agreement on huggingface.co with the account owning --hf-token")
        print("  - Proxy:   unset HTTP_PROXY HTTPS_PROXY if a proxy blocks the target host")
        print("  - Disk:    df -h ~/.cache")
        sys.exit(1)
    else:
        print(f"SUCCESS: {len(tasks)} model repo(s) cached/verified")


if __name__ == "__main__":
    main()
