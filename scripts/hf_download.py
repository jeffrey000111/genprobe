"""Download one Hugging Face repo snapshot into $HF_HOME with include/exclude globs, then check it.

Called by scripts/download_models.sh on coe-hpc1. Uses the Python API (snapshot_download)
rather than the `hf download` CLI: the CLI's multi-pattern flag syntax changed between
huggingface_hub releases (2.x silently treated extra --include patterns as file names),
while snapshot_download's allow_patterns/ignore_patterns has been stable.

    python scripts/hf_download.py stabilityai/stable-diffusion-xl-base-1.0 \
        --include "*.json" "*.txt" "*/*fp16.safetensors"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--include", nargs="+", required=True, help="glob patterns to download")
    ap.add_argument("--exclude", nargs="+", default=None, help="glob patterns to skip")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--expect", nargs="+", default=["model_index.json"],
                    help="files that must exist afterwards (default: diffusers' model_index.json)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=4,
    ))

    files = sorted(p for p in path.rglob("*") if p.is_file())
    total = sum(os.path.getsize(p) for p in files)  # follows the cache's symlinks to blobs
    weights = [p for p in files if p.suffix == ".safetensors"]
    print(f"snapshot: {path}")
    print(f"files: {len(files)} ({len(weights)} .safetensors), {total / 1e9:.2f} GB")
    for p in weights:
        print(f"  {p.relative_to(path)}  {os.path.getsize(p) / 1e9:.2f} GB")

    problems = [f"{f} missing" for f in args.expect if not (path / f).is_file()]
    if not weights:
        problems.append("no .safetensors weights downloaded")
    for sub in sorted({p.relative_to(path).parts[0] for p in weights if len(p.relative_to(path).parts) > 1}):
        if not any((path / sub).glob("*.json")):
            problems.append(f"{sub}/ has weights but no config json")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
