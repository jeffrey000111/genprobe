"""Read the local Hugging Face cache without importing huggingface_hub.

Jobs on HPC3 run offline (HF_HUB_OFFLINE=1), so a model that was never downloaded on
coe-hpc1 only shows up as a confusing error deep inside from_pretrained. Metrics check the
cache up front with these helpers and record the snapshot they loaded.
"""

from __future__ import annotations

import os
from pathlib import Path


def hub_cache_dir() -> Path:
    """Where huggingface_hub keeps repos: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_dir(repo_id: str, repo_type: str = "model") -> Path:
    return hub_cache_dir() / f"{repo_type}s--{repo_id.replace('/', '--')}"


def cached_revision(repo_id: str, ref: str = "main", repo_type: str = "model") -> str | None:
    """Commit hash that `ref` points to in the local cache, or None if never downloaded."""
    p = repo_cache_dir(repo_id, repo_type) / "refs" / ref
    try:
        sha = p.read_text().strip()
    except OSError:
        return None
    return sha or None


def require_cached(repo_id: str, files: tuple[str, ...] = (), download_hint: str = "") -> str:
    """Return the cached commit hash of `repo_id`; raise with a fix-it message if missing."""
    sha = cached_revision(repo_id)
    missing = [] if sha else ["refs/main"]
    if sha:
        snap = repo_cache_dir(repo_id) / "snapshots" / sha
        missing = [f for f in files if not (snap / f).exists()]
    if missing:
        hint = f"\nDownload it on coe-hpc1: {download_hint}" if download_hint else ""
        raise FileNotFoundError(
            f"{repo_id} is not (fully) in the local HF cache {hub_cache_dir()} "
            f"(missing: {', '.join(missing)}).{hint}")
    return sha  # type: ignore[return-value]
