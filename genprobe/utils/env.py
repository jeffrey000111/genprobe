"""Environment fingerprint: everything needed to explain why two runs differ.

Package versions come from importlib.metadata, so this never imports torch just to
read a version string. GPU details come from torch if it is already importable,
falling back to nvidia-smi.
"""

from __future__ import annotations

import functools
import importlib.metadata as md
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

PACKAGES = (
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "t2v-metrics", "open-clip-torch", "huggingface-hub", "qwen-vl-utils", "flash-attn",
    "easyocr", "opencv-python-headless", "pydantic", "numpy", "pillow",
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


# Paths whose changes affect what a run computes. Files the cluster itself writes and that
# are tracked in git (results/, env/*.lock.txt, docs/hpc_check.md) are deliberately left out,
# so a clean checkout isn't reported as dirty just because a job produced output.
CODE_PATHS = ("genprobe", "configs", "slurm", "scripts", "data/prompts", "data/labels", "pyproject.toml")


def git_commit(repo_dir: str | None = None) -> str:
    """Short commit hash, '-dirty' if code paths have uncommitted changes; 'nogit' outside a repo."""
    cwd = repo_dir or os.getcwd()
    sha = _run(["git", "-C", cwd, "rev-parse", "--short", "HEAD"])
    top = _run(["git", "-C", cwd, "rev-parse", "--show-toplevel"])
    if sha is None or top is None:
        return "nogit"
    paths = [p for p in CODE_PATHS if os.path.exists(os.path.join(top, p))]
    dirty = _run(["git", "-C", top, "status", "--porcelain", "--untracked-files=no", "--", *paths])
    return f"{sha}-dirty" if dirty else sha


def package_versions(packages=PACKAGES) -> dict[str, str | None]:
    out = {}
    for name in packages:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None, "gpu_mem_gb": None, "cuda_runtime": None, "driver": None,
                            "bf16_supported": None}
    if "torch" in sys.modules or package_versions(("torch",))["torch"]:
        try:
            import torch

            info["cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = props.name
                info["gpu_mem_gb"] = round(props.total_memory / 1e9, 1)
                info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if smi:
        name, _, driver = smi.splitlines()[0].partition(",")
        info["gpu"] = info["gpu"] or name.strip()
        info["driver"] = driver.strip()
    return info


@functools.lru_cache(maxsize=1)
def fingerprint() -> dict[str, Any]:
    slurm = {k: os.environ.get(k) for k in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
        "SLURMD_NODENAME",
    ) if os.environ.get(k)}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "packages": package_versions(),
        **gpu_info(),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "slurm": slurm,
    }


def run_id() -> str:
    """Stable ID for this job: Slurm array job/task when available, else a timestamp+pid."""
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        return f"slurm{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    if os.environ.get("SLURM_JOB_ID"):
        return f"slurm{os.environ['SLURM_JOB_ID']}"
    return datetime.now().strftime("local%Y%m%d-%H%M%S") + f"_{os.getpid()}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
