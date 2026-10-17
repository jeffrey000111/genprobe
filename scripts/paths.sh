# Paths shared by job scripts (on HPC3) and setup scripts (on coe-hpc1).
# shellcheck shell=bash
#
# /home and /scratch are the same filesystems on coe-hpc1, HPC3 (g17) and the compute nodes,
# so every path below means the same thing on all of them (docs/hpc_notes.md, D-011, D-012).
# Override any of them by exporting it before sourcing this file.

export GENPROBE_ROOT="${GENPROBE_ROOT:-$HOME/genprobe}"
# Python virtualenvs, one per role (gen, eval). On /home: scratch is purged each semester.
export GENPROBE_ENVS="${GENPROBE_ENVS:-$HOME/envs}"
# Per-user folder inside the CMPE-261 class scratch dir: re-downloadable data only.
export GENPROBE_SCRATCH="${GENPROBE_SCRATCH:-/scratch/cmpe261-fa26/$USER}"
# Model weights (~60 GB for the four generators).
export HF_HOME="${HF_HOME:-$GENPROBE_SCRATCH/hf_home}"
# The HF access token stays in the private home dir, never on the shared class scratch.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
# uv: package cache on scratch (multi-GB, re-downloadable); managed Python on /home.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GENPROBE_SCRATCH/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"   # cache and envs are on different filesystems
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac

# Create the per-user scratch folder, private to this user (the class dir is group-shared).
ensure_scratch() {
  if ! mkdir -p "$GENPROBE_SCRATCH" 2>/dev/null; then
    echo "ERROR: cannot create $GENPROBE_SCRATCH (are you in the cmpe261-fa26 group? run: id)" >&2
    return 2
  fi
  chmod 700 "$GENPROBE_SCRATCH"
  if ! mkdir -p "$HF_HOME" "$UV_CACHE_DIR" 2>/dev/null; then
    echo "ERROR: cannot create $HF_HOME or $UV_CACHE_DIR" >&2
    return 2
  fi
}

# True when this machine can reach PyPI (coe-hpc1 can; HPC3 and its compute nodes cannot).
has_internet() {
  curl -sI --max-time 10 https://pypi.org >/dev/null 2>&1
}
