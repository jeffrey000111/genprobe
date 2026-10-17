# Paths shared by job scripts (on HPC3) and setup scripts (on coe-hpc1).
# shellcheck shell=bash
#
# /home is the same filesystem on coe-hpc1, HPC3 (g17) and the compute nodes, so every /home
# path below means the same thing on all of them (docs/hpc_notes.md, D-011, D-012).
# Override any of them by exporting it before sourcing this file.
#
# Nothing here lives on /scratch: from coe-hpc1, file locks (flock) on /scratch hang forever,
# and both uv and huggingface_hub lock files while they download. Locks on /home work.

export GENPROBE_ROOT="${GENPROBE_ROOT:-$HOME/genprobe}"
# Python virtualenvs, one per role (gen, eval).
export GENPROBE_ENVS="${GENPROBE_ENVS:-$HOME/envs}"
# Model weights (~60 GB for the four generators). Downloaded on coe-hpc1, read offline on HPC3.
export HF_HOME="${HF_HOME:-$HOME/hf_home}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
# uv's download cache is disposable, so it goes on the local disk of whichever machine runs
# uv (in practice coe-hpc1): fast, private to that machine, and no NFS locking involved.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"   # cache and envs are on different filesystems
# Per-user folder in the CMPE-261 class scratch dir. Not used by default (see above); kept for
# large temporary files written by jobs on HPC3, where locking is not needed.
export GENPROBE_SCRATCH="${GENPROBE_SCRATCH:-/scratch/cmpe261-fa26/$USER}"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac

# Create the folders the setup scripts write to.
ensure_dirs() {
  local d
  for d in "$GENPROBE_ENVS" "$HF_HOME" "$UV_CACHE_DIR" "$(dirname "$HF_TOKEN_PATH")"; do
    if ! mkdir -p "$d" 2>/dev/null; then
      echo "ERROR: cannot create $d" >&2
      return 2
    fi
  done
  chmod 700 "$UV_CACHE_DIR"   # /tmp is shared by every user of the machine
}

# True when this machine can reach PyPI (coe-hpc1 can; HPC3 and its compute nodes cannot).
has_internet() {
  curl -sI --max-time 10 https://pypi.org >/dev/null 2>&1
}
