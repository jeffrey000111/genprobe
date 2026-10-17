# Sourced by every job script. Edit the paths once, here.
# shellcheck shell=bash
#
# Cluster: SJSU CoE HPC3 (login node g17, reached as `ssh coe-hpc3` from coe-hpc1).
# Storage (Oct 2026): /home is shared and ~99% full; /scratch has ~13 TB free but is purged
# periodically (see docs/hpc_notes.md). So: re-downloadable things (model weights, pip cache)
# go to /scratch; code, envs and generated data stay in /home.

# --- EDIT IF NEEDED --------------------------------------------------------------
export GENPROBE_ROOT="${GENPROBE_ROOT:-$HOME/genprobe}"
export CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
export GENPROBE_SCRATCH="${GENPROBE_SCRATCH:-/scratch/$USER}"
# Model weights (~100 GB): re-downloadable, so scratch is the right place.
export HF_HOME="${HF_HOME:-$GENPROBE_SCRATCH/hf_home}"
# 1 if compute nodes cannot reach huggingface.co (slurm/probe_gpu.sbatch tells you).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# GPUs we accept for generation; P100/V100 lack bf16 (guide §3.6).
export GENPROBE_GPU_REGEX="${GENPROBE_GPU_REGEX:-A100|H100}"
# ----------------------------------------------------------------------------------

if ! mkdir -p "$HF_HOME" 2>/dev/null; then
  echo "ERROR: cannot create HF_HOME=$HF_HOME; set GENPROBE_SCRATCH or HF_HOME in slurm/common.sh" >&2
  exit 2
fi

activate_env() {
  # conda's activate scripts trip over `set -u`, so relax it just for activation
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$1"
  set -u
}

require_gpu() {
  local name
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "GPU: $name"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  if ! grep -qE "$GENPROBE_GPU_REGEX" <<<"$name"; then
    echo "ERROR: got '$name', need one matching /$GENPROBE_GPU_REGEX/. Exiting before loading anything." >&2
    exit 3
  fi
}
