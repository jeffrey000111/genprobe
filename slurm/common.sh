# Sourced by every job script on HPC3. Paths live in scripts/paths.sh (shared with the setup
# scripts that run on coe-hpc1); job-only settings live here.
# shellcheck shell=bash

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/paths.sh"

# HPC3 compute nodes have no internet: models must already be in $HF_HOME
# (scripts/download_models.sh on coe-hpc1 puts them there).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# vLLM (judge env) would otherwise try to send usage statistics from an offline node.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# vLLM 0.32 picks FlashInfer's top-k/top-p sampler even when FlashInfer can't JIT-compile it (no
# nvcc on the nodes), and engine start-up then dies in its warm-up (judge-probe job 92312;
# vllm-project/vllm issues 49497, 59954). Use vLLM's native PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
# GPUs we accept by default; generation narrows this to one card type (slurm/gen.sbatch).
export GENPROBE_GPU_REGEX="${GENPROBE_GPU_REGEX:-A100|H100}"

if [[ ! -d "$HF_HOME" ]]; then
  echo "ERROR: HF_HOME=$HF_HOME does not exist; run scripts/download_models.sh on coe-hpc1 first" >&2
  exit 2
fi

# Activate one of the virtualenvs built by scripts/setup_envs.sh (gen, eval).
activate_env() {
  local venv="$GENPROBE_ENVS/$1"
  if [[ ! -f "$venv/bin/activate" ]]; then
    echo "ERROR: env '$1' not found at $venv. Build it on coe-hpc1:" >&2
    echo "  ssh sjsu \"cd ~/genprobe && bash scripts/setup_envs.sh $1\"" >&2
    exit 2
  fi
  set +u
  # shellcheck disable=SC1090,SC1091
  source "$venv/bin/activate"
  set -u
}

# Slurm sets CUDA_VISIBLE_DEVICES, but HPC3 does not hide the other GPUs from nvidia-smi
# (no cgroup device limits), so always query the GPU we were actually given.
require_gpu() {
  local idx="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$idx" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty; request a GPU with --gres" >&2
    exit 3
  fi
  local name
  name=$(nvidia-smi -i "${idx%%,*}" --query-gpu=name --format=csv,noheader)
  echo "GPU (CUDA_VISIBLE_DEVICES=$idx): $name"
  nvidia-smi -i "${idx%%,*}" --query-gpu=name,memory.total,driver_version --format=csv
  if ! grep -qE "$GENPROBE_GPU_REGEX" <<<"$name"; then
    echo "ERROR: got '$name', need one matching /$GENPROBE_GPU_REGEX/. Exiting before loading anything." >&2
    exit 3
  fi
}
