#!/bin/bash
# Create the two core conda envs (guide §3.3) and lock them. Run on the LOGIN node.
#   CU=cu126 bash scripts/setup_envs.sh gen
#   CU=cu126 bash scripts/setup_envs.sh eval
#
# CU must be <= the "CUDA Version" that nvidia-smi shows on the GPU nodes (from
# slurm/probe_gpu.sbatch). Driver too old for cu126 -> try cu124 / cu121.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:?usage: setup_envs.sh gen|eval}
CU=${CU:-cu126}
# /home is nearly full on HPC3: keep the multi-GB pip download cache on scratch.
if mkdir -p "/scratch/$USER/pip_cache" 2>/dev/null; then
  export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
fi
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
if [[ ! -f "$CONDA_SH" ]]; then
  cat >&2 <<'EOF'
Miniforge not found. Install it into your home directory first (login node):
  cd ~ && curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3 && rm Miniforge3-*.sh
Then rerun this script. (Or set CONDA_SH to your conda.sh if conda lives elsewhere.)
EOF
  exit 2
fi
# conda's activation scripts trip over `set -u`
set +u
source "$CONDA_SH"
if ! conda env list | grep -qE "^$ENV\s"; then
  conda create -n "$ENV" python=3.11 -y
fi
conda activate "$ENV"
set -u
python -m pip install --upgrade pip

case "$ENV" in
  gen)
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CU"
    pip install diffusers transformers accelerate sentencepiece protobuf safetensors
    ;;
  eval)
    # t2v-metrics 3.x needs torch>=2.7 and transformers>=5 (guide §7.4)
    pip install "torch>=2.7" torchvision --index-url "https://download.pytorch.org/whl/$CU"
    pip install t2v-metrics open_clip_torch
    ;;
  *) echo "unknown env $ENV" >&2; exit 2 ;;
esac
pip install -e ".[analysis]"

mkdir -p env
pip freeze --exclude-editable > "env/$ENV.lock.txt"
python - <<'PY'
import importlib.metadata as md
for p in ["torch", "diffusers", "transformers", "accelerate", "t2v-metrics", "open-clip-torch"]:
    try: print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError: pass
PY
conda clean -a -y >/dev/null 2>&1 || true   # drop conda's package tarballs from /home
echo "Locked to env/$ENV.lock.txt — commit it."
