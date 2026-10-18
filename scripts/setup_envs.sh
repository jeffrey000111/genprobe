#!/bin/bash
# Build a Python virtualenv for HPC3 jobs. Run it on **coe-hpc1**, the only SJSU HPC machine
# with internet. /home is shared, so the env is usable on HPC3 and its compute nodes as is.
#
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh gen"
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh eval"
#
# coe-hpc1 runs CentOS 7 (glibc 2.17) and HPC3 runs glibc 2.34, so packages are selected for
# HPC3's platform with uv's --python-platform, not for the machine doing the install. uv and
# its managed Python only need glibc 2.17, so both run on either machine (D-012).
# Afterwards, test the env on a GPU node: sbatch slurm/probe_gpu.sbatch (on HPC3).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:?usage: setup_envs.sh gen|eval}
# shellcheck disable=SC1091
source scripts/paths.sh
TARGET_PLATFORM=${TARGET_PLATFORM:-x86_64-manylinux_2_34}   # HPC3's glibc
PY=${PY:-3.11}

if ! has_internet; then
  echo "No internet on $(hostname). Run this on coe-hpc1 instead:" >&2
  echo "  ssh sjsu \"cd ~/genprobe && bash scripts/setup_envs.sh $ENV\"" >&2
  exit 2
fi
ensure_dirs
mkdir -p "$GENPROBE_ENVS" env

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version
uv python install "$PY"

VENV="$GENPROBE_ENVS/$ENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "$PY" --seed
fi
# --build-constraints: some sdists (image-reward) still need pkg_resources at build time.
INSTALL=(uv pip install --python "$VENV/bin/python" --python-platform "$TARGET_PLATFORM"
         --build-constraints scripts/build-constraints.txt --excludes scripts/excludes.txt)

case "$ENV" in
  gen)
    # PyPI's Linux torch wheels bundle their CUDA runtime; HPC3's driver (590, CUDA 13.1)
    # supports all current ones, so no special index is needed.
    "${INSTALL[@]}" torch torchvision diffusers transformers accelerate sentencepiece protobuf safetensors
    ;;
  eval)
    # Require t2v-metrics>=3.0 (Qwen2.5-VL backbones, transformers>=4.52). Left unpinned, the
    # resolver kept the newest torch and fell back to t2v-metrics 1.2 + transformers 4.36.
    "${INSTALL[@]}" "t2v-metrics>=3.0" torchvision open_clip_torch
    ;;
  *) echo "unknown env '$ENV' (expected gen or eval)" >&2; exit 2 ;;
esac
"${INSTALL[@]}" -e ".[analysis]"

if [[ "$ENV" == eval ]]; then
  # t2v_metrics refuses to import unless an `ffmpeg` program is on PATH (it only uses it for
  # video), and HPC3 has none. imageio-ffmpeg, a t2v-metrics dependency, ships a static build:
  # link it into the venv's bin/, which is on PATH whenever the env is activated.
  FF=$(ls "$VENV"/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-* 2>/dev/null | head -1 || true)
  if [[ -n "$FF" ]]; then
    ln -sf "$FF" "$VENV/bin/ffmpeg" && echo "linked $VENV/bin/ffmpeg -> $FF"
  else
    echo "WARNING: imageio-ffmpeg's ffmpeg not found; t2v_metrics will refuse to import" >&2
  fi
fi

uv pip freeze --python "$VENV/bin/python" --exclude-editable > "env/$ENV.lock.txt"
grep -iE "^(torch|torchvision|diffusers|transformers|accelerate|t2v-metrics|open-clip-torch)==" "env/$ENV.lock.txt" || true
echo
echo "Built $VENV for $TARGET_PLATFORM; locked to env/$ENV.lock.txt."
echo "Packages were chosen for HPC3, so test there, not here:"
echo "  ssh sjsu3 \"cd ~/genprobe && sbatch slurm/probe_gpu.sbatch\""
