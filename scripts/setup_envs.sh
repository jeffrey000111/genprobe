#!/bin/bash
# Build a Python virtualenv for HPC3 jobs. Run it on **coe-hpc1**, the only SJSU HPC machine
# with internet. /home is shared, so the env is usable on HPC3 and its compute nodes as is.
#
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh gen"
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh eval"
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh aux"    EasyOCR for the OCR checker
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh judge"  vLLM for the VLM judge
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh video"  Wan2.2 video generation (module D)
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh vbench" VBench video quality dimensions (module D)
#   ssh sjsu "cd ~/genprobe && bash scripts/setup_envs.sh train"  Diffusion-DPO training (module D)
#
# coe-hpc1 runs CentOS 7 (glibc 2.17) and HPC3 runs glibc 2.34, so packages are selected for
# HPC3's platform with uv's --python-platform, not for the machine doing the install. uv and
# its managed Python only need glibc 2.17, so both run on either machine (D-012).
# Afterwards, test the env on a GPU node: sbatch slurm/probe_gpu.sbatch (on HPC3).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:?usage: setup_envs.sh gen|eval|aux|judge|video|vbench|train}
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
# Fewer parallel writes into the env on NFS.
export UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-4}

# On NFS, a file that is still open when uv tidies up becomes an `.nfs*` placeholder that can't
# be removed yet ("Device or resource busy", os error 16; first seen installing vLLM's numba
# into the judge env). The placeholder goes away once the handle closes and installs are
# idempotent, so try again.
install() {
  local i
  for i in 1 2 3; do
    "${INSTALL[@]}" "$@" && return 0
    echo "!! install attempt $i failed; retrying in 20 s (NFS placeholder files clear once closed)" >&2
    sleep 20
  done
  return 1
}

case "$ENV" in
  gen)
    # PyPI's Linux torch wheels bundle their CUDA runtime; HPC3's driver (590, CUDA 13.1)
    # supports all current ones, so no special index is needed.
    install torch torchvision diffusers transformers accelerate sentencepiece protobuf safetensors
    ;;
  eval)
    # Require t2v-metrics>=3.0 (Qwen2.5-VL backbones, transformers>=4.52). Left unpinned, the
    # resolver kept the newest torch and fell back to t2v-metrics 1.2 + transformers 4.36.
    install "t2v-metrics>=3.0" torchvision open_clip_torch
    ;;
  aux)
    # Scorers whose dependencies shouldn't touch the eval env: EasyOCR (opencv, scikit-image,
    # shapely, ...) for the OCR checker. Its weights: download_models.sh easyocr.
    install torch torchvision easyocr
    ;;
  judge)
    # VLM judge (guide §9.4, D-027): vLLM runs Qwen3-VL-8B-Instruct offline. vLLM pins its own
    # torch and transformers, so it gets an env of its own; uv picks the newest vLLM wheel.
    install vllm
    ;;
  video)
    # Wan2.2 TI2V-5B (guide §11.2, D-036). Its own env, so the image generators' gen env is never
    # re-resolved mid-project; the same diffusers release as gen. imageio-ffmpeg writes the mp4s
    # (it ships a static ffmpeg); ftfy is what Wan's prompt cleaning uses when present.
    install torch torchvision "diffusers==0.41.0" transformers accelerate sentencepiece protobuf safetensors \
      ftfy imageio imageio-ffmpeg av
    ;;
  vbench)
    # VBench (guide §11.3, D-036), six dimensions for custom videos. Its README asks for torch built
    # with CUDA <= 12.1 and transformers 4.33.2; torch 2.4.1's PyPI wheels are built with CUDA 12.1.
    # VBench itself is not pip-installed (its setup.py demands a visible GPU at install time): the
    # pinned source checkout from scripts/download_vbench.sh goes on the path through a .pth file.
    # openai-clip still imports pkg_resources, which setuptools dropped: keep an older setuptools.
    install "torch==2.4.1" "torchvision==0.19.1" "numpy<2" "timm>=0.9,<=1.0.12" "transformers==4.33.2" \
      openai-clip decord pyiqa easydict omegaconf opencv-python-headless scipy scikit-image tqdm av "setuptools<70"
    ;;
  train)
    # Diffusion-DPO LoRA on SD 1.5 (guide §11.4, D-037): diffusers' research script needs peft,
    # datasets and (imported unconditionally) wandb; jobs run it with WANDB_MODE=disabled. The same
    # diffusers release as gen, so the evaluation images come from the same pipeline code.
    # transformers stays on 5.x: diffusers 0.41 needs huggingface-hub >= 1.32, which no 4.x release
    # accepts (pinning transformers<5 resolved to 4.12, whose tokenizers no longer builds). The
    # script's offline tokenizer lookup is avoided by giving it the local snapshot (slurm/dpo.sbatch).
    install torch torchvision "diffusers==0.41.0" transformers accelerate peft datasets wandb tensorboard \
      ftfy Jinja2 sentencepiece protobuf safetensors
    ;;
  *) echo "unknown env '$ENV' (expected gen, eval, aux, judge, video, vbench or train)" >&2; exit 2 ;;
esac
install -e ".[analysis]"
if [[ "$ENV" == train ]]; then
  # diffusers' DPO research script at the tag of the installed diffusers, checked against a pinned hash.
  DPO_URL=https://raw.githubusercontent.com/huggingface/diffusers/v0.41.0/examples/research_projects/diffusion_dpo/train_diffusion_dpo.py
  DPO_SHA256=3d84934230dab97a4556b8382c1fcb937ffe38be636b76d35ddf731e82988bc0
  mkdir -p "$GENPROBE_ENVS/train-src"
  curl -sSfL --retry 3 -o "$GENPROBE_ENVS/train-src/train_diffusion_dpo.py" "$DPO_URL"
  echo "$DPO_SHA256  $GENPROBE_ENVS/train-src/train_diffusion_dpo.py" | sha256sum -c -
fi
if [[ "$ENV" == vbench ]]; then
  SITE=$(ls -d "$VENV"/lib/python*/site-packages | head -1)
  echo "$GENPROBE_ENVS/vbench-src" > "$SITE/vbench_src.pth"
  echo "VBench source path: $GENPROBE_ENVS/vbench-src (scripts/download_vbench.sh clones it)"
fi

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
grep -iE "^(torch|torchvision|diffusers|transformers|accelerate|t2v-metrics|open-clip-torch|easyocr|vllm)==" "env/$ENV.lock.txt" || true
echo
echo "Built $VENV for $TARGET_PLATFORM; locked to env/$ENV.lock.txt."
echo "Packages were chosen for HPC3, so test there, not here:"
echo "  ssh sjsu3 \"cd ~/genprobe && sbatch slurm/probe_gpu.sbatch\""
