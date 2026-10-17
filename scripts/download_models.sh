#!/bin/bash
# Download generator weights on **coe-hpc1** (it has internet) into $HF_HOME (~/hf_home);
# jobs on HPC3 then read them offline. Only the diffusers-format files that
# configs/gen/*.yaml load are fetched, never the duplicate single-file checkpoints.
#
#   ssh -t sjsu "cd ~/genprobe && bash scripts/download_models.sh login"   once: paste an HF read token
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh sd15 sdxl"
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh all"
#
# SD3.5 Medium and FLUX.1-schnell are gated: accept their terms on huggingface.co first.
# Re-running is safe: files already in the cache are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/paths.sh
export HF_HUB_OFFLINE=0

if ! has_internet; then
  echo "No internet on $(hostname). Run this on coe-hpc1: ssh sjsu \"cd ~/genprobe && bash $0 $*\"" >&2
  exit 2
fi
if ! command -v uvx >/dev/null 2>&1; then
  echo "uv not found; run scripts/setup_envs.sh gen first (it installs uv)" >&2
  exit 2
fi
ensure_dirs
mkdir -p "$(dirname "$HF_TOKEN_PATH")"
HF=(uvx --python 3.11 --from huggingface_hub hf)   # CLI, used only for `auth login`
# Downloads go through the Python API (scripts/hf_download.py): the CLI's multi-pattern
# --include syntax changed in huggingface_hub 2.x and silently dropped the config files.
PYDL=(uv run --no-project --python 3.11 --with "huggingface_hub>=1.0" python scripts/hf_download.py)
FAILED=()

download() {
  local name=$1; shift
  echo "== $name"
  if ! "${PYDL[@]}" "$@"; then
    echo "!! $name failed (gated repo not accepted on huggingface.co? network?)" >&2
    FAILED+=("$name")
  fi
}

[[ $# -gt 0 ]] || { sed -n '2,13p' "$0"; exit 2; }
if [[ "$1" == all ]]; then set -- sd15 sdxl sd35m flux_schnell; fi

for m in "$@"; do
  case "$m" in
    login)
      "${HF[@]}" auth login
      chmod 600 "$HF_TOKEN_PATH" 2>/dev/null || true
      ;;
    sd15)
      download sd15 stable-diffusion-v1-5/stable-diffusion-v1-5 \
        --include "*.json" "*.txt" "*/*fp16.safetensors"
      ;;
    sdxl)
      download sdxl stabilityai/stable-diffusion-xl-base-1.0 \
        --include "*.json" "*.txt" "*/*fp16.safetensors"
      ;;
    sd35m)
      # text_encoders/ holds extra single-file encoders for ComfyUI; diffusers doesn't use them
      download sd35m stabilityai/stable-diffusion-3.5-medium \
        --include "*.json" "*.txt" "*.model" "*/*.safetensors" --exclude "text_encoders/*"
      ;;
    flux_schnell)
      download flux_schnell black-forest-labs/FLUX.1-schnell \
        --include "*.json" "*.txt" "*.model" "*/*.safetensors"
      ;;
    *) echo "unknown model '$m' (expected login, sd15, sdxl, sd35m, flux_schnell or all)" >&2; exit 2 ;;
  esac
done

echo
echo "== cache contents ($HF_HOME/hub, apparent size)"
du -sh --apparent-size "$HF_HOME"/hub/models--* 2>/dev/null || true
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED: ${FAILED[*]} — fix the cause and rerun just those (finished files are reused)" >&2
  exit 1
fi
