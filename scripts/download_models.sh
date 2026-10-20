#!/bin/bash
# Download model weights on **coe-hpc1** (it has internet) into $HF_HOME (~/hf_home);
# jobs on HPC3 then read them offline. Generators: only the diffusers-format files that
# configs/gen/*.yaml load, never the duplicate single-file checkpoints. Scorers: only the
# files genprobe/metrics/*.py load.
#
#   ssh -t sjsu "cd ~/genprobe && bash scripts/download_models.sh login"   once: paste an HF read token
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh sd15 sdxl"
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh all"        the four generators
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh scorers"    clip_l14 + qwen25vl7b
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh checkers"   grounding_dino_tiny + easyocr
#                                                                          (easyocr needs the aux env)
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh m2"         pickscore hpsv2 sd_safety nsfw_vit
#   ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh judge"      qwen3vl8b (VLM judge)
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

[[ $# -gt 0 ]] || { sed -n '2,18p' "$0"; exit 2; }
if [[ "$1" == all ]]; then set -- sd15 sdxl sd35m flux_schnell; fi
if [[ "$1" == scorers ]]; then set -- clip_l14 qwen25vl7b; fi
if [[ "$1" == checkers ]]; then set -- grounding_dino_tiny easyocr; fi
if [[ "$1" == m2 ]]; then set -- pickscore hpsv2 sd_safety nsfw_vit; fi
if [[ "$1" == judge ]]; then set -- qwen3vl8b; fi

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
    clip_l14)
      # CLIPScore (genprobe/metrics/clipscore.py): OpenAI CLIP ViT-L/14 as open_clip loads it.
      # Only the safetensors copy; open_clip tries it before the .bin.
      download clip_l14 timm/vit_large_patch14_clip_224.openai \
        --include "open_clip_model.safetensors" "open_clip_config.json" \
        --expect open_clip_model.safetensors
      ;;
    qwen25vl7b)
      # VQAScore backbone (genprobe/metrics/vqascore.py), ~16.6 GB. Not gated.
      download qwen25vl7b Qwen/Qwen2.5-VL-7B-Instruct \
        --include "*.json" "*.txt" "*.safetensors" \
        --expect config.json preprocessor_config.json tokenizer_config.json chat_template.json model.safetensors.index.json
      ;;
    grounding_dino_tiny)
      # Detector checker (genprobe/metrics/detect.py), ~0.7 GB. Not gated. Skips the .bin copy.
      download grounding_dino_tiny IDEA-Research/grounding-dino-tiny \
        --include "*.json" "*.txt" "*.safetensors" \
        --expect config.json preprocessor_config.json tokenizer_config.json model.safetensors
      ;;
    pickscore)
      # Preference score (genprobe/metrics/pickscore.py), CLIP-H, ~3.9 GB. Processor files included.
      download pickscore yuvalkirstain/PickScore_v1 \
        --include "*.json" "*.txt" "model.safetensors" \
        --expect config.json preprocessor_config.json tokenizer_config.json model.safetensors
      ;;
    hpsv2)
      # HPSv2.1 (genprobe/metrics/hpsv2.py): only the v2.1 checkpoint; the repo has no safetensors.
      download hpsv2 xswu/HPSv2 --include "HPS_v2.1_compressed.pt" --expect HPS_v2.1_compressed.pt --weights-ext .pt
      ;;
    sd_safety)
      # SD safety checker (genprobe/metrics/safety.py), ~1.2 GB; the repo only has a .bin.
      download sd_safety CompVis/stable-diffusion-safety-checker \
        --include "*.json" "pytorch_model.bin" --expect config.json preprocessor_config.json pytorch_model.bin \
        --weights-ext .bin
      ;;
    nsfw_vit)
      # Second safety classifier (genprobe/metrics/safety.py), ViT-B, ~0.3 GB. Skips optimizer/YOLO files.
      download nsfw_vit Falconsai/nsfw_image_detection \
        --include "config.json" "preprocessor_config.json" "model.safetensors" \
        --expect config.json preprocessor_config.json model.safetensors
      ;;
    qwen3vl8b)
      # VLM judge (genprobe/judge.py, D-027), ~17.5 GB. Not gated. A different model from
      # VQAScore's Qwen2.5-VL-7B, as the H3 pre-registration requires.
      download qwen3vl8b Qwen/Qwen3-VL-8B-Instruct \
        --include "*.json" "*.txt" "*.safetensors" \
        --expect config.json preprocessor_config.json tokenizer_config.json chat_template.json model.safetensors.index.json
      ;;
    easyocr)
      # OCR checker (genprobe/metrics/ocr.py): CRAFT detector + english_g2 recognizer, ~100 MB,
      # into $EASYOCR_MODULE_PATH/model. URLs and md5 sums come from the installed EasyOCR's
      # own config file (aux env), so they match the version the jobs run.
      echo "== easyocr"
      cfg=$(ls "$GENPROBE_ENVS"/aux/lib/python*/site-packages/easyocr/config.py 2>/dev/null | head -1 || true)
      if [[ -z "$cfg" ]]; then
        echo "!! easyocr: build the aux env first: bash scripts/setup_envs.sh aux" >&2
        FAILED+=(easyocr)
      elif ! uv run --no-project --python 3.11 python scripts/easyocr_download.py "$cfg" \
          "$EASYOCR_MODULE_PATH/model" craft english_g2; then
        FAILED+=(easyocr)
      fi
      ;;
    *) echo "unknown model '$m' (expected login, sd15, sdxl, sd35m, flux_schnell, all, clip_l14, qwen25vl7b," \
            "scorers, grounding_dino_tiny, easyocr, checkers, pickscore, hpsv2, sd_safety, nsfw_vit, m2," \
            "qwen3vl8b or judge)" >&2; exit 2 ;;
  esac
done

echo
echo "== cache contents ($HF_HOME/hub, apparent size)"
du -sh --apparent-size "$HF_HOME"/hub/models--* "$EASYOCR_MODULE_PATH"/model 2>/dev/null || true
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED: ${FAILED[*]} — fix the cause and rerun just those (finished files are reused)" >&2
  exit 1
fi
