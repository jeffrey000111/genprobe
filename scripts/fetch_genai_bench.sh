#!/bin/bash
# Download GenAI-Bench-1600 (Apache-2.0) on **coe-hpc1** into data/external/genai_bench/ (not in git).
#
#   ssh sjsu "cd ~/genprobe && bash scripts/fetch_genai_bench.sh"          prompts, skill tags, human ratings (~1.5 MB)
#   ssh sjsu "cd ~/genprobe && bash scripts/fetch_genai_bench.sh images"   + the 6 models' image zips (~3.5 GB, module B)
#
# genai_image.json: {"00000": {"id", "prompt", "prompt in Chinese", "models": {model: [3 ratings 1-5]}}, ...}
# genai_skills.json: {skill name: [prompt indices]}
# The dataset revision is pinned so prompt set v1 can be rebuilt exactly. Re-running skips
# files that are already there.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/paths.sh
REPO=BaiqiL/GenAI-Bench-1600
REV=e51e8a8070fadfbfe1f160436bf14e7664e87247   # main as of 2024-08-24
OUT=data/external/genai_bench

has_internet || { echo "No internet on $(hostname); run on coe-hpc1: ssh sjsu \"cd ~/genprobe && bash $0 $*\"" >&2; exit 2; }
mkdir -p "$OUT"
files=(README.md genai_image.json genai_skills.json)
if [[ "${1:-}" == images ]]; then
  files+=(DALLE_3.zip DeepFloyd_I_XL_v1.zip Midjourney_6.zip SDXL_2_1.zip SDXL_Base.zip SDXL_Turbo.zip)
fi
for f in "${files[@]}"; do
  if [[ -s "$OUT/$f" ]]; then echo "have $f"; continue; fi
  curl -sSfL --retry 3 -o "$OUT/$f.part" "https://huggingface.co/datasets/$REPO/resolve/$REV/$f"
  mv "$OUT/$f.part" "$OUT/$f"
  echo "got  $f ($(du -h "$OUT/$f" | cut -f1))"
done
echo "$REPO@$REV" > "$OUT/SOURCE"
(cd "$OUT" && sha256sum "${files[@]}" > SHA256SUMS)
echo "sha256 recorded in $OUT/SHA256SUMS"
