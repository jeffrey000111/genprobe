#!/bin/bash
# Download GenAI-Bench-Video (GenAI-Bench-800: 800 prompts x 4 video models = 3,200 videos, 3 human
# ratings each; MIT) on **coe-hpc1** into data/external/genai_bench_video/ (not in git), for the video
# meta-evaluation (guide §11.3, D-035). File names follow t2v_metrics' GenAIBench_Video loader.
#
#   ssh sjsu "cd ~/genprobe && bash scripts/fetch_genai_bench_video.sh"
#
# genai_video.json: {"<index>": {"prompt", "models": {model: [3 ratings 1-5]}, ...}, ...}
# genai_skills.json: {skill name: [prompt indices]}; <model>.zip unzips to <model>/<index>.mp4.
# The dataset revision is resolved once and pinned in SOURCE. Re-running skips finished files.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/paths.sh
REPO=zhiqiulin/GenAI-Bench-800
OUT=data/external/genai_bench_video
MODELS=(Floor33 Gen2 Pika_v1 Modelscope)

has_internet || { echo "No internet on $(hostname); run on coe-hpc1: ssh sjsu \"cd ~/genprobe && bash $0\"" >&2; exit 2; }
mkdir -p "$OUT"
if [[ -s "$OUT/SOURCE" ]]; then
  REV=$(cut -d@ -f2 "$OUT/SOURCE")
else
  REV=$(curl -sSfL "https://huggingface.co/api/datasets/$REPO" | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
fi
echo "$REPO@$REV"
files=(genai_video.json genai_skills.json)
for m in "${MODELS[@]}"; do files+=("$m.zip"); done
for f in "${files[@]}"; do
  if [[ -s "$OUT/$f" ]]; then echo "have $f"; continue; fi
  curl -sSfL --retry 3 -o "$OUT/$f.part" "https://huggingface.co/datasets/$REPO/resolve/$REV/$f"
  mv "$OUT/$f.part" "$OUT/$f"
  echo "got  $f ($(du -h "$OUT/$f" | cut -f1))"
done
echo "$REPO@$REV" > "$OUT/SOURCE"
(cd "$OUT" && sha256sum "${files[@]}" > SHA256SUMS)
for m in "${MODELS[@]}"; do
  unzip -q -n "$OUT/$m.zip" -d "$OUT"
  echo "unzipped $m: $(find "$OUT/$m" -name '*.mp4' | wc -l) videos"
done
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/genai_video.json"))
k = sorted(d)[:2]
print("prompts:", len(d), "first keys:", k, "fields:", sorted(d[k[0]]), "models:", sorted(d[k[0]]["models"]))
PY
