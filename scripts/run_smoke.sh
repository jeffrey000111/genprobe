#!/bin/bash
# One command for the M1 smoke run on HPC3: generate the 20 smoke prompts with all four models
# (seed 0), score every image with CLIPScore and VQAScore, then write results/smoke/summary.md.
# Run on HPC3's login node (it only calls sbatch), after scripts/setup_envs.sh and
# scripts/download_models.sh have been run once on coe-hpc1:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_smoke.sh"
#   scripts/hpc_sync.sh pull && cat results/smoke/summary.md        (on the laptop, later)
#
# Everything resumes: rerunning skips images and scores that already exist. Scoring waits for
# all generation jobs (afterany, so a failed model still leaves a summary that shows the gap).
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${PROMPTS:-data/prompts/smoke.jsonl}
MODELS=(${MODELS:-sd15 sdxl sd35m flux_schnell})
GEN_TIME=${GEN_TIME:-01:00:00}   # 20 images per model take minutes; short limits schedule sooner

command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
mkdir -p logs
ids=()
for m in "${MODELS[@]}"; do
  [[ -f configs/gen/$m.yaml ]] || { echo "no config configs/gen/$m.yaml" >&2; exit 2; }
  id=$(sbatch --parsable --array=0 --time="$GEN_TIME" --job-name="gen-$m" \
    slurm/gen.sbatch "configs/gen/$m.yaml" "$PROMPTS" 1 --seeds 0)
  id=${id%%;*}
  echo "submitted generation $m: job $id"
  ids+=("$id")
done
DEPENDENCY="afterany:$(IFS=:; echo "${ids[*]}")" PROMPTS="$PROMPTS" bash scripts/submit_smoke_scoring.sh
