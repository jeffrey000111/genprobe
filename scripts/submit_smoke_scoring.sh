#!/bin/bash
# M1: score the smoke images (4 models x 20 prompts, seed 0) with CLIPScore and VQAScore on
# one H100 each, then build the summary table on a CPU node once both have finished.
# Run on HPC3's login node (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/submit_smoke_scoring.sh"
#
# Needs the scoring weights first (on coe-hpc1): bash scripts/download_models.sh scorers
# Logs: logs/score-clipscore_<id>_0.out, logs/score-vqascore_<id>_0.out, logs/summarize_<id>.out
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${PROMPTS:-data/prompts/smoke.jsonl}
SAMPLES=${SAMPLES:-data/manifests/gen_*.jsonl}
TAG=${TAG:-$(basename "$PROMPTS" .jsonl)}
METRICS=(${METRICS:-clipscore vqascore})

command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
ls $SAMPLES >/dev/null 2>&1 || { echo "no manifests match $SAMPLES: generate the smoke images first" >&2; exit 2; }
mkdir -p logs

ids=()
for m in "${METRICS[@]}"; do
  id=$(sbatch --parsable --array=0 --job-name="score-$m" slurm/score.sbatch "$m" "$SAMPLES" "$TAG" 1 --prompts "$PROMPTS")
  id=${id%%;*}
  echo "submitted $m: job $id"
  ids+=("$id")
done
dep=$(IFS=:; echo "${ids[*]}")
sid=$(sbatch --parsable --dependency="afterany:$dep" slurm/summarize.sbatch "$PROMPTS" "${METRICS[@]}")
echo "submitted summary: job ${sid%%;*} (starts after $dep)"
echo "follow with: squeue -u \$USER"
