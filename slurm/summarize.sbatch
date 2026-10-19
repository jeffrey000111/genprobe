#!/bin/bash
# Summary tables for one prompt set (CPU only; reads manifests and scores, writes results/).
#
#   sbatch slurm/summarize.sbatch data/prompts/smoke.jsonl clipscore vqascore
#   args: <prompts.jsonl> <metric>...      SPLIT=test|dev (env) restricts to one split
#
# Output: results/<prompts stem>[_<split>]/summary.md, summary.json, per_sample.csv, then the
# leaderboard, paired comparisons and figures (genprobe.figures). Bring them to the laptop with
# scripts/hpc_sync.sh pull.
#SBATCH --job-name=summarize
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
PROMPTS=${1:?prompts jsonl}; shift
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
activate_env "${GENPROBE_ENV:-eval}"
cd "$GENPROBE_ROOT"
METRICS=("$@"); [[ ${#METRICS[@]} -gt 0 ]] || METRICS=(clipscore vqascore)
status=0
python -m genprobe.summarize --prompts "$PROMPTS" --metrics "${METRICS[@]}" ${SPLIT:+--split "$SPLIT"} || status=$?
# Figures even when some scores are missing (the summary says which), so partial runs are visible.
OUT="results/$(basename "$PROMPTS" .jsonl)${SPLIT:+_$SPLIT}"
python -m genprobe.figures --results "$OUT" --prompts "$PROMPTS" ${SPLIT:+--split "$SPLIT"} || status=$?
exit $status
