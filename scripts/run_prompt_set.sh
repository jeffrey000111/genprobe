#!/bin/bash
# Generate, score and summarize one prompt set on HPC3, chained with Slurm dependencies.
# Run on HPC3's login node (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_prompt_set.sh data/prompts/prompts_v1.jsonl"
#
# Knobs (env): MODELS (all four), GEN_SHARDS (4), GEN_CONCURRENCY (2 shards of a model at a
# time), GEN_TIME (02:00:00), SEEDS (empty = each config's seeds), plus the scoring knobs of
# scripts/submit_scoring.sh. Scoring starts when every generation job has ended (afterany, so a
# failed shard still leaves a summary that shows the gap). Everything resumes: rerunning skips
# images and scores that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:?usage: run_prompt_set.sh <prompts.jsonl>}
MODELS=(${MODELS:-sd15 sdxl sd35m flux_schnell})
GEN_SHARDS=${GEN_SHARDS:-4}
GEN_CONCURRENCY=${GEN_CONCURRENCY:-2}
GEN_TIME=${GEN_TIME:-02:00:00}
SEEDS=${SEEDS:-}

command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "no prompts file $PROMPTS" >&2; exit 2; }
mkdir -p logs
ids=()
for m in "${MODELS[@]}"; do
  [[ -f configs/gen/$m.yaml ]] || { echo "no config configs/gen/$m.yaml" >&2; exit 2; }
  id=$(sbatch --parsable --array=0-$((GEN_SHARDS - 1))%"$GEN_CONCURRENCY" --time="$GEN_TIME" \
    --job-name="gen-$m" slurm/gen.sbatch "configs/gen/$m.yaml" "$PROMPTS" "$GEN_SHARDS" ${SEEDS:+--seeds $SEEDS})
  id=${id%%;*}
  echo "submitted generation $m ($GEN_SHARDS shard(s)): job $id"
  ids+=("$id")
done
DEPENDENCY="afterany:$(IFS=:; echo "${ids[*]}")" bash scripts/submit_scoring.sh "$PROMPTS"
