#!/bin/bash
# Score one prompt set's images with each metric (one job array per metric on the H100s), then
# write its summary on a CPU node once scoring is done. Run on HPC3 (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/submit_scoring.sh data/prompts/smoke.jsonl"
#   ssh sjsu3 "cd ~/genprobe && METRICS=vqascore bash scripts/submit_scoring.sh data/prompts/smoke.jsonl"
#
# Knobs (env): METRICS, SUMMARY_METRICS, SCORE_SHARDS (1), SCORE_TIME (00:45:00), SPLITS (all;
# e.g. "test dev" writes results/<set>_test and results/<set>_dev), and DEPENDENCY=afterany:<ids>
# to wait for other jobs (scripts/run_prompt_set.sh uses it). Scoring resumes: samples already
# scored with the same metric_version are skipped.
# Logs: logs/score-<metric>_<id>_<shard>.out, logs/summarize_<id>.out
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:-data/prompts/smoke.jsonl}
SAMPLES=${SAMPLES:-data/manifests/gen_*.jsonl}
TAG=${TAG:-$(basename "$PROMPTS" .jsonl)}
METRICS=(${METRICS:-clipscore vqascore})
SUMMARY_METRICS=(${SUMMARY_METRICS:-clipscore vqascore})
SCORE_SHARDS=${SCORE_SHARDS:-1}
# Short limits let Slurm backfill the jobs into gaps on busy nodes.
SCORE_TIME=${SCORE_TIME:-00:45:00}
SPLITS=(${SPLITS:-all})
DEPENDENCY=${DEPENDENCY:-}

command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "no prompts file $PROMPTS" >&2; exit 2; }
if [[ -z "$DEPENDENCY" ]]; then
  ls $SAMPLES >/dev/null 2>&1 || { echo "no manifests match $SAMPLES: generate the images first" >&2; exit 2; }
fi
mkdir -p logs

ids=()
for m in "${METRICS[@]}"; do
  id=$(sbatch --parsable --array=0-$((SCORE_SHARDS - 1)) --time="$SCORE_TIME" --job-name="score-$m" \
    ${DEPENDENCY:+--dependency=$DEPENDENCY} \
    slurm/score.sbatch "$m" "$SAMPLES" "$TAG" "$SCORE_SHARDS" --prompts "$PROMPTS")
  id=${id%%;*}
  echo "submitted $m ($SCORE_SHARDS shard(s)): job $id"
  ids+=("$id")
done
dep=$(IFS=:; echo "${ids[*]}")
for s in "${SPLITS[@]}"; do
  split=$s; [[ "$s" == all ]] && split=""
  sid=$(sbatch --parsable --dependency="afterany:$dep" --export=ALL,SPLIT="$split" \
    slurm/summarize.sbatch "$PROMPTS" "${SUMMARY_METRICS[@]}")
  echo "submitted summary ($s): job ${sid%%;*} (starts after $dep)"
done
echo "follow with: squeue -u \$USER"
