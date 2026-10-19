#!/bin/bash
# Score one prompt set's images with each metric (one job array per metric on the H100s), then
# write its summary on a CPU node once scoring is done. Run on HPC3 (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/submit_scoring.sh data/prompts/smoke.jsonl"
#   ssh sjsu3 "cd ~/genprobe && METRICS=vqascore bash scripts/submit_scoring.sh data/prompts/smoke.jsonl"
#   ssh sjsu3 "cd ~/genprobe && METRICS='ocr detect' SUMMARY_METRICS='clipscore vqascore ocr detect' bash scripts/submit_scoring.sh data/prompts/smoke.jsonl"
#
# Metrics in CPU_METRICS (default: the checkers ocr and detect) run on cpuqs with
# slurm/score_cpu.sbatch and CPU_SCORE_TIME (01:00:00); the rest on the H100s (D-019).
#
# Knobs (env): METRICS, SUMMARY_METRICS, SCORE_SHARDS (1), SCORE_TIME (00:45:00), SPLITS (all;
# e.g. "test dev" writes results/<set>_test and results/<set>_dev), DEPENDENCY=afterany:<ids>
# to wait for other jobs (scripts/run_prompt_set.sh uses it), and SUMMARY_AFTER=<id>:<id> for
# summaries that must also wait for scoring jobs submitted earlier (e.g. the H100 metrics), and
# IDS_FILE=<path> to append the scoring job ids there (for jobs that must wait for them). Scoring resumes: samples already
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
CPU_METRICS=" ${CPU_METRICS-ocr detect} "
CPU_SCORE_TIME=${CPU_SCORE_TIME:-01:00:00}
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
  script=slurm/score.sbatch; t=$SCORE_TIME; where="H100"
  if [[ "$CPU_METRICS" == *" $m "* ]]; then script=slurm/score_cpu.sbatch; t=$CPU_SCORE_TIME; where="cpuqs"; fi
  id=$(sbatch --parsable --array=0-$((SCORE_SHARDS - 1)) --time="$t" --job-name="score-$m" \
    ${DEPENDENCY:+--dependency=$DEPENDENCY} \
    "$script" "$m" "$SAMPLES" "$TAG" "$SCORE_SHARDS" --prompts "$PROMPTS")
  id=${id%%;*}
  echo "submitted $m ($SCORE_SHARDS shard(s), $where): job $id"
  [[ -z "${IDS_FILE:-}" ]] || echo "$id" >> "$IDS_FILE"
  ids+=("$id")
done
dep=$(IFS=:; echo "${ids[*]}")${SUMMARY_AFTER:+:$SUMMARY_AFTER}
for s in "${SPLITS[@]}"; do
  split=$s; [[ "$s" == all ]] && split=""
  sid=$(sbatch --parsable --dependency="afterany:$dep" --export=ALL,SPLIT="$split" \
    slurm/summarize.sbatch "$PROMPTS" "${SUMMARY_METRICS[@]}")
  echo "submitted summary ($s): job ${sid%%;*} (starts after $dep)"
done
echo "follow with: squeue -u \$USER"
