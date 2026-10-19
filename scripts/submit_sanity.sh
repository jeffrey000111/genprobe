#!/bin/bash
# Submit the metric sanity checks (guide §7.9, D-023): one job per metric (H100, or cpuqs for
# the checkers), then one CPU job that writes results/sanity/summary.md. Run on HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/submit_sanity.sh"
#   ssh sjsu3 "cd ~/genprobe && METRICS='pickscore hpsv2' bash scripts/submit_sanity.sh"
#
# Knobs (env): METRICS (clipscore vqascore ocr detect), CPU_METRICS (ocr detect),
# PROMPTS (data/prompts/prompts_v1.jsonl), DEPENDENCY=afterany:<ids> to wait for other jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
METRICS=(${METRICS:-clipscore vqascore ocr detect})
CPU_METRICS=" ${CPU_METRICS-ocr detect} "
PROMPTS=${PROMPTS:-data/prompts/prompts_v1.jsonl}
DEPENDENCY=${DEPENDENCY:-}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3)" >&2; exit 2; }
mkdir -p logs
ids=()
for m in "${METRICS[@]}"; do
  script=slurm/sanity.sbatch; where=H100
  if [[ "$CPU_METRICS" == *" $m "* ]]; then script=slurm/sanity_cpu.sbatch; where=cpuqs; fi
  id=$(sbatch --parsable --job-name="sanity-$m" ${DEPENDENCY:+--dependency=$DEPENDENCY} \
    "$script" "$m" --prompts "$PROMPTS")
  echo "submitted sanity $m ($where): job ${id%%;*}"
  ids+=("${id%%;*}")
done
dep=$(IFS=:; echo "${ids[*]}")
rid=$(sbatch --parsable --dependency="afterany:$dep" -p cpuqs -c 1 --mem=2G -t 00:05:00 -J sanity-report \
  -o "logs/%x_%j.out" --wrap "bash -c 'source slurm/common.sh && activate_env eval && python -m genprobe.sanity --report'")
echo "submitted sanity report: job ${rid%%;*} (after $dep)"
