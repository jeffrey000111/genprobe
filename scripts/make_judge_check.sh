#!/bin/bash
# Blind human check of the judge (genprobe/judge_check.py, docs/judge_check_protocol.md, D-033), on a
# CPU node. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_judge_check.sh"           # package
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_judge_check.sh analyze"   # after labelling
#
# make:    data/judge_check/judge_check_package.zip (copy to the laptop) and data/judge_check/key.json
#          (leave it closed until labelling is done); items drawn from module C's test run (seed 0).
# analyze: data/labels/judge_check_labels.csv + the key + the judge's grades -> results/judge_check/
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-make}
case "$MODE" in
  make) CMD="python -m genprobe.judge_check make" ;;
  analyze) CMD="python -m genprobe.judge_check analyze --labels data/labels/judge_check_labels.csv" ;;
  *) echo "usage: $0 [make|analyze]" >&2; exit 2 ;;
esac
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:20:00 -J judge-check --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  $CMD"
