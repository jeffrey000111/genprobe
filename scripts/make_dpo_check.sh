#!/bin/bash
# Blind human check of DPO vs base SD 1.5 (genprobe/dpo_check.py, docs/dpo_check_protocol.md, D-037),
# on a CPU node, after the DPO test images exist. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_dpo_check.sh"           # package
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_dpo_check.sh analyze"   # after labelling
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-make}
case "$MODE" in
  make) CMD="python -m genprobe.dpo_check make" ;;
  analyze) CMD="python -m genprobe.dpo_check analyze --labels data/labels/dpo_check_labels.csv" ;;
  *) echo "usage: $0 [make|analyze]" >&2; exit 2 ;;
esac
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:20:00 -J dpo-check --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  $CMD"
