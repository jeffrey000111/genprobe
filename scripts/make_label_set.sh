#!/bin/bash
# Blind labelling set of dev images for checking the checkers (genprobe/label_set.py, D-024):
# numbered contact sheets with no scores and no model names, a labels.csv to fill in, and the
# item -> sample_id key. Runs on a CPU node. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_label_set.sh"
#
# Then copy the folder to the laptop (data/ is git-ignored):
#   mkdir -p data/review/v1_dev && scp 'sjsu3:genprobe/data/review/v1_dev/*' data/review/v1_dev/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-data/review/v1_dev}
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:10:00 -J label-set --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.label_set make --out '$OUT'"
