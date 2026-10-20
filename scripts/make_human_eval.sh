#!/bin/bash
# Package for the own human evaluation (genprobe/human_eval.py, docs/human_eval_protocol.md), on a
# CPU node: draws the items (seed 0), writes 768 px images and the offline labelling page into a
# zip, and the answer key separately. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_human_eval.sh"
#
# Output: data/human_eval/human_eval_package.zip (copy to the laptop), data/human_eval/key.json
# (leave it closed until labelling is done).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:20:00 -J human-eval --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.human_eval make"
