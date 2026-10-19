#!/bin/bash
# Report figures and example panels for prompt set v1, on a CPU node: redraws the figures
# (genprobe/figures.py) for the test and dev summaries and picks the example images from the
# test split (genprobe/examples.py, D-025). On HPC3, after the summaries exist:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_report_assets.sh"
#
# Output: results/prompts_v1_{test,dev}/fig_* + CSVs, results/examples/*.jpg + index.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${PROMPTS:-data/prompts/prompts_v1.jsonl}
STEM=$(basename "$PROMPTS" .jsonl)
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:15:00 -J report-assets --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.figures --results results/${STEM}_test --prompts '$PROMPTS' --split test &&
  python -m genprobe.figures --results results/${STEM}_dev --prompts '$PROMPTS' --split dev &&
  python -m genprobe.examples --results results/${STEM}_test --prompts '$PROMPTS'"
