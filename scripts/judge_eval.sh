#!/bin/bash
# Re-run the analysis of a judge round (genprobe/judge_eval.py) on a CPU node, without the GPU
# part (data/judge/<tag>.jsonl already exists). On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/judge_eval.sh test_r1 test"
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag}; ROUND=${2:?round: dev or test}
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:15:00 -J judge-eval --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/judge/bin/activate\"
  python -m genprobe.judge_eval --tag '$TAG' --round '$ROUND'"
