#!/bin/bash
# Prompts, smoke subset and image manifest for GenAI-Bench (genprobe/genai_bench.py), on a CPU
# node; opens every image once to check it. On HPC3, after fetch_genai_bench.sh images (coe-hpc1):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/prepare_genai_bench.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:20:00 -J prep-genai --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.genai_bench prepare --verify"
