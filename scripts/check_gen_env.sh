#!/bin/bash
# Check the gen env and the pinned generator weights on a CPU node (cpuqs, minutes) before
# queueing generation on H100s. Loads no weights. Run on HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/check_gen_env.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:10:00 -J check-gen --chdir="$PWD" bash -c '
  source scripts/paths.sh
  source "$GENPROBE_ENVS/gen/bin/activate"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python scripts/check_gen_configs.py'
