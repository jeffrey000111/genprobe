#!/bin/bash
# Check the eval env on a CPU node (cpuqs: short queue, no GPU needed): scoring weights in the
# HF cache, open_clip's CLIP config, Qwen2.5-VL weight names vs this transformers version,
# and the Qwen processor / chat template / image preprocessing on real smoke images.
# Loads no weights (the model is built on the meta device). Run on HPC3 before queueing
# scoring jobs for an H100:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/check_eval_env.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
# --chdir: same repo path on the compute node (shared /home)
srun -p cpuqs -c 2 --mem=16G -t 00:10:00 -J check-eval --chdir="$PWD" bash -c '
  source scripts/paths.sh
  source "$GENPROBE_ENVS/eval/bin/activate"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo "node: $(hostname)"
  python scripts/check_eval_env.py'
