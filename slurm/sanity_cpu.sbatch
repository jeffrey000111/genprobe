#!/bin/bash
# Sanity checks for one metric on a CPU node (cpuqs), for the checkers (D-019, D-023).
#
#   sbatch slurm/sanity_cpu.sbatch detect
#   args: <metric> [extra args for genprobe.sanity...]
#SBATCH --job-name=sanity-cpu
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
METRIC=${1:?metric}; shift
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
# The env each metric runs in (Metric.env; tests/test_checkers.py keeps the two in sync).
case "$METRIC" in
  ocr) DEFAULT_ENV=aux ;;
  *) DEFAULT_ENV=eval ;;
esac
activate_env "${GENPROBE_ENV:-$DEFAULT_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$GENPROBE_ROOT"
python -m genprobe.sanity --metric "$METRIC" --device cpu "$@"
