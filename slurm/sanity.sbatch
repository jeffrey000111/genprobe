#!/bin/bash
# Sanity checks for one metric on an H100 (genprobe/sanity.py, guide §7.9, D-023): mismatched
# prompts, one-word counterfactuals, repeatability, on dev-split images.
#
#   sbatch slurm/sanity.sbatch vqascore
#   args: <metric> [extra args for genprobe.sanity...]
#
# Output: results/sanity/<metric>.json and results/sanity/summary.md.
#SBATCH --job-name=sanity
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
METRIC=${1:?metric}; shift
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="${SCORE_GPU_REGEX:-H100}"
require_gpu
# The env each metric runs in (Metric.env; tests/test_checkers.py keeps the two in sync).
case "$METRIC" in
  ocr) DEFAULT_ENV=aux ;;
  *) DEFAULT_ENV=eval ;;
esac
activate_env "${GENPROBE_ENV:-$DEFAULT_ENV}"
cd "$GENPROBE_ROOT"
python -m genprobe.sanity --metric "$METRIC" --device cuda "$@"
