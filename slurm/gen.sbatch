#!/bin/bash
# One shard of a generation job array. Shard index = array task id.
#
#   sbatch --array=0-3%2 slurm/gen.sbatch configs/gen/sdxl.yaml data/prompts/smoke.jsonl 4 --seeds 0
#   args: <config> <prompts.jsonl> <num_shards> [extra args for genprobe.generate...]
#
# Resubmitting the same command is always safe: finished samples are skipped.
#SBATCH --job-name=gen
#SBATCH --partition=nsfqs              # HPC3 NSF cluster, 13 x H100 NVL 94 GB (max 2 days)
# H100 NVL only (D-009): all four generators fit without CPU offload (FLUX needs ~50 GB, the
# A100s have 40 GB), and one card type keeps per-image timings comparable across models.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00                # shorter jobs schedule sooner; resume covers the rest
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL
##SBATCH --mail-user=YOUR_ID@sjsu.edu  # remove one '#' and fill in to get failure emails

set -euo pipefail
CONFIG=${1:?config yaml}; PROMPTS=${2:?prompts jsonl}; NUM_SHARDS=${3:?num shards}; shift 3
# Slurm runs a spooled copy of this script, so locate the repo via the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="${GEN_GPU_REGEX:-H100 NVL}"   # one card type for comparable timings
require_gpu
activate_env gen
cd "$GENPROBE_ROOT"
python -m genprobe.generate --config "$CONFIG" --prompts "$PROMPTS" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" "$@"
