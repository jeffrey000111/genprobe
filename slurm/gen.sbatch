#!/bin/bash
# One shard of a generation job array. Shard index = array task id.
#
#   sbatch --array=0-3%2 slurm/gen.sbatch configs/gen/sdxl.yaml data/prompts/smoke.jsonl 4 --seeds 0
#   args: <config> <prompts.jsonl> <num_shards> [extra args for genprobe.generate...]
#
# Resubmitting the same command is always safe: finished samples are skipped.
#SBATCH --job-name=gen
#SBATCH --partition=gpuqs              # HPC3 short GPU queue (max 2 days)
# A100 only: cs001-cs004 have 4 each, the most capacity on HPC3, and one card type keeps
# per-image timings comparable across models (D-009). For an H100 instead, override at
# submit time: sbatch --gres=gpu:h100:1 ...
#SBATCH --gres=gpu:a100:1
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
require_gpu
activate_env gen
cd "$GENPROBE_ROOT"
python -m genprobe.generate --config "$CONFIG" --prompts "$PROMPTS" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" "$@"
