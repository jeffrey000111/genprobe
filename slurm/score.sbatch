#!/bin/bash
# One shard of a scoring job array for a single metric.
#
#   sbatch --array=0-1 slurm/score.sbatch clipscore "data/manifests/gen_sdxl_*.jsonl" sdxl 2 --prompts data/prompts/smoke.jsonl
#   args: <metric> <samples glob (quoted)> <tag> <num_shards> [extra args for genprobe.score...]
#
# Scoring is resumable, so it can also run on an H100 when the A100s are busy:
#   sbatch --gres=gpu:h100:1 ...   (timings of scoring jobs are not compared, so mixing is fine)
#SBATCH --job-name=score
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
METRIC=${1:?metric}; SAMPLES=${2:?samples glob}; TAG=${3:?tag}; NUM_SHARDS=${4:?num shards}; shift 4
# Slurm runs a spooled copy of this script, so locate the repo via the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
# 7B VLM scorers need >= 40 GB, so the same A100/H100 rule applies.
export GENPROBE_GPU_REGEX="${SCORE_GPU_REGEX:-A100|H100}"
require_gpu
activate_env "${GENPROBE_ENV:-eval}"
cd "$GENPROBE_ROOT"
python -m genprobe.score --metric "$METRIC" --samples "$SAMPLES" --tag "$TAG" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" "$@"
