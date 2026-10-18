#!/bin/bash
# One shard of a scoring job array for a single metric.
#
#   sbatch --array=0 slurm/score.sbatch clipscore "data/manifests/gen_*.jsonl" smoke 1 --prompts data/prompts/smoke.jsonl
#   args: <metric> <samples glob (quoted)> <tag> <num_shards> [extra args for genprobe.score...]
#
# Scoring runs on the nsfqs H100 NVL nodes like generation (D-009): the gpuqs A100 nodes
# answer "CUDA-capable device(s) is/are busy or unavailable" (probe job 91529).
# Resubmitting the same command is safe: samples already scored with the same
# metric_version are skipped.
#SBATCH --job-name=score
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
METRIC=${1:?metric}; SAMPLES=${2:?samples glob}; TAG=${3:?tag}; NUM_SHARDS=${4:?num shards}; shift 4
# Slurm runs a spooled copy of this script, so locate the repo via the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
# 7B VLM scorers need bf16 and ~20 GB; P100s (16 GB, no bf16) are rejected.
export GENPROBE_GPU_REGEX="${SCORE_GPU_REGEX:-H100}"
require_gpu
activate_env "${GENPROBE_ENV:-eval}"
cd "$GENPROBE_ROOT"
python -m genprobe.score --metric "$METRIC" --samples "$SAMPLES" --tag "$TAG" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" "$@"
