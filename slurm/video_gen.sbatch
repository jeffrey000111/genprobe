#!/bin/bash
# Wan2.2 TI2V-5B text-to-video (genprobe/video_gen.py, D-036), one shard of a job array, on an
# H100 NVL in the `video` env. Submitted by scripts/run_video_gen.sh.
#
#   cd ~/genprobe && LIMIT=3 sbatch slurm/video_gen.sbatch 121 1      (smoke: 3 prompts at 121 frames)
#   cd ~/genprobe && sbatch --array=0-1 slurm/video_gen.sbatch 121 2   (160 prompts, 2 shards)
#
# Output: data/video/wan22_f<frames>/, data/video/manifests/, logs/video-gen_<id>_<shard>.out.
# Resubmitting is safe: finished videos are skipped.
#SBATCH --job-name=video-gen
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
FRAMES=${1:?frames (4k+1)}; NUM_SHARDS=${2:-1}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="H100 NVL"
require_gpu
activate_env video
cd "$GENPROBE_ROOT"
python -m genprobe.video_gen select
python -m genprobe.video_gen generate --frames "$FRAMES" ${LIMIT:+--limit "$LIMIT"} \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS"
