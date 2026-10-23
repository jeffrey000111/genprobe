#!/bin/bash
# VBench's six custom-input dimensions (genprobe/video_eval.py, D-036) for one or more models' videos
# of data/prompts/wan_video.jsonl, on an H100 in the `vbench` env, weights from $HF_HOME/vbench
# (scripts/download_vbench.sh). Submitted by scripts/run_video_gen.sh.
#
#   cd ~/genprobe && LIMIT=3 sbatch slurm/vbench.sbatch wan22_f121 Floor33     (smoke)
#   cd ~/genprobe && sbatch slurm/vbench.sbatch wan22_f121 Floor33 Gen2 Pika_v1 Modelscope
#
# Output: data/video/vbench/<model>.jsonl (+ raw/), logs/vbench_<id>.out
#SBATCH --job-name=vbench
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: sbatch slurm/vbench.sbatch <model> [<model> ...]" >&2; exit 2; }
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
activate_env vbench
cd "$GENPROBE_ROOT"
export VBENCH_CACHE_DIR="$HF_HOME/vbench"
# DINO's torch.hub entry point loads its weights through torch.hub's checkpoint cache, not the path
# VBench passes; point that cache at the downloaded file so nothing is fetched on an offline node.
export TORCH_HOME="$VBENCH_CACHE_DIR/torch_home"
mkdir -p "$TORCH_HOME/hub/checkpoints"
ln -sf "$VBENCH_CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth" "$TORCH_HOME/hub/checkpoints/dino_vitbase16_pretrain.pth"
for m in "$@"; do
  python -m genprobe.video_eval vbench --model "$m" ${LIMIT:+--limit "$LIMIT"}
done
