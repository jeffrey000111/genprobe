#!/bin/bash
# Video meta-evaluation analysis (genprobe/video_meta_eval.py, D-035) on a CPU node, after every
# video has every score; it refuses to compute on partial scores. Submitted by
# scripts/run_video_meta_eval.sh full.
#SBATCH --job-name=video-meta-eval
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
cd "$GENPROBE_ROOT"
activate_env eval
python -m genprobe.video_meta_eval
