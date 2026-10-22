#!/bin/bash
# Probe of the judge env: vLLM loads Qwen3-VL-8B-Instruct offline on an H100 and answers a few
# test questions (one image, two images, JSON-constrained output, logprobs, throughput);
# scripts/check_judge_env.py. On HPC3, after setup_envs.sh judge + download_models.sh judge:
#
#   cd ~/genprobe && sbatch slurm/judge_probe.sbatch                       Qwen3-VL-8B
#   cd ~/genprobe && sbatch slurm/judge_probe.sbatch --judge gemma4_31b    Gemma 4 31B (D-030)
#
# Output: logs/judge-probe_<jobid>.out
#SBATCH --job-name=judge-probe
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
activate_env judge
cd "$GENPROBE_ROOT"
echo "C compiler (triton builds small helpers at run time): $(command -v gcc || command -v cc || echo none)"
python scripts/check_judge_env.py "$@"
