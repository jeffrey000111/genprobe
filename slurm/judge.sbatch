#!/bin/bash
# One round of the VLM judge (genprobe/judge.py) and its analysis (genprobe/judge_eval.py) on an
# H100, in the judge env (vLLM). Rounds: dev (rubric iteration, 80 prompts) or test (once, with
# the frozen rubric; pre-registered H3).
#
#   cd ~/genprobe && sbatch slurm/judge.sbatch dev dev_r1
#   cd ~/genprobe && sbatch --time=01:00:00 slurm/judge.sbatch test test_r1
#
# Output: data/judge/<tag>.jsonl (+ .meta.json), results/judge/<tag>/, logs/judge_<jobid>.out
#SBATCH --job-name=judge
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROUND=${1:?round: dev or test}; TAG=${2:?tag, e.g. dev_r1}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
activate_env judge
cd "$GENPROBE_ROOT"
python -m genprobe.judge run --round "$ROUND" --tag "$TAG"
python -m genprobe.judge_eval --tag "$TAG" --round "$ROUND"
