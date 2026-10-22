#!/bin/bash
# Final grading of an agent run by the independent judge (D-030, D-032) on an H100: the run's
# report (groups and their images) -> genprobe.judge grade (judge env) -> genprobe.agent.compare.
#
#   cd ~/genprobe && sbatch slurm/grade.sbatch devrule1
#
# Output: data/judge/grade_<run>_<judge>.jsonl, results/agent/<run>/judged.md, logs/grade_<jobid>.out
#SBATCH --job-name=grade
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
RUN=${1:?agent run name}; JUDGE=${2:-gemma4_31b}
TAG="grade_${RUN}_${JUDGE}"
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
in_env eval python -m genprobe.agent.run report --run "$RUN"
in_env judge python -m genprobe.judge grade --images "data/agent/$RUN/final.jsonl" --tag "$TAG" --judge "$JUDGE"
in_env judge python -m genprobe.agent.compare --run "$RUN" --grades "$TAG"
