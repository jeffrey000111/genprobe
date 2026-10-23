#!/bin/bash
# Final grading of one or more agent runs on the same split by the independent judge (D-030,
# D-032), on an H100: each run's report -> genprobe.judge grade (the first run's images in full, the
# others' agent images only, since the baselines are the same images) -> one genprobe.agent.compare
# over all runs.
#
#   cd ~/genprobe && sbatch slurm/grade.sbatch devrule2 devsdxl2
#   JUDGE=gemma4_31b NAME=dev_r2 sbatch slurm/grade.sbatch devrule2 devsdxl2
#
# Output: data/judge/grade_<run>_<judge>.jsonl, results/agent/<name>/judged.md, logs/grade_<jobid>.out
#SBATCH --job-name=grade
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: sbatch slurm/grade.sbatch <run> [<run> ...]" >&2; exit 2; }
JUDGE=${JUDGE:-gemma4_31b}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
CMP=()
first=1
for RUN in "$@"; do
  TAG="grade_${RUN}_${JUDGE}"
  in_env eval python -m genprobe.agent.run report --run "$RUN"
  if [[ ! -f "data/judge/$TAG.jsonl" ]]; then
    EXTRA=(); [[ $first == 1 ]] || EXTRA=(--only-group-prefix agent)
    in_env judge python -m genprobe.judge grade --images "data/agent/$RUN/final.jsonl" --tag "$TAG" --judge "$JUDGE" "${EXTRA[@]}"
  fi
  CMP+=(--run "$RUN" --grades "$TAG")
  first=0
done
in_env judge python -m genprobe.agent.compare "${CMP[@]}" ${NAME:+--name "$NAME"}
