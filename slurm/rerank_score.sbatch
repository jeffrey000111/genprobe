#!/bin/bash
# Re-ranking curve (genprobe/rerank.py, D-034), step 2 of 2, on an H100 after the generation shards:
# VQAScore for the new images (eval env) -> picks of best-of-N (N = 1, 2, 4, 9) -> the independent
# judge grades every picked image (judge env, Gemma 4 31B, frozen rubric, pass = 5) -> analysis and
# figure. Submitted by scripts/run_rerank.sh.
#
#   cd ~/genprobe && sbatch --time=03:00:00 slurm/rerank_score.sbatch full
#   cd ~/genprobe && PER_CATEGORY=5 sbatch slurm/rerank_score.sbatch smoke
#
# Output: data/rerank/scores/, data/rerank/picks_<name>.jsonl, data/judge/grade_rerank_<name>_gemma4_31b.jsonl,
# results/rerank/ (full) or results/rerank_smoke/ (smoke)
#SBATCH --job-name=rerank-score
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
NAME=${1:?name: smoke or full}
JUDGE=${JUDGE:-gemma4_31b}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
PC=(); [[ -z "${PER_CATEGORY:-}" ]] || PC=(--per-category "$PER_CATEGORY")
OUTDIR=results/rerank; [[ "$NAME" == full ]] || OUTDIR="results/rerank_$NAME"
TAG="grade_rerank_${NAME}_${JUDGE}"
PICKS="data/rerank/picks_$NAME.jsonl"
in_env eval python -m genprobe.score --metric vqascore --samples "data/rerank/manifests/gen_*.jsonl" --tag rerank \
  --prompts data/prompts/prompts_v1.jsonl --out data/rerank
in_env eval python -m genprobe.rerank picks --out "$PICKS" "${PC[@]}"
[[ -f "data/judge/$TAG.jsonl" ]] || in_env judge python -m genprobe.judge grade --images "$PICKS" --tag "$TAG" --judge "$JUDGE"
in_env eval python -m genprobe.rerank analyze --picks "$PICKS" --grades "$TAG" --out "$OUTDIR"
