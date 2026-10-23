#!/bin/bash
# Re-ranking curve (guide §11.1, genprobe/rerank.py, D-034): generation shards, then one job that
# scores, picks, grades and analyses. Run on HPC3's login node (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_rerank.sh smoke"   # 20 test prompts (5 per category)
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_rerank.sh full"    # all 480 test prompts, after approval
#
# The smoke images are part of the full run (resume skips them); the smoke grades and results are
# kept apart (data/judge/grade_rerank_smoke_*, results/rerank_smoke/).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_rerank.sh smoke|full}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
mkdir -p logs
case "$NAME" in
  smoke) export PER_CATEGORY=5; SHARDS=1; GEN_T=00:45:00; SCORE_T=00:50:00 ;;
  full) unset PER_CATEGORY; SHARDS=2; GEN_T=02:30:00; SCORE_T=03:00:00 ;;
  *) echo "usage: run_rerank.sh smoke|full" >&2; exit 2 ;;
esac
gid=$(sbatch --parsable --array=0-$((SHARDS - 1)) --time="$GEN_T" slurm/rerank_gen.sbatch "$SHARDS")
gid=${gid%%;*}
echo "submitted generation ($SHARDS shard(s)): job $gid"
sid=$(sbatch --parsable --dependency="afterok:$gid" --time="$SCORE_T" slurm/rerank_score.sbatch "$NAME")
echo "submitted scoring + judge + analysis: job ${sid%%;*} (starts after $gid)"
