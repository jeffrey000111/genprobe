#!/bin/bash
# Re-ranking curve (genprobe/rerank.py, D-034), step 1 of 2: seeds 4-8 for the test prompts, on an
# H100, one shard of a job array. SDXL for every prompt; the routed model (SD3.5 Medium for counting
# and spatial, FLUX.1 schnell for text and GenAI-Bench) for its categories. Images and manifests go
# to data/rerank/, so prompt set v1 (M2) is untouched. Submitted by scripts/run_rerank.sh.
#
#   cd ~/genprobe && sbatch --array=0-1 slurm/rerank_gen.sbatch 2            (full: 480 prompts)
#   cd ~/genprobe && PER_CATEGORY=5 sbatch --array=0 slurm/rerank_gen.sbatch 1   (smoke: 20 prompts)
#
# Resubmitting is safe: finished images are skipped.
#SBATCH --job-name=rerank-gen
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
NUM_SHARDS=${1:?num shards}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="${GEN_GPU_REGEX:-H100 NVL}"   # the card type of prompt set v1's timings
require_gpu
activate_env gen
cd "$GENPROBE_ROOT"
python -m genprobe.rerank prompts ${PER_CATEGORY:+--per-category "$PER_CATEGORY"}
for m in sdxl sd35m flux_schnell; do
  [[ -s "data/rerank/prompts_$m.jsonl" ]] || continue
  python -m genprobe.generate --config "configs/gen/$m.yaml" --prompts "data/rerank/prompts_$m.jsonl" \
    --split test --seeds 4 5 6 7 8 --out data/rerank \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS"
done
