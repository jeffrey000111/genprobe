#!/bin/bash
# Scorer throughput by batch size on one H100 (genprobe.service.bench, D-040): what one GPU worker
# of the scoring service can absorb. About 10 minutes; no service or database is involved.
#
#   sbatch slurm/service_bench.sbatch vqascore 1,2,4,8,16
#   sbatch slurm/service_bench.sbatch clipscore 1,4,16,32
#SBATCH --job-name=service_bench
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
METRIC=${1:?metric}; BATCH_SIZES=${2:-1,2,4,8,16}; IMAGES=${3:-128}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="${SCORE_GPU_REGEX:-H100}"
require_gpu
activate_env "${GENPROBE_ENV:-eval}"
cd "$GENPROBE_ROOT"
python -m genprobe.service.bench --metric "$METRIC" --device cuda --batch-sizes "$BATCH_SIZES" \
  --images "$IMAGES" --out "results/service/bench_${METRIC}_h100.jsonl"
