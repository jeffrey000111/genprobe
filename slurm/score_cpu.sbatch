#!/bin/bash
# One shard of a scoring job array on a CPU node (cpuqs), for the small checker models
# (ocr: EasyOCR, detect: Grounding DINO tiny). Their GPU queue wait would be far longer than
# their CPU run time (D-019). Same arguments as slurm/score.sbatch:
#
#   sbatch --array=0 slurm/score_cpu.sbatch ocr "data/manifests/gen_*.jsonl" smoke 1 --prompts data/prompts/smoke.jsonl
#   args: <metric> <samples glob (quoted)> <tag> <num_shards> [extra args for genprobe.score...]
#
# Resubmitting the same command is safe: samples already scored with the same metric_version
# are skipped.
#SBATCH --job-name=score-cpu
#SBATCH --partition=cpuqs
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
METRIC=${1:?metric}; SAMPLES=${2:?samples glob}; TAG=${3:?tag}; NUM_SHARDS=${4:?num shards}; shift 4
# Slurm runs a spooled copy of this script, so locate the repo via the submit dir.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
# The env each metric runs in (Metric.env; tests/test_checkers.py keeps the two in sync).
case "$METRIC" in
  ocr) DEFAULT_ENV=aux ;;
  *) DEFAULT_ENV=eval ;;
esac
activate_env "${GENPROBE_ENV:-$DEFAULT_ENV}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK:-?}  env: ${GENPROBE_ENV:-$DEFAULT_ENV}"
cd "$GENPROBE_ROOT"
python -m genprobe.score --metric "$METRIC" --samples "$SAMPLES" --tag "$TAG" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" --device cpu "$@"
