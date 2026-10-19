#!/bin/bash
# Pipeline check of a scoring run, on a CPU node (genprobe/score_check.py): per metric, samples
# expected and scored, errors, metric_version, value range and seconds per image. Nothing is
# compared with human ratings. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && SAMPLES=data/manifests/ext_genai_bench.jsonl PROJECT=9600 bash scripts/check_scores.sh data/prompts/genai_bench_smoke.jsonl clipscore vqascore pickscore hpsv2"
#
# Knobs (env): SAMPLES (data/manifests/gen_*.jsonl), TAG (the prompts file's name), PROJECT
# (number of images to project the scoring time to).
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:?usage: check_scores.sh <prompts.jsonl> <metric>...}; shift
SAMPLES=${SAMPLES:-data/manifests/gen_*.jsonl}
TAG=${TAG:-$(basename "$PROMPTS" .jsonl)}
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 1 --mem=4G -t 00:10:00 -J score-check --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.score_check --samples '$SAMPLES' --prompts '$PROMPTS' --tag '$TAG' ${PROJECT:+--project $PROJECT} $*"
