#!/bin/bash
# One command for the M1 smoke run on HPC3: the 20 smoke prompts with all four models (seed 0),
# CLIPScore + VQAScore, then results/smoke/summary.md. Run on HPC3's login node, after
# scripts/setup_envs.sh and scripts/download_models.sh have been run once on coe-hpc1:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_smoke.sh"
#   scripts/hpc_sync.sh pull && cat results/smoke/summary.md        (on the laptop, later)
set -euo pipefail
cd "$(dirname "$0")/.."
GEN_SHARDS=1 GEN_CONCURRENCY=1 GEN_TIME=01:00:00 SEEDS=0 SCORE_SHARDS=1 SCORE_TIME=00:45:00 SPLITS=all \
  exec bash scripts/run_prompt_set.sh data/prompts/smoke.jsonl
