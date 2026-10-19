#!/bin/bash
# M2 full run: prompt set v1 (600 prompts x 4 seeds) with all four models = 9,600 images, then
# CLIPScore + VQAScore and summaries for the test and dev splits. ~8 GPU-hours on H100 NVL
# (estimate from the smoke run, docs/hpc_notes.md). Run on HPC3's login node:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/check_gen_env.sh && bash scripts/run_v1.sh"
#
# 4 generation shards per model (~600 images each), at most 2 per model at a time; 2 scoring
# shards per metric. Resubmitting is safe: finished images and scores are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
GEN_SHARDS=4 GEN_CONCURRENCY=2 GEN_TIME=02:00:00 SCORE_SHARDS=2 SCORE_TIME=02:00:00 SPLITS="test dev" \
  exec bash scripts/run_prompt_set.sh data/prompts/prompts_v1.jsonl
