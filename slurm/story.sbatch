#!/bin/bash
# Multi-shot videos from a brief (genprobe/story.py, D-038), one stage per job on an H100; submitted in
# order by scripts/run_story.sh. NAME = smoke (1 dev brief, all arms), dev (8 briefs, arm B: sets
# tau_v) or test (24 briefs, all arms).
#
#   STAGE kf:    shot plan (judge env) -> FLUX keyframes, seed 0 (gen) -> VQAScore (eval); with the agent
#                (smoke, test) seeds 1 and 2 where the earlier ones missed 0.9, each scored
#   STAGE vid0:  the first round of videos (video env), one shard of a job array -> VQAScore on video (eval)
#   STAGE vid12: the agent's second and third attempts, each round scored before the next
#   STAGE judge: Gemma 4 31B on frames (judge env) -> tune (dev) or analyze (smoke, test)
#
#   cd ~/genprobe && sbatch slurm/story.sbatch kf smoke
#SBATCH --job-name=story
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
STAGE=${1:?stage: kf, vid0, vid12 or judge}; NAME=${2:?name: smoke, dev or test}; NS=${3:-1}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="H100 NVL"
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
SHOTS=data/story/shots.jsonl
AGENT=1; [[ "$NAME" == dev ]] && AGENT=0
score() {   # metric, manifest file, tag, out
  if [[ -f "$2" ]]; then
    in_env eval python -m genprobe.score --metric "$1" --samples "$2" --tag "$3" --prompts "$SHOTS" --out "$4"
  else
    echo "no $2: nothing to score"
  fi
}

case "$STAGE" in
  kf)
    in_env judge python -m genprobe.story plan --name "$NAME"
    rounds=(0); [[ $AGENT == 1 ]] && rounds=(0 1 2)
    for r in "${rounds[@]}"; do
      in_env gen python -m genprobe.story keyframes --name "$NAME" --round "$r"
      score vqascore data/story/kf/manifests/gen_fluxkf_shard000of001.jsonl "story_kf_${NAME}_r$r" data/story/kf
    done
    ;;
  vid0)
    i=${SLURM_ARRAY_TASK_ID:-0}
    in_env video python -m genprobe.story videos --name "$NAME" --round 0 --shard "$i" --num-shards "$NS"
    score vqascore_video "data/story/video/manifests/gen_${NAME}_r0_shard$(printf %03d "$i")of$(printf %03d "$NS").jsonl" \
      "story_vid_${NAME}_r0_sh$i" data/story/video
    ;;
  vid12)
    for r in 1 2; do
      in_env video python -m genprobe.story videos --name "$NAME" --round "$r"
      score vqascore_video "data/story/video/manifests/gen_${NAME}_r${r}_shard000of001.jsonl" "story_vid_${NAME}_r$r" data/story/video
    done
    ;;
  judge)
    in_env judge python -m genprobe.story judge --name "$NAME"
    if [[ "$NAME" == dev ]]; then
      in_env eval python -m genprobe.story tune
    else
      in_env eval python -m genprobe.story analyze --name "$NAME"
    fi
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
