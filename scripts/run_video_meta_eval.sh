#!/bin/bash
# Video meta-evaluation (guide §11.3, genprobe/video_meta_eval.py, D-035): score GenAI-Bench-Video
# with every metric, then (full run) the analysis on a CPU node once all scoring jobs have ended.
# Run on HPC3's login node (it only calls sbatch), after scripts/prepare_genai_bench_video.sh:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_video_meta_eval.sh smoke"   # 20 prompts, 80 videos: pipeline only
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_video_meta_eval.sh full"    # 800 prompts, after approval
#
# Smoke: scoring of the 20 smoke prompts, then a pipeline check (counts, errors, ranges, seconds);
# nothing is compared with human ratings. Scores are reused by the full run (resume).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_video_meta_eval.sh smoke|full}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
case "$NAME" in
  smoke) P=data/prompts/genai_bench_video_smoke.jsonl; SH=1; TV=00:40:00; TF=00:40:00 ;;
  full) P=data/prompts/genai_bench_video.jsonl; SH=2; TV=02:00:00; TF=01:30:00 ;;
  *) echo "usage: run_video_meta_eval.sh smoke|full" >&2; exit 2 ;;
esac
IDS=$(mktemp)
common=(CPU_METRICS="" SUMMARY=0 IDS_FILE="$IDS")
env "${common[@]}" SAMPLES=data/manifests/ext_genai_video.jsonl METRICS=vqascore_video SCORE_SHARDS=$SH SCORE_TIME=$TV \
  TAG="video_$NAME" bash scripts/submit_scoring.sh "$P"
env "${common[@]}" SAMPLES=data/manifests/ext_genai_video_mid.jsonl METRICS=vqascore SCORE_SHARDS=$SH SCORE_TIME=$TF \
  TAG="video_mid_$NAME" bash scripts/submit_scoring.sh "$P"
env "${common[@]}" SAMPLES=data/manifests/ext_genai_video_frames.jsonl METRICS="clipscore pickscore hpsv2" SCORE_SHARDS=1 \
  SCORE_TIME=$TF TAG="video_frames_$NAME" bash scripts/submit_scoring.sh "$P"
dep=$(paste -sd: "$IDS"); rm -f "$IDS"
if [[ "$NAME" == full ]]; then
  id=$(sbatch --parsable --dependency="afterany:$dep" slurm/video_meta_eval.sbatch)
  echo "submitted analysis: job ${id%%;*} (cpuqs, starts after $dep) -> results/video_meta_eval/"
else
  echo "when squeue is empty, check the pipeline (no human ratings involved):"
  echo "  SAMPLES=data/manifests/ext_genai_video.jsonl TAG=video_smoke PROJECT=3200 bash scripts/check_scores.sh $P vqascore_video"
  echo "  SAMPLES=data/manifests/ext_genai_video_mid.jsonl TAG=video_mid_smoke PROJECT=3200 bash scripts/check_scores.sh $P vqascore"
  echo "  SAMPLES=data/manifests/ext_genai_video_frames.jsonl TAG=video_frames_smoke PROJECT=28800 bash scripts/check_scores.sh $P clipscore pickscore hpsv2"
fi
