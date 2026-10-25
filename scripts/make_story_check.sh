#!/bin/bash
# Blind human check of the multi-shot videos (genprobe/story_check.py, docs/story_check_protocol.md,
# D-038), on a CPU node, after the test stage of scripts/run_story.sh. On HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_story_check.sh"           # package
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_story_check.sh analyze"   # after labelling
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-make}
case "$MODE" in
  make) CMD="python -m genprobe.story_check make" ;;
  analyze) CMD="python -m genprobe.story_check analyze --labels data/labels/story_check_labels.csv" ;;
  *) echo "usage: $0 [make|analyze]" >&2; exit 2 ;;
esac
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
# the video env has ffmpeg (imageio-ffmpeg) for joining the shots
srun -p cpuqs -n 1 -c 4 --mem=8G -t 00:30:00 -J story-check --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/video/bin/activate\"
  $CMD"
