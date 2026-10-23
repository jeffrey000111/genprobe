#!/bin/bash
# GenAI-Bench-Video for the video meta-evaluation (genprobe/genai_bench_video.py, D-035), on a CPU
# node: prompt file, video manifest, then 9 frames per video (PyAV, eval env). On HPC3, after
# scripts/fetch_genai_bench_video.sh ran on coe-hpc1:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/prepare_genai_bench_video.sh"
#
# Output: data/prompts/genai_bench_video{,_smoke}.jsonl, data/manifests/ext_genai_video{,_frames,_mid}.jsonl,
# data/external/genai_bench_video/frames/
set -euo pipefail
cd "$(dirname "$0")/.."
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
# 8 worker processes; videos whose frames are already there are skipped, so a cut-off run resumes.
srun -p cpuqs -n 1 -c 8 --mem=32G -t 02:00:00 -J video-frames --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.genai_bench_video prepare && python -m genprobe.genai_bench_video frames --workers 8"
