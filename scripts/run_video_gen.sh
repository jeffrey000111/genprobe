#!/bin/bash
# Wan2.2 videos and their evaluation (guide §11.2-11.3, D-036). Run on HPC3's login node (it only
# calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_video_gen.sh smoke"
#       3 prompts at 121 and at 81 frames (timing for D-036's frame rule), then a pipeline check:
#       VBench on 3 Wan and 3 Floor33 videos, VQAScore on the 3 Wan videos, and the timing table
#       (results/video_gen_smoke/timing.md).
#   ssh sjsu3 "cd ~/genprobe && FRAMES=121 bash scripts/run_video_gen.sh full"     after approval
#       160 prompts in 2 shards, then VQAScore on the Wan videos and VBench on all five models' videos.
#       The analysis (python -m genprobe.video_eval analyze) runs once the video meta-evaluation has
#       scored the older models' videos too.
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_video_gen.sh ltx-smoke"    LTX-Video 0.9.5, 49 frames (D-039):
#       3 prompts, VQAScore on video and VBench on them (a pipeline and timing check)
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_video_gen.sh ltx-full"     after approval: 160 prompts, VQAScore,
#       VBench, then the analysis with LTX-Video as a column (results/video/)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_video_gen.sh smoke|full}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
mkdir -p logs
if [[ "$NAME" == smoke ]]; then
  a=$(sbatch --parsable --export=ALL,LIMIT=3 --time=00:45:00 slurm/video_gen.sbatch 121 1); a=${a%%;*}
  b=$(sbatch --parsable --export=ALL,LIMIT=3 --time=00:45:00 slurm/video_gen.sbatch 81 1); b=${b%%;*}
  echo "submitted Wan smoke: 121 frames job $a, 81 frames job $b"
  v=$(sbatch --parsable --dependency="afterok:$a" --export=ALL,LIMIT=3 --time=00:45:00 slurm/vbench.sbatch wan22_f121 Floor33)
  echo "submitted VBench smoke: job ${v%%;*} (after $a)"
  q=$(SAMPLES="data/video/manifests/gen_wan22_f121_*.jsonl" METRICS=vqascore_video CPU_METRICS="" SUMMARY=0 \
      TAG=wan_smoke DEPENDENCY="afterok:$a" bash scripts/submit_scoring.sh data/prompts/genai_bench_video.jsonl)
  echo "$q"
  t=$(sbatch --parsable --dependency="afterany:$a:$b" -p cpuqs -c 1 --mem=4G -t 00:10:00 -J video-timing \
      -o logs/%x_%j.out --wrap "source scripts/paths.sh && source \$GENPROBE_ENVS/eval/bin/activate && python -m genprobe.video_gen timing")
  echo "submitted timing table: job ${t%%;*} -> results/video_gen_smoke/timing.md"
elif [[ "$NAME" == full ]]; then
  FRAMES=${FRAMES:?set FRAMES (121, 81 or 49) as D-036 decides from the smoke run}
  g=$(sbatch --parsable --array=0-1 --time=03:30:00 slurm/video_gen.sbatch "$FRAMES" 2); g=${g%%;*}
  echo "submitted Wan generation ($FRAMES frames, 2 shards): job $g"
  SAMPLES="data/video/manifests/gen_wan22_f${FRAMES}_*.jsonl" METRICS=vqascore_video CPU_METRICS="" SUMMARY=0 \
    TAG="wan_f$FRAMES" DEPENDENCY="afterok:$g" SCORE_TIME=01:00:00 bash scripts/submit_scoring.sh data/prompts/genai_bench_video.jsonl
  if [[ "${NO_VBENCH:-0}" == 1 ]]; then
    echo "VBench not submitted (NO_VBENCH=1); later: sbatch --time=03:00:00 slurm/vbench.sbatch wan22_f$FRAMES Floor33 Gen2 Pika_v1 Modelscope"
  else
    v=$(sbatch --parsable --dependency="afterok:$g" --time=03:00:00 slurm/vbench.sbatch "wan22_f$FRAMES" Floor33 Gen2 Pika_v1 Modelscope)
    echo "submitted VBench (5 models): job ${v%%;*}"
  fi
elif [[ "$NAME" == ltx-smoke || "$NAME" == ltx-full ]]; then
  M=ltxv095_f49
  if [[ "$NAME" == ltx-smoke ]]; then LIM=3; TAG=ltx_smoke; T=00:45:00; else LIM=; TAG=ltx_f49; T=02:00:00; fi
  g=$(sbatch --parsable --export=ALL,MODEL=ltxv095${LIM:+,LIMIT=$LIM} --time="$T" slurm/video_gen.sbatch 49 1); g=${g%%;*}
  echo "submitted LTX-Video generation ($NAME): job $g"
  IDS=$(mktemp)
  SAMPLES="data/video/manifests/gen_${M}_*.jsonl" METRICS=vqascore_video CPU_METRICS="" SUMMARY=0 IDS_FILE="$IDS" \
    TAG="$TAG" DEPENDENCY="afterok:$g" SCORE_TIME=00:45:00 bash scripts/submit_scoring.sh data/prompts/genai_bench_video.jsonl
  q=$(tail -1 "$IDS"); rm -f "$IDS"
  v=$(sbatch --parsable --dependency="afterok:$g" --export=ALL${LIM:+,LIMIT=$LIM} --time="$T" slurm/vbench.sbatch "$M"); v=${v%%;*}
  echo "submitted VBench ($M): job $v"
  if [[ "$NAME" == ltx-full ]]; then
    a=$(sbatch --parsable --dependency="afterok:$q:$v" -p cpuqs -c 2 --mem=8G -t 00:20:00 -J video-eval \
        -o logs/%x_%j.out --wrap "source scripts/paths.sh && source \$GENPROBE_ENVS/eval/bin/activate && python -m genprobe.video_eval analyze --wan wan22_f49 --extra $M")
    echo "submitted analysis: job ${a%%;*} -> results/video/"
  fi
else
  echo "usage: run_video_gen.sh smoke|full|ltx-smoke|ltx-full" >&2; exit 2
fi
