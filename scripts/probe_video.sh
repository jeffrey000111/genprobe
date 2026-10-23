#!/bin/bash
# Read-only checks before module D's video part (guide §11.2-11.3), on HPC3's login node: free
# space, how t2v_metrics (installed in the eval env) loads GenAI-Bench's video ratings and scores
# video with Qwen2-VL, and whether the gen env's diffusers has the Wan pipeline. Changes nothing.
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/probe_video.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/paths.sh
T2V=$(ls -d "$GENPROBE_ENVS"/eval/lib/python3*/site-packages/t2v_metrics 2>/dev/null | head -1)
echo "== space"
df -h /home | tail -1
du -sh "$HF_HOME" "$GENPROBE_ENVS" data 2>/dev/null
quota -s 2>/dev/null | tail -3
echo "== t2v_metrics: GenAI-Bench video data ($T2V)"
grep -rn "GenAI-Bench-800\|genai_video\|Floor33\|GenAIBench_Video\|class GenAIBench" "$T2V" 2>/dev/null | head -20
f=$(grep -rln "GenAIBench_Video\|genai_video" "$T2V" 2>/dev/null | head -1)
[[ -n "$f" ]] && grep -n "url\|http\|hf_hub\|json\|zip\|models\|rating" "$f" | head -60
echo "== t2v_metrics: Qwen2-VL video scoring"
q=$(grep -rl "class Qwen2VLModel" "$T2V" 2>/dev/null | head -1)
echo "$q"
[[ -n "$q" ]] && grep -n "video\|fps\|nframes\|Does this\|question_template\|max_pixels" "$q" | head -60
echo "== gen env: diffusers Wan pipeline"
"$GENPROBE_ENVS/gen/bin/python" -c "import diffusers; from diffusers import WanPipeline, AutoencoderKLWan; print('diffusers', diffusers.__version__, 'WanPipeline ok')" 2>&1 | tail -2
"$GENPROBE_ENVS/gen/bin/python" -m pip show imageio imageio-ffmpeg av 2>/dev/null | grep -E "^(Name|Version)"
echo "== eval env: video readers"
"$GENPROBE_ENVS/eval/bin/python" -m pip show decord av qwen-vl-utils 2>/dev/null | grep -E "^(Name|Version)"
