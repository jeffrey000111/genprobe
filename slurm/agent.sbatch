#!/bin/bash
# One run of the generation-debugging agent (genprobe/agent/, D-031) on an H100, all rounds in one
# job so no round waits in the queue: plan (eval env) -> generate (gen env) -> check (VQAScore on
# the GPU, detector and OCR on CPU, as in prompt set v1) -> ingest, until no prompt has a next
# step; then the report. Resumable: a finished round is skipped, a half-done one picks up where
# it stopped.
#
#   cd ~/genprobe && sbatch slurm/agent.sbatch devrule1 dev rule
#   cd ~/genprobe && sbatch --time=03:00:00 slurm/agent.sbatch testrule test rule
#
# Output: data/agent/<run>/ (state, audit log, images, final.jsonl), results/agent/<run>/,
# logs/agent_<jobid>.out
#SBATCH --job-name=agent
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
RUN=${1:?run name, e.g. devrule1}; SPLIT=${2:?split: dev or test}; POLICY=${3:-rule}
MAX_ROUNDS=${MAX_ROUNDS:-8}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX=H100
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
D="data/agent/$RUN"
[[ -f "$D/state.json" ]] || in_env eval python -m genprobe.agent.run init --run "$RUN" --split "$SPLIT" --policy "$POLICY"
for k in $(seq 1 "$MAX_ROUNDS"); do
  [[ -f "$D/round$k/ingested" ]] && continue
  [[ -f "$D/round$k/requests.jsonl" ]] || in_env eval python -m genprobe.agent.run plan --run "$RUN" --round "$k"
  n=$(grep -c . "$D/round$k/requests.jsonl" || true)
  echo "== round $k: $n attempts"
  if [[ "$n" -eq 0 ]]; then touch "$D/round$k/ingested"; break; fi
  in_env gen python -m genprobe.agent.gen --run "$RUN" --round "$k" || echo "(some generations failed; ingest marks them)"
  S=(--samples "$D/manifests/gen_r${k}_*.jsonl" --prompts "$D/round$k/score_prompts.jsonl" --tag "agent_${RUN}_r$k")
  in_env eval python -m genprobe.score --metric vqascore "${S[@]}" --device cuda
  in_env eval python -m genprobe.score --metric detect "${S[@]}" --device cpu
  in_env aux python -m genprobe.score --metric ocr "${S[@]}" --device cpu
  in_env eval python -m genprobe.agent.run ingest --run "$RUN" --round "$k"
  touch "$D/round$k/ingested"
done
in_env eval python -m genprobe.agent.run report --run "$RUN"
