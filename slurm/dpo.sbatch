#!/bin/bash
# Diffusion-DPO on SD 1.5 (genprobe/dpo.py, D-037), one stage per job on an H100; submitted in order
# by scripts/run_dpo.sh. NAME = smoke (20 training prompts, 20 steps, 2 test prompts per category)
# or full (1,200 training prompts, 2,000 steps, all 480 test prompts).
#
#   STAGE cand:  training prompts -> SD 1.5 seeds 0-3 (gen env) -> VQAScore (eval) -> preference pairs
#   STAGE train: diffusers' train_diffusion_dpo.py (train env) -> dev images of the base model and of
#                the checkpoints -> judge (judge env) -> the dev-best checkpoint (data/dpo/selected.json)
#   STAGE test:  test images of base and the selected checkpoint, seeds 0-3 -> VQAScore, CLIPScore,
#                PickScore, HPSv2.1 -> judge on seed 0
#
#   cd ~/genprobe && sbatch slurm/dpo.sbatch cand smoke
#SBATCH --job-name=dpo
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
STAGE=${1:?stage: cand, train or test}; NAME=${2:?name: smoke or full}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/common.sh"
export GENPROBE_GPU_REGEX="H100 NVL"
require_gpu
cd "$GENPROBE_ROOT"
in_env() { local env=$1; shift; ( activate_env "$env"; "$@" ); }
JUDGE=${JUDGE:-gemma4_31b}
if [[ "$NAME" == smoke ]]; then
  SM=(--smoke); TP=data/prompts/dpo_train_smoke.jsonl; PAIRS=data/dpo/pairs_smoke; LORA=data/dpo/lora_smoke
  STEPS=20; CKPTS=(10 20); P=smk_; LIM=(--limit 2)
else
  SM=(); TP=data/prompts/dpo_train.jsonl; PAIRS=data/dpo/pairs; LORA=data/dpo/lora
  STEPS=2000; CKPTS=(250 500 1000 2000); P=dpo_; LIM=()
fi
DEV_ARMS=("${P}base"); DEV_SPECS=(--arm "${P}base")
for c in "${CKPTS[@]}"; do DEV_ARMS+=("${P}s$c"); DEV_SPECS+=(--arm "${P}s$c=$LORA/checkpoint-$c"); done

case "$STAGE" in
  cand)
    in_env eval python -m genprobe.dpo prompts "${SM[@]}"
    in_env gen python -m genprobe.generate --config configs/gen/sd15.yaml --prompts "$TP" --out data/dpo/cand
    in_env eval python -m genprobe.score --metric vqascore --samples "data/dpo/cand/manifests/gen_*.jsonl" \
      --tag "dpo_cand_$NAME" --prompts "$TP" --out data/dpo/cand
    # the pairs are written with pyarrow, which needs NumPy 2 (the eval env pins NumPy 1 for t2v-metrics)
    in_env train python -m genprobe.dpo pairs "${SM[@]}"
    ;;
  train)
    export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1
    # The local snapshot of SD 1.5 at the pinned revision (configs/gen/sd15.yaml) instead of the repo id:
    # under transformers 5 the script's AutoTokenizer call goes looking on the Hub for files a local
    # folder simply doesn't have, and fails offline (dpo job 94200). Same files, no lookups.
    SD15_DIR="$HF_HOME/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    [[ -f "$SD15_DIR/tokenizer/tokenizer_config.json" ]] || { echo "SD 1.5 snapshot not found at $SD15_DIR" >&2; exit 2; }
    if [[ ! -f "$LORA/checkpoint-$STEPS/pytorch_lora_weights.safetensors" ]]; then
      in_env train accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
        "$GENPROBE_ENVS/train-src/train_diffusion_dpo.py" \
        --pretrained_model_name_or_path "$SD15_DIR" --variant fp16 \
        --output_dir "$LORA" --mixed_precision fp16 --dataset_name "$PAIRS" --dataset_split_name train --resolution 512 \
        --train_batch_size 16 --gradient_accumulation_steps 2 --gradient_checkpointing --rank 8 \
        --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 --max_train_steps "$STEPS" \
        --checkpointing_steps "${CKPTS[0]}" --seed 0 --report_to tensorboard --dataloader_num_workers 4
    fi
    in_env train python -m genprobe.dpo generate "${DEV_SPECS[@]}" --split dev --seeds 0 "${LIM[@]}"
    in_env eval python -m genprobe.dpo images --arms "${DEV_ARMS[@]}" --split dev --seed 0 "${LIM[@]}" \
      --out "data/dpo/dev_images_$NAME.jsonl"
    TAG="grade_dpo_dev_${NAME}_$JUDGE"
    [[ -f "data/judge/$TAG.jsonl" ]] || in_env judge python -m genprobe.judge grade --images "data/dpo/dev_images_$NAME.jsonl" \
      --tag "$TAG" --judge "$JUDGE"
    in_env eval python -m genprobe.dpo select --grades "$TAG" --arms "${DEV_ARMS[@]}" "${LIM[@]}"
    ;;
  test)
    SEL=$(python3 -c "import json; print(json.load(open('data/dpo/selected.json'))['selected'])")
    STEP=${SEL##*_s}
    in_env train python -m genprobe.dpo generate --arm "${P}base" --arm "$SEL=$LORA/checkpoint-$STEP" --split test \
      --seeds 0 1 2 3 "${LIM[@]}"
    for m in vqascore clipscore pickscore hpsv2; do
      in_env eval python -m genprobe.score --metric "$m" --samples "data/dpo/eval/manifests/gen_*.jsonl" \
        --tag "dpo_eval_$NAME" --prompts data/prompts/prompts_v1.jsonl --out data/dpo/eval
    done
    in_env eval python -m genprobe.dpo images --arms "${P}base" "$SEL" --split test --seed 0 "${LIM[@]}" \
      --out "data/dpo/test_images_$NAME.jsonl"
    TAG="grade_dpo_test_${NAME}_$JUDGE"
    [[ -f "data/judge/$TAG.jsonl" ]] || in_env judge python -m genprobe.judge grade --images "data/dpo/test_images_$NAME.jsonl" \
      --tag "$TAG" --judge "$JUDGE"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
