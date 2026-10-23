#!/bin/bash
# Diffusion-DPO on SD 1.5 (guide §11.4, genprobe/dpo.py, D-037): the three GPU stages of
# slurm/dpo.sbatch in order, the CPU checkers (detector, OCR) on the test images, then the analysis.
# Run on HPC3's login node (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_dpo.sh smoke"
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_dpo.sh full"     after approval (about 6 GPU-hours)
#
# Output: results/dpo/{dev_selection,summary}[_smoke].md; images and LoRA weights stay in data/dpo/.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_dpo.sh smoke|full}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
case "$NAME" in
  smoke) TC=00:40:00; TT=01:00:00; TE=00:50:00; LIM="--limit 2"; P=smk_ ;;
  full) TC=02:30:00; TT=03:00:00; TE=03:30:00; LIM=""; P=dpo_ ;;
  *) echo "usage: run_dpo.sh smoke|full" >&2; exit 2 ;;
esac
mkdir -p logs
c=$(sbatch --parsable --time="$TC" slurm/dpo.sbatch cand "$NAME"); c=${c%%;*}
t=$(sbatch --parsable --time="$TT" --dependency="afterok:$c" slurm/dpo.sbatch train "$NAME"); t=${t%%;*}
e=$(sbatch --parsable --time="$TE" --dependency="afterok:$t" slurm/dpo.sbatch test "$NAME"); e=${e%%;*}
echo "submitted DPO $NAME: candidates $c -> training + dev choice $t -> test images, scores, judge $e"
IDS=$(mktemp)
SAMPLES="data/dpo/eval/manifests/gen_*.jsonl" METRICS="detect ocr" SCORE_SHARDS=4 SUMMARY=0 TAG="dpo_eval_$NAME" \
  DEPENDENCY="afterok:$e" IDS_FILE="$IDS" bash scripts/submit_scoring.sh data/prompts/prompts_v1.jsonl
dep=$(paste -sd: "$IDS"); rm -f "$IDS"
a=$(sbatch --parsable --dependency="afterany:$e:$dep" -p cpuqs -c 2 --mem=8G -t 00:20:00 -J dpo-analyze -o logs/%x_%j.out \
    --wrap "source scripts/paths.sh && source \$GENPROBE_ENVS/eval/bin/activate && python -m genprobe.dpo analyze --grades grade_dpo_test_${NAME}_gemma4_31b --base ${P}base --dpo \$(python3 -c \"import json; print(json.load(open('data/dpo/selected.json'))['selected'])\") $LIM")
echo "submitted analysis: job ${a%%;*} -> results/dpo/"
