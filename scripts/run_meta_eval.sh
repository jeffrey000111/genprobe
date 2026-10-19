#!/bin/bash
# Metric meta-evaluation on GenAI-Bench (genprobe/meta_eval.py; pre-registered H1, H2 and the
# human ceiling in docs/decisions.md), on a CPU node. Refuses to compute anything until every
# metric has scored all 9,600 images. On HPC3, after the full scoring run:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_meta_eval.sh"
#
# Output: results/meta_eval/{hypotheses.md, agreement_by_tag.csv, summary.json, fig_meta_eval_heatmap.png}.
set -euo pipefail
cd "$(dirname "$0")/.."
METRICS=${METRICS:-clipscore vqascore pickscore hpsv2}
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -n 1 -c 2 --mem=8G -t 00:20:00 -J meta-eval --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.meta_eval --metrics $METRICS"
