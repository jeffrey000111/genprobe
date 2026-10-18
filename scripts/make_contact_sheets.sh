#!/bin/bash
# Build contact sheets of the smoke images (one JPEG per category, every model side by side,
# scores under each tile) and a label template, on a CPU node. Run on HPC3:
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/make_contact_sheets.sh"
#   PROMPTS=data/prompts/other.jsonl bash scripts/make_contact_sheets.sh
#
# Output: data/review/<prompts stem>_<category>.jpg and <prompts stem>_labels.csv (never
# overwritten). Copy them to the laptop (into the git-ignored data/ folder) with:
#   mkdir -p data/review && scp 'sjsu3:genprobe/data/review/*' data/review/
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${PROMPTS:-data/prompts/smoke.jsonl}
command -v srun >/dev/null || { echo "srun not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
srun -p cpuqs -c 2 --mem=8G -t 00:10:00 -J contact-sheets --chdir="$PWD" bash -c "
  source scripts/paths.sh
  source \"\$GENPROBE_ENVS/eval/bin/activate\"
  python -m genprobe.contact_sheet --prompts '$PROMPTS' --out data/review"
