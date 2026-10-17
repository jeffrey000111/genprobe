#!/bin/bash
# Step 0 (guide §3.1): run on the HPC3 login node (g17); appends everything to docs/hpc_notes.md.
#   cd ~/genprobe && bash scripts/hpc_check.sh
# Light commands only (sinfo, df, a dry-run sbatch) — safe on a login node.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=docs/hpc_notes.md
mkdir -p logs

{
  echo
  echo "## hpc_check $(date '+%Y-%m-%d %H:%M') on $(hostname)"
  echo
  echo '```text'
  echo "### partitions"
  sinfo -s 2>&1
  echo
  echo "### GPU nodes: name, partition, GRES, state, memory MB"
  sinfo -h -N -o "%N %P %G %T %m" 2>/dev/null | awk '$3 ~ /gpu/' | sort -u
  echo
  echo "### can this account use these queues? (sbatch --test-only submits nothing)"
  for spec in "gpuqs gpu:a100:1" "gpuqs gpu:h100:1" "nsfqs gpu:h100:1"; do
    set -- $spec
    printf -- "--- -p %s --gres=%s: " "$1" "$2"
    sbatch --test-only -p "$1" --gres="$2" -t 00:05:00 --wrap=hostname 2>&1 | tail -1
  done
  echo
  echo "### my queue"
  squeue -u "$USER" 2>&1
  echo
  echo "### storage"
  df -h "$HOME" /scratch /data 2>&1
  du -sh "$HOME" 2>/dev/null
  [[ -d "/scratch/$USER" ]] && du -sh "/scratch/$USER" 2>/dev/null
  echo
  echo "### internet from login node"
  curl -sI --max-time 10 https://huggingface.co | head -1 || echo "no internet from login node"
  curl -sI --max-time 10 https://github.com | head -1 || echo "no github from login node"
  echo '```'
} >>"$OUT" 2>&1

echo "Appended to $OUT. Next:"
echo "  1. Read it; the --test-only lines show which GPU queues your account may use"
echo "  2. sbatch slurm/probe_gpu.sbatch && squeue -u \$USER   (output: logs/probe-gpu_<jobid>.out)"
