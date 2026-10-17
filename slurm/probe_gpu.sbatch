#!/bin/bash
# Step 0 probe on HPC3: which A100 do we get (40 or 80 GB), driver/CUDA version, can the
# compute node reach the internet and see /scratch, does torch work?
#   cd ~/genprobe && sbatch slurm/probe_gpu.sbatch     then read logs/probe-gpu_<jobid>.out
#SBATCH --job-name=probe-gpu
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
echo "== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  partition: ${SLURM_JOB_PARTITION:-?}"
echo "== nvidia-smi"; nvidia-smi
echo "== GPU summary"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== internet from compute node"
if curl -sI --max-time 10 https://huggingface.co | head -1; then echo "-> online"; else echo "-> OFFLINE: keep HF_HUB_OFFLINE=1"; fi
echo "== disks visible from compute node"
df -h "$HOME" /scratch /data 2>&1 | sed 's/^/   /'
if mkdir -p "/scratch/$USER" 2>/dev/null && touch "/scratch/$USER/.probe" 2>/dev/null; then
  echo "-> /scratch/$USER is writable"; rm -f "/scratch/$USER/.probe"
else
  echo "-> /scratch/$USER NOT writable: set GENPROBE_SCRATCH in slurm/common.sh"
fi
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  if conda env list | grep -qE '^gen\s'; then
    conda activate gen
    echo "== torch in 'gen' env"
    python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", (x @ x).shape)
PY
  else
    echo "(no 'gen' env yet; rerun this probe after scripts/setup_envs.sh)"
  fi
fi
