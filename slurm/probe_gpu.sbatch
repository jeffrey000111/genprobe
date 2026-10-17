#!/bin/bash
# GPU probe on HPC3: which card did we get, can the envs built on coe-hpc1 import torch and
# use the GPU here, are the downloaded models visible?
#   cd ~/genprobe && sbatch slurm/probe_gpu.sbatch                                   H100 (generation)
#   cd ~/genprobe && sbatch -p gpuqs --gres=gpu:a100:1 slurm/probe_gpu.sbatch        A100 (scoring)
# Output: logs/probe-gpu_<jobid>.out
#SBATCH --job-name=probe-gpu
#SBATCH --partition=nsfqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/paths.sh"
echo "== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  partition: ${SLURM_JOB_PARTITION:-?}"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,memory.total,driver_version --format=csv
echo "== glibc: $(ldd --version | head -1)"
echo "== scratch: $GENPROBE_SCRATCH"
if [[ -w "$GENPROBE_SCRATCH" ]]; then echo "-> writable"; else echo "-> NOT writable (run scripts/setup_envs.sh on coe-hpc1 first)"; fi
echo "== models in $HF_HOME/hub"
ls "$HF_HOME/hub" 2>/dev/null | grep '^models--' || echo "(none yet: run scripts/download_models.sh on coe-hpc1)"

for env in gen eval; do
  py="$GENPROBE_ENVS/$env/bin/python"
  echo "== env '$env'"
  if [[ ! -x "$py" ]]; then echo "(not built yet)"; continue; fi
  "$py" - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ("torch", "diffusers", "transformers", "t2v-metrics", "open-clip-torch"):
    try:
        print(f"  {p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
import torch
print("torch cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok", tuple((x @ x).shape))
import genprobe
print("genprobe importable from", genprobe.__file__)
PY
done
