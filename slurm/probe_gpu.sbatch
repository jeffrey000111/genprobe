#!/bin/bash
# GPU probe on HPC3: which card did we get, can the envs built on coe-hpc1 import torch and
# use the GPU here, are the downloaded models visible?
#   cd ~/genprobe && sbatch slurm/probe_gpu.sbatch                                   H100 NVL (all jobs)
#   cd ~/genprobe && sbatch -p gpuqs --gres=gpu:a100:1 slurm/probe_gpu.sbatch        A100 (broken for us, D-009)
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
echo "== C compiler (triton, used by vLLM, compiles small helpers at run time): $(command -v gcc || command -v cc || echo none)"
echo "== models in $HF_HOME/hub"
ls "$HF_HOME/hub" 2>/dev/null | grep '^models--' || echo "(none yet: run scripts/download_models.sh on coe-hpc1)"

for env in gen eval aux judge; do
  py="$GENPROBE_ENVS/$env/bin/python"
  echo "== env '$env'"
  if [[ ! -x "$py" ]]; then echo "(not built yet)"; continue; fi
  PATH="$GENPROBE_ENVS/$env/bin:$PATH" "$py" - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "t2v-metrics",
          "open-clip-torch", "qwen-vl-utils", "easyocr", "opencv-python-headless", "decord", "flash-attn",
          "vllm", "triton", "flashinfer-python"):
    try:
        print(f"  {p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
for mod in ("diffusers", "open_clip", "qwen_vl_utils", "easyocr", "vllm"):
    try:
        __import__(mod)
        print(f"  import {mod}: ok")
    except ModuleNotFoundError as e:
        if e.name == mod:
            continue  # not part of this env
        print(f"  import {mod}: FAILED {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  import {mod}: FAILED {type(e).__name__}: {e}")
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
