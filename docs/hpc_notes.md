# HPC notes

SJSU College of Engineering HPC. Curated notes, edited on the laptop. Raw output of
`bash scripts/hpc_check.sh` goes to `docs/hpc_check.md` on the cluster and comes back with
`scripts/hpc_sync.sh pull`; probe-job logs come back under `logs/`.

## What we found on 2026-10-16 (login MOTD, `sinfo`, `df`, probe jobs)

- **Two machines, one filesystem.** coe-hpc1 (CentOS 7, glibc 2.17) has **internet** but no
  working Slurm. HPC3 (login node g17, Rocky Linux, glibc 2.34) runs Slurm, but neither g17
  nor its compute nodes can reach the internet. `/home` (and `/scratch`) are the same on both
  (on compute nodes `/home` → `/fs/atipa/home`, same paths). So: install and download on
  coe-hpc1, run jobs on HPC3 (D-012).
- **Login path:** VPN → `coe-hpc1.sjsu.edu` (SJSUOne password) → `coe-hpc3` (ed25519 key).
  From the laptop: SSH aliases `sjsu` (hpc1) and `sjsu3` (HPC3 via ProxyJump).
- **Access** comes through the CMPE-261 (Fall 2026) class allocation, max 6 months. The course
  instructor's rule: never run jobs on the head node.
- **The old HPC web page is out of date** (its node list and partition name `gpu` no longer apply).
- **GPU isolation is by `CUDA_VISIBLE_DEVICES` only.** Slurm sets it, but `nvidia-smi` in a job
  still lists every GPU on the node, so scripts query `nvidia-smi -i $CUDA_VISIBLE_DEVICES`.
- **File locks on `/scratch` hang when taken from coe-hpc1** (`fcntl.flock` never returns; a
  `uv python install` sat in `futex_wait` for 77 min with only `.lock` and an `.nfs*` file in
  its cache). Locks on `/home` work. uv and huggingface_hub both lock while downloading, so
  nothing that coe-hpc1 downloads may live on `/scratch` (D-011).

### Partitions on HPC3 (all usable by this account; `sbatch --test-only`)

| Partition | Max time | Nodes / GPUs |
| --- | --- | --- |
| defq (default), cpuqs / cpuqm / cpuql | 4–6 h, 5 d / 14 d / 21 d | CPU only |
| **gpuqs** / gpuqm / gpuql | **2 d** / 7 d / 14 d | cs001–cs004: 4 × **A100-PCIE-40GB** each; g16, g18, g19: `gpu:h100:1`; g1–g15: P100 (many down) |
| **nsfqs** / nsfqm / nsfql | **2 d** / 14 d / 21 d | g20–g32: 1 × **H100 NVL 94 GB** each (NSF CC* grant cluster) |
| condo, chemq | 30 d | faculty queues |

GRES are typed, so jobs ask for a card directly (`--gres=gpu:h100:1`, `--gres=gpu:a100:1`).

**The gpuqs A100 nodes don't work for us (2026-10-17).** Probe job 91529 on cs00x: torch in
both envs reports "CUDA-capable device(s) is/are busy or unavailable". `CUDA_VISIBLE_DEVICES`
is always `0` there and `nvidia-smi` lists all four cards, so the job is probably handed a
card that another job already holds (or the cards are in exclusive-process mode). Everything
runs on nsfqs instead (D-009). Worth reporting to the HPC admins with the job ID.

### Storage

| Mount | Size | Free | Notes |
| --- | --- | --- | --- |
| `/home` | 124T | 2.4T (99% used) | code, venvs (`~/envs`), weights (`~/hf_home`, ~60 GB), `data/`; keep small |
| `/scratch/cmpe261-fa26` | 108T pool | 13T | class dir (group `cmpe261-fa26`). **Purged each semester** (MOTD). File locks hang from coe-hpc1, so unused for now |
| coe-hpc1 `/tmp` (local disk) | 100G | 32G | uv download cache (disposable) |
| `/data` | 99T | 7.2T (93% used) | not used |

`~/.cache` held 38 GB from earlier projects (33 GB Hugging Face, 5 GB pip) — candidates to clean.

**HF cache layout.** Downloads on coe-hpc1 use huggingface_hub 2.x, which keeps the file
contents in one shared `~/hf_home/hub/blobs/`; the per-model `models--*` folders only hold
refs and symlinks, so `du` on them looks tiny (all four generators: ~55 GB in `blobs/`). The
envs' older huggingface_hub (gen: 1.33, eval: 0.x via transformers 4.49) read this cache
fine, since they only follow `refs/main` and the snapshot symlinks.

### Env versions (lock files in `env/`, 2026-10-17)

| Env | Python | torch (CUDA) | Key packages |
| --- | --- | --- | --- |
| gen | 3.11 | 2.14.1 (cu13.0) | diffusers 0.41.0, transformers 5.20.1, huggingface-hub 1.33.0 |
| eval | 3.11 | 2.5.1 (cu12.4) | t2v-metrics 3.0, transformers 4.49.0, open-clip-torch 3.3.0, torchvision 0.20.1 |

The eval env pins an older torch because t2v-metrics 3.0 pins transformers 4.49; that is why
generation and scoring live in separate envs.

**t2v_metrics 3.0 doesn't import here (2026-10-18), and the plugins no longer need it.**
First error: "ffmpeg is a required system requirement but not found" (it checks PATH at
import, even for images). imageio-ffmpeg, installed as a t2v-metrics dependency, bundles a
static ffmpeg 7.0.2, so `setup_envs.sh eval` links it into the venv
(`~/envs/eval/bin/ffmpeg -> .../site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`).
Second error: `No module named 'llava'`: the PyPI release imports every backbone up front and
expects git-only packages (LLaVA-NeXT, OpenAI CLIP, a pytorchvideo fork) per its
V_3.0_README. VQAScore is therefore computed by our own port of its Qwen path (D-014).
`scripts/check_eval_env.sh` checks everything the plugins need on a CPU node in minutes
(weights cached, weight names match transformers 4.49, processor and chat template on real
smoke images), so a long GPU wait is never wasted on an import or loading error.

**Queue waits.** On the night of 2026-10-17/18, nsfqs was full: two 2-hour scoring jobs
were estimated to start ~5.5 h after submission (gpuqs H100s: ~11 h), and actually started
in the morning. Short `--time` limits help Slurm backfill; check env problems on `cpuqs`
first so a long GPU wait isn't wasted on an import error.

## Decisions

| Item | Value | How I know |
| --- | --- | --- |
| Generation GPU | `nsfqs`, `gpu:h100:1` → H100 NVL 94 GB | probe (`srun ... nvidia-smi`) |
| Scoring GPU | `nsfqs`, `gpu:h100:1` → H100 NVL 94 GB (A100s fail, see below) | probe job 91529 |
| CPU jobs | `cpuqs` (summaries) | `sinfo` |
| Max job time | 2 days on gpuqs / nsfqs (jobs use ≤ 6 h) | `sinfo` |
| Driver / CUDA | 590.44.01 / CUDA 13.1 — any current PyPI torch wheel works | probe job 91208 |
| Internet | coe-hpc1: yes. g17 and compute nodes: no → `HF_HUB_OFFLINE=1` in jobs | curl tests |
| Python envs | uv venvs in `~/envs/{gen,eval,aux}` (aux: EasyOCR), built on coe-hpc1 for `x86_64-manylinux_2_34` | D-012, D-017 |
| `HF_HOME` | `~/hf_home` (not scratch: locks hang from coe-hpc1) | lock test 2026-10-16 |
| uv cache | coe-hpc1 local `/tmp/$USER-uv-cache` | lock test 2026-10-16 |
| Typical queue wait | | `squeue -u $USER --start` |

## GPU-hour estimates

From the smoke run (20 images per model, seed 0; median of the non-warmup images), copied
from `results/smoke/summary.md` (commit 7ec27f3). Smoke-set numbers, for planning only (D-010):

| Model | Card | Batch | Steps | s / image | Peak mem GB | Full run (2,400 images) GPU-h |
| --- | --- | --- | --- | --- | --- | --- |
| sd15 | H100 NVL | 4 | 50 | 0.53 | 4.3 | 0.35 |
| sdxl | H100 NVL | 2 | 30 | 1.90 | 15.8 | 1.27 |
| sd35m | H100 NVL | 2 | 40 | 4.65 | 20.7 | 3.10 |
| flux_schnell | H100 NVL | 1 | 4 | 1.13 | 36.3 | 0.75 |

All four generators together: ~5.5 GPU-hours for 2,400 images each, plus model loading.
Prompt set v1 (600 prompts × 4 seeds = 2,400 images per model) with SDXL at 50 steps instead
of 30 (D-006): SD1.5 ~0.35, SDXL ~2.1, SD3.5M ~3.1, FLUX ~0.75 GPU-h, plus ~3 min of loading per
shard (16 shards) ≈ 7 GPU-h; scoring 9,600 images with CLIPScore + VQAScore ≈ 1–1.5 GPU-h.
`scripts/run_v1.sh` submits it all (4 generation shards per model, ≤ 2 at a time per model).
