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

## Decisions

| Item | Value | How I know |
| --- | --- | --- |
| Generation GPU | `nsfqs`, `gpu:h100:1` → H100 NVL 94 GB | probe (`srun ... nvidia-smi`) |
| Scoring GPU | `nsfqs`, `gpu:h100:1` → H100 NVL 94 GB (A100s fail, see below) | probe job 91529 |
| CPU jobs | `cpuqs` (summaries) | `sinfo` |
| Max job time | 2 days on gpuqs / nsfqs (jobs use ≤ 6 h) | `sinfo` |
| Driver / CUDA | 590.44.01 / CUDA 13.1 — any current PyPI torch wheel works | probe job 91208 |
| Internet | coe-hpc1: yes. g17 and compute nodes: no → `HF_HUB_OFFLINE=1` in jobs | curl tests |
| Python envs | uv venvs in `~/envs/{gen,eval}`, built on coe-hpc1 for `x86_64-manylinux_2_34` | D-012 |
| `HF_HOME` | `~/hf_home` (not scratch: locks hang from coe-hpc1) | lock test 2026-10-16 |
| uv cache | coe-hpc1 local `/tmp/$USER-uv-cache` | lock test 2026-10-16 |
| Typical queue wait | | `squeue -u $USER --start` |

## GPU-hour estimates

Fill in from the smoke run (`gen_seconds` in the manifests, warmup rows excluded):

| Model | Card | Batch | s / image | Full run (2,400 images) GPU-h |
| --- | --- | --- | --- | --- |
| sd15 | H100 NVL | | | |
| sdxl | H100 NVL | | | |
| sd35m | H100 NVL | | | |
| flux_schnell | H100 NVL | | | |
