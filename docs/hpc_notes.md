# HPC notes

SJSU College of Engineering HPC. Curated notes, edited on the laptop. Raw output of
`bash scripts/hpc_check.sh` goes to `docs/hpc_check.md` on the cluster and comes back with
`scripts/hpc_sync.sh pull`; fill the tables below from it and from `slurm/probe_gpu.sbatch`.

## What we found on 2026-10-16 (login MOTD, `sinfo`, `df`)

- **HPC1 is only a gateway now.** Its compute and GPU nodes were migrated to **HPC3**; Slurm on
  coe-hpc1 no longer answers. Path: VPN → `coe-hpc1.sjsu.edu` (SJSUOne password) →
  `ssh coe-hpc3` (key in `~/.ssh/id_ed25519`, shared home). HPC3's login node is `g17`.
- **Access** comes through the CMPE-261 (Fall 2026) class allocation, max 6 months. The course
  instructor's rule: never run jobs on the head node.
- **The old HPC page is out of date** (its node list and partition name `gpu` no longer apply).

### Partitions on HPC3

| Partition | Max time | Nodes / GPUs |
| --- | --- | --- |
| defq (default), cpuqs / cpuqm / cpuql | 4–6 h, 5 d / 14 d / 21 d | CPU only |
| **gpuqs** / gpuqm / gpuql | **2 d** / 7 d / 14 d | cs001–cs004: `gpu:a100:4` each (16 A100); g16, g18, g19: `gpu:h100:1`; g1–g15: `gpu:p100:1` (many down) |
| nsfqs / nsfqm / nsfql | 2 d / 14 d / 21 d | g20–g32: `gpu:h100:1` (NSF CC* grant cluster; access to be checked) |
| condo, chemq | 30 d | faculty queues |

GRES are typed, so jobs ask for a card directly: `--gres=gpu:a100:1` (D-009).

### Storage on HPC3

| Mount | Size | Free | Notes |
| --- | --- | --- | --- |
| `/home` (shared with coe-hpc1) | 124T | 2.4T (99% used) | code, conda envs, `data/`; keep small |
| `/scratch` → `/fs/sfs/scratch` | 108T | 13T (89% used) | **purged: MOTD says files are deleted after 21 days of each semester** — weights and caches only |
| `/data` | 99T | 7.2T (93% used) | not used |
| `/opt` (local) | 693G | 299G | not used |

## Decisions

| Item | Value | How I know |
| --- | --- | --- |
| Login path | laptop → coe-hpc1 → coe-hpc3 (g17) | MOTD |
| GPU partition / GRES | `gpuqs`, `gpu:a100:1` | `sinfo` |
| Max job time | 2 days on gpuqs (jobs use ≤ 6 h) | `sinfo` |
| GPU memory (40 GB or 80 GB) | | probe job |
| Driver / max CUDA version | | probe job → choose `CU=` for setup_envs.sh |
| Compute nodes online? | | probe job → `HF_HUB_OFFLINE` in slurm/common.sh |
| `HF_HOME` | `/scratch/$USER/hf_home` | storage table above |
| nsfqs usable by this account? | | `hpc_check.sh` (`sbatch --test-only`) |
| Typical queue wait | | `squeue -u $USER --start` |

## GPU-hour estimates

Fill in from the smoke run (`gen_seconds` in the manifests, warmup rows excluded):

| Model | Card | Batch | s / image | Full run (2,400 images) GPU-h |
| --- | --- | --- | --- | --- |
| sd15 | | | | |
| sdxl | | | | |
| sd35m | | | | |
| flux_schnell | | | | |
