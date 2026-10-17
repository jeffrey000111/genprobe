# GenProbe

**Which automatic metrics agree with human judgment, on which kinds of prompts — and can the
trustworthy ones drive automatic fixes?**

GenProbe generates images from several open text-to-image models, scores them with pluggable
metrics (VQAScore, CLIPScore, detector- and OCR-based checks, preference models, safety-filter
false positives, cost), meta-evaluates those metrics against human ratings by skill category,
and uses the reliable ones in a generation-debugging agent.

> **Status: in progress (M0 — infrastructure).** No results yet. Every number that appears
> here later comes from a table in `results/`.

## Quick start (no GPU needed)

```bash
pip install -e .
make test          # sharding, resume, metric interface, config validation
make smoke-fake    # 20 prompts x 4 model configs with fake images -> data/_fake/
```

`smoke-fake` runs the full generation path (sharding, resume, atomic writes, manifests)
with a CPU backend that draws noise images, so the plumbing can be checked on a login node.

## Real run on Slurm

Written for SJSU CoE HPC3, whose nodes are offline; installs and downloads happen on the
gateway coe-hpc1, which shares `/home` and `/scratch` with it (see `docs/hpc_notes.md`).
Paths are set in `scripts/paths.sh`.

```bash
# on coe-hpc1 (has internet), from ~/genprobe
bash scripts/setup_envs.sh gen            # uv venv in ~/envs/gen, wheels chosen for HPC3
bash scripts/setup_envs.sh eval
bash scripts/download_models.sh login     # once, HF read token (SD3.5 / FLUX are gated)
bash scripts/download_models.sh all       # ~60 GB into the class scratch dir

# on HPC3 (g17), from ~/genprobe
bash scripts/hpc_check.sh                 # partitions, GPU nodes, storage -> docs/hpc_check.md
sbatch slurm/probe_gpu.sbatch             # envs import torch and see the H100?
sbatch --array=0-1 slurm/gen.sbatch configs/gen/sdxl.yaml data/prompts/smoke.jsonl 2 --seeds 0
```

Resubmitting any job is safe: finished samples are skipped, half-written files are cleaned up.

## Working from a laptop

Code is edited and unit-tested locally; the cluster only runs jobs. With an SSH alias `sjsu3`
(see `scripts/hpc_sync.sh` for the expected `~/.ssh/config` entries):

```bash
scripts/hpc_sync.sh push    # committed code, configs, prompt sets -> cluster
ssh sjsu3 "cd ~/genprobe && sbatch --array=0-1 slurm/gen.sbatch configs/gen/sdxl.yaml data/prompts/smoke.jsonl 2 --seeds 0"
scripts/hpc_sync.sh pull    # results/, logs/, env locks, docs/hpc_check.md -> laptop
```

`CLAUDE.md` and `.claude/settings.json` hold the working rules for Claude Code; submitting or
cancelling cluster jobs always asks for confirmation.

## Layout

```text
configs/gen/        one YAML per model (validated by pydantic; hashed into every record)
data/prompts/       prompt sets (in git); everything else under data/ is not
genprobe/
  generate.py       batch generation: sharding + resume + manifests
  score.py          run one metric over a sample set: sharding + resume
  metrics/          one plugin per metric, subclassing metrics/base.py
  backends.py       diffusers backend + CPU fake backend for testing
  utils/            shard, manifest, atomic IO, environment fingerprint
slurm/              job scripts (common.sh: job settings, GPU check, env activation)
scripts/            paths, env build + model download (coe-hpc1), HPC checks, laptop sync
docs/               hpc_notes.md (curated), hpc_check.md (raw), decisions.md
results/            small tables and figures that back REPORT.md
```

## Traceability

Every generation record stores the config hash, git commit, GPU, run ID, seed, scheduler,
timing and peak memory; every score record stores the metric version (including batch size).
A full environment fingerprint is written once per job to `data/runs/`.

## Models and licenses

| Model | License |
| --- | --- |
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M |
| SDXL base 1.0 | CreativeML OpenRAIL++-M |
| Stable Diffusion 3.5 Medium | Stability AI Community License |
| FLUX.1 [schnell] | Apache-2.0 |

No model weights or generated images are committed to this repository.
