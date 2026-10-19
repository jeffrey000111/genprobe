# GenProbe

**Which automatic metrics agree with human judgment, on which kinds of prompts — and can the
trustworthy ones drive automatic fixes?**

GenProbe generates images from several open text-to-image models, scores them with pluggable
metrics (VQAScore, CLIPScore, detector- and OCR-based checks, preference models, safety-filter
false positives, cost), meta-evaluates those metrics against human ratings by skill category,
and uses the reliable ones in a generation-debugging agent.

> **Status: in progress.** M2 is done: 9,600 images (4 models × 600 prompts × 4 seeds) scored
> with 8 metrics, checkers validated on 100 blind-labelled dev images, metric sanity checks run;
> results in [`REPORT.md`](REPORT.md) and `results/prompts_v1_test/`. Next: M3, which metrics
> agree with human ratings, by skill. Every number here comes from a table in `results/`.

**Headline (test split, 480 prompts):** SD3.5 Medium and FLUX.1 schnell tie on prompt following
(VQAScore 0.680 vs 0.670, paired difference not significant) while FLUX is 4.1× faster; negation
is among the hardest skills for every model (VQAScore 0.21–0.29); rendered text is right (OCR
check) on 48 % of images at 1–2 words but 24 % at 5–8 words; CLIPScore barely separates the models. See
[`REPORT.md`](REPORT.md) and `results/prompts_v1_test/figures.md`.

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
gateway coe-hpc1, which shares `/home` with it (see `docs/hpc_notes.md`).
Paths are set in `scripts/paths.sh`.

```bash
# on coe-hpc1 (has internet), from ~/genprobe
bash scripts/setup_envs.sh gen            # uv venv in ~/envs/gen, wheels chosen for HPC3
bash scripts/setup_envs.sh eval
bash scripts/download_models.sh login     # once, HF read token (SD3.5 / FLUX are gated)
bash scripts/download_models.sh all       # generators, ~60 GB into ~/hf_home
bash scripts/download_models.sh scorers   # CLIP ViT-L/14 + Qwen2.5-VL-7B, ~18 GB
bash scripts/setup_envs.sh aux            # EasyOCR for the OCR checker
bash scripts/download_models.sh checkers  # Grounding DINO tiny + EasyOCR weights
bash scripts/download_models.sh m2        # PickScore, HPSv2.1, two safety filters, ~7.5 GB

# on HPC3 (g17), from ~/genprobe
bash scripts/hpc_check.sh                 # partitions, GPU nodes, storage -> docs/hpc_check.md
sbatch slurm/probe_gpu.sbatch             # envs import torch and see the H100?
sbatch --array=0-1 slurm/gen.sbatch configs/gen/sdxl.yaml data/prompts/smoke.jsonl 2 --seeds 0
bash scripts/submit_scoring.sh data/prompts/smoke.jsonl   # CLIPScore + VQAScore, then
                                          # results/smoke/summary.md (CPU job)
bash scripts/check_gen_env.sh && bash scripts/run_v1.sh   # M2: prompt set v1, 9,600 images
METRICS="ocr detect" bash scripts/submit_scoring.sh data/prompts/prompts_v1.jsonl   # checkers, CPU nodes
METRICS="pickscore hpsv2 nsfw_sd nsfw_vit" CPU_METRICS="" bash scripts/submit_scoring.sh data/prompts/prompts_v1.jsonl
bash scripts/submit_sanity.sh             # metric sanity checks on dev images -> results/sanity/
bash scripts/make_label_set.sh            # blind labelling sheets for checking the checkers
bash scripts/make_report_assets.sh        # figures + example panels for REPORT.md
```

Resubmitting any job is safe: finished samples are skipped, half-written files are cleaned up.

### Reproduce the smoke run with one command

```bash
# on HPC3, once the envs and weights above exist
bash scripts/run_smoke.sh     # 4 generation jobs -> CLIPScore + VQAScore jobs -> summary (CPU)
# on the laptop, when squeue is empty
scripts/hpc_sync.sh pull && cat results/smoke/summary.md
```

The summary has per-model scores with bootstrap CIs over prompts, per-category tables, metric
sanity checks, agreement with the human labels in `data/labels/`, and generation cost.

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
data/prompts/       prompt sets (in git); everything else under data/ is not, except
data/labels/        human labels (sample_id, correct = y/n/?, note), in git
genprobe/
  generate.py       batch generation: sharding + resume + manifests
  score.py          run one metric over a sample set: sharding + resume
  summarize.py      per-model / per-category tables with bootstrap CIs, cost table
  figures.py        leaderboard, paired model comparisons, category heatmap, quality-cost
                    frontier, text-length figure (run by slurm/summarize.sbatch)
  sanity.py         metric sanity checks: mismatched prompts, one-word counterfactuals,
                    repeatability (guide §7.9), results/sanity/
  label_set.py      blind labelling sets (numbered sheets, no scores or model names)
  examples.py       example panels for the report, picked by rule (results/examples/)
  contact_sheet.py  thumbnails of every image per category + a label template, for manual review
  metrics/          one plugin per metric, subclassing base.py: clipscore.py, vqascore.py, and the
                    checkers ocr.py (EasyOCR, text prompts) and detect.py (Grounding DINO,
                    counting / spatial prompts), preference scores pickscore.py and hpsv2.py,
                    safety filters safety.py (false positives on benign images)
  backends.py       diffusers backend + CPU fake backend for testing
  utils/            shard, manifest, atomic IO, environment fingerprint, HF cache checks
slurm/              job scripts (common.sh: job settings, GPU check, env activation)
scripts/            paths, env build + model download (coe-hpc1), HPC checks, laptop sync
docs/               hpc_notes.md (curated), decisions.md; hpc_check.md (raw, not in git)
results/            small tables and figures that back REPORT.md
```

## Traceability

Every generation record stores the config hash, git commit, GPU, run ID, seed, scheduler,
timing and peak memory; every score record stores the metric version (library version,
weights revision, prompt template and batch size), so results from different scorer setups
are never averaged together.
A full environment fingerprint is written once per job to `data/runs/`.

## Models and licenses

| Model | License |
| --- | --- |
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M |
| SDXL base 1.0 | CreativeML OpenRAIL++-M |
| Stable Diffusion 3.5 Medium | Stability AI Community License |
| FLUX.1 [schnell] | Apache-2.0 |
| Scoring: OpenAI CLIP ViT-L/14 (CLIPScore) | MIT |
| Scoring: Qwen2.5-VL-7B-Instruct (VQAScore) | Apache-2.0 |
| Scoring: Grounding DINO tiny (detector checker) | Apache-2.0 |
| Scoring: EasyOCR CRAFT + english_g2 (OCR checker) | Apache-2.0 |
| Scoring: PickScore v1 | none stated on its model page |
| Scoring: HPSv2.1 | Apache-2.0 |
| Scoring: SD safety checker (CompVis) | none stated on its model page |
| Scoring: Falconsai NSFW image detection (ViT) | Apache-2.0 |

No model weights or generated images are committed to this repository.
