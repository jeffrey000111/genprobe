# Decision log

Every choice that affects a number in REPORT.md goes here, with the reason. Newest last.
Format: **D-NNN (date) — decision.** Why. What would change it.

## Infrastructure

**D-001 (2026-10-16) — `sample_id = {model}__{prompt_id}__s{seed}`; images at
`{out}/images/{model}/{prompt_id}_s{seed}.png`.** One join key for every manifest and score
table. Model names and prompt IDs may not contain `__`, enforced in code.

**D-002 (2026-10-16) — Shards = `sorted(samples, key=sample_id)[k::n]`; manifest files are
named `gen_{model}_shard{k}of{n}.jsonl`.** Sorting makes shard membership independent of
input order. Including `n` in the filename means re-sharding never appends two runs to one
file. Resume reads *all* of a model's manifests, so changing `n` between runs is safe.

**D-003 (2026-10-16) — A sample counts as done when its latest record has status `ok` or
`blank`, was produced with the current `config_hash`, and (for `ok`) the image file exists.**
`blank` is terminal because the same seed and settings reproduce the same blank image;
retrying would loop. Blank and error rates are reported per model instead. Any config change
invalidates old samples, which are regenerated and overwritten.

**D-004 (2026-10-16) — Timing: CUDA events around the pipeline call, divided by batch size;
the first batch of every job is flagged `warmup: true` and excluded from timing statistics.**
Keeps the images (no wasted compute) while removing CUDA init/compile cost. Per-image times
are only compared within one GPU type and batch size, never across offload settings.

**D-005 (2026-10-16) — SD1.5 runs with `safety_checker=None`.** The bundled checker blacks
out benign images, which would corrupt every quality and alignment metric. Safety-filter
behaviour is measured separately as false positives on benign images (guide §7.8).

**D-008 (2026-10-16) — Code is edited on the laptop and pushed to the cluster with
`scripts/hpc_sync.sh` (including `.git`); the cluster never commits.** Each side owns its own
files: the laptop owns code, configs and prompt sets; the cluster owns `data/`, `results/`,
`logs/`, env lock files and `docs/hpc_check.md` (raw `hpc_check.sh` output; curated notes stay in the laptop-owned `docs/hpc_notes.md`), which are pulled back. `git_commit` counts a
checkout as dirty only for changes under code paths, so job outputs that are tracked in git
(e.g. `results/`) don't mark a clean run as `-dirty`.

**D-009 (2026-10-16) — All generation runs on A100s in HPC3's `gpuqs` (`--gres=gpu:a100:1`).**
The four cs00x nodes hold 16 A100s, far more capacity than the three H100s in the same
queue, and a single card type keeps per-image timings comparable across models (guide §4.2,
§8.4). Scoring may use H100s since its timings are not compared.

**D-011 (2026-10-16) — Model weights and pip caches live in `/scratch/$USER`; code, conda envs
and `data/` stay in `/home`.** `/home` is shared and 99% full; `/scratch` has 13 TB free but is
purged each semester, so only re-downloadable files go there. If scratch is wiped, weights
are re-downloaded; generated images are never at risk.

**D-006 (TODO) — Generation parameters.** Use each model card's recommended settings, no
tuning, to represent out-of-the-box quality. Re-check the SDXL card for steps/cfg and record
the final values and source here before the full run.

**D-007 (TODO) — Batch size per model.** Set after the smoke run from `peak_mem_gb`; fixed
for the whole project because it enters `config_hash` and can shift numerics slightly.

## Prompt set

**D-010 (2026-10-16) — `data/prompts/smoke.jsonl` (20 template prompts: 7 counting,
7 spatial, 6 text) is only for pipeline testing in M1.** It is not part of prompt set v1
and no reported number comes from it.

## Pre-registered hypotheses (module B)

Drafted from the guide (§9.1) before any meta-evaluation was run. Review and finalize before
M3; anything not listed here is reported as exploratory.

- **H1** — On counting prompts, VQAScore's pairwise agreement with human ratings is higher
  than CLIPScore's.
- **H2** — Preference models (PickScore etc.) agree with humans on instruction following less
  often than VQAScore does.
- **H3** — The VLM judge shows position bias in pairwise comparisons (verdict changes when
  the two images are swapped).
