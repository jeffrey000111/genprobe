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

**D-009 (2026-10-16, revised 2026-10-17) — All generation and scoring runs on H100 NVL 94 GB
in HPC3's `nsfqs` (`--gres=gpu:h100:1`, jobs check the card name).** First choice for
generation was the A100s (16 cards in gpuqs), but they have 40 GB and FLUX.1-schnell needs
~50 GB without CPU offload; offloading would make its timing incomparable. nsfqs has 13
identical H100 NVLs, and one card type keeps per-image timings comparable across models
(guide §4.2, §8.4). Scoring was then planned for the A100s to leave the H100s free, but probe
job 91529 on cs00x failed in both envs with "CUDA-capable device(s) is/are busy or
unavailable" (Slurm always sets `CUDA_VISIBLE_DEVICES=0` there and `nvidia-smi` lists all
four cards, so the job is likely sent to a card someone else holds). Scoring is short and
resumable, so it moves to nsfqs too. Revisit if the A100 nodes start working (ask the HPC
admins) or nsfqs queues get long.

**D-011 (2026-10-16, revised same day) — Model weights live in `~/hf_home`; uv's download
cache lives on coe-hpc1's local `/tmp`; nothing is downloaded to `/scratch`.** The first plan
put weights and caches in the class scratch dir (13 TB free, purged each semester) to keep
`/home` (99% full) small. In practice `uv python install` hung for 77 minutes on coe-hpc1, and
a direct test showed why: `flock` on `/scratch` never returns from coe-hpc1, while on `/home`
it succeeds. uv and huggingface_hub both lock files while downloading, so they cannot write
to `/scratch` from the only machine with internet. Cost: ~60 GB of weights on `/home`, offset
by cleaning old caches. The HF token stays in `~/.cache/huggingface/token`.

**D-012 (2026-10-16) — Python envs are uv virtualenvs built on coe-hpc1 for HPC3's platform.**
HPC3 and its compute nodes are offline; coe-hpc1 is online and shares `/home` with them,
but runs CentOS 7 (glibc 2.17) while HPC3 has glibc 2.34. `scripts/setup_envs.sh` therefore
uses `uv pip install --python-platform x86_64-manylinux_2_34`, so wheels are chosen for HPC3
rather than for the machine running the install; uv and its managed Python need only glibc
2.17, so the same interpreter runs on both. Envs are verified on a GPU node by
`slurm/probe_gpu.sbatch`, never by importing them on coe-hpc1. Conda (used in the original
guide) is dropped: it can't install offline on HPC3, and on coe-hpc1 it would target CentOS 7.
Source-only dependencies are built with `setuptools<82` (`scripts/build-constraints.txt`):
image-reward, pulled in by t2v-metrics, still imports `pkg_resources`, removed in setuptools 82.
PyPI's unrelated `apex` package (listed by t2v-metrics 3.0) is excluded (`scripts/excludes.txt`).
The eval env gets an `ffmpeg` by linking imageio-ffmpeg's bundled static binary into the
venv's `bin/` (2026-10-18): `import t2v_metrics` refuses to run without one, and HPC3 has no
system ffmpeg or internet to fetch one. Only video scoring would actually call it.

**D-006 (TODO) — Generation parameters.** Use each model card's recommended settings, no
tuning, to represent out-of-the-box quality. Re-check the SDXL card for steps/cfg and record
the final values and source here before the full run.

**D-007 (TODO) — Batch size per model.** Set after the smoke run from `peak_mem_gb`; fixed
for the whole project because it enters `config_hash` and can shift numerics slightly.

## Metrics

**D-013 (2026-10-17) — CLIPScore = cosine similarity of OpenAI CLIP ViT-L/14 (224 px)
embeddings, computed with open_clip directly (not through t2v_metrics), fp32, batch 32.**
`value` is the raw cosine; Hessel et al.'s `2.5 × max(cos, 0)` is stored in
`extra["clipscore_w2.5"]` (identical ranking whenever cos > 0). ViT-L/14 follows the guide
(§7.3). Doing it in open_clip makes the activation explicit: OpenAI weights need QuickGELU,
and open_clip 3.3 only *warns* when the `openai` tag is loaded without it, so the plugin
forces it, turns that warning into an error and checks the built model. Prompts over 77 CLIP
tokens are truncated by CLIP and flagged `extra["truncated"]`. Weights:
`timm/vit_large_patch14_clip_224.openai` (the HF copy open_clip itself loads).

**D-014 (2026-10-17, revised 2026-10-18) — VQAScore = P("Yes") from Qwen2.5-VL-7B-Instruct
(bf16, SDPA attention) for the question `Does this figure show "{}"? Please answer Yes or No.`,
computed by `genprobe/metrics/vqascore.py`, a direct port of t2v_metrics' Qwen scoring path.**
Qwen2.5-VL-7B is the guide's main backbone (§7.4); the paper's CLIP-FlanT5 stays optional.
The first plan was to call t2v_metrics 3.0 (the newest release on PyPI; the guide's 3.1 is
GitHub-only and needs transformers 5). On HPC3 its import failed twice before reaching any
model: first it requires an `ffmpeg` program on PATH (fixed by linking imageio-ffmpeg's
binary), then `No module named 'llava'`: the PyPI package imports every backbone at import
time, and some need git-only packages (LLaVA-NeXT, which pins torch 2.1; OpenAI CLIP; a
pytorchvideo fork) that the Qwen path never uses. Rather than install conflicting packages
the Qwen backbone doesn't need, the plugin reproduces t2v_metrics' `Qwen2VLModel.forward`
on transformers 4.49 + qwen-vl-utils: same chat message (image, then question; no system
message), `process_vision_info` resizing with `do_resize=False`, `generate(max_new_tokens=1,
do_sample=False, temperature=1.0, output_scores=True)` and the softmax over the full
vocabulary at the "Yes" token. Using `generate` (not raw forward logits) keeps the model's
generation-config processors (`repetition_penalty=1.05`), as t2v_metrics does. The template
hash and the weights' commit are part of `metric_version`; the plugin also refuses a partial
weight load and records P("No") per image as a diagnostic. **Open item:** check parity with
t2v_metrics itself on ~20 images once an env where it imports exists (3.1 from source), and
report the max absolute difference. A second backbone from another family will be added as a
control before module B (guide §7.4).

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
