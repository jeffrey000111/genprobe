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

**D-006 (2026-10-19) — Generation parameters = what each model card's own example runs, no
tuning (out-of-the-box quality).** SD1.5: 50 steps, cfg 7.5, 512 px (card example calls
`pipe(prompt)` → pipeline defaults). SDXL base: 50 steps, cfg 5.0, 1024 px (same: the card
example uses the pipeline defaults; the M1 smoke run used 30 steps, changed so that all four
models follow one rule). SD3.5 Medium: 40 steps, cfg 4.5, 1024 px, T5 max 256 tokens (card).
FLUX.1-schnell: 4 steps, guidance 0, 1024 px, max 256 tokens (card). Precision: fp16 for the
UNet models (their released fp16 weights), bf16 for the two DiTs (as their cards load them).
Weights are pinned to the commit in the local HF cache (`revision` in `configs/gen/`).

**D-007 (2026-10-19) — Batch size per model = the smoke-run values: SD1.5 4, SDXL 2, SD3.5M 2,
FLUX 1.** Peak memory in the smoke run was 4.3 / 15.8 / 20.7 / 36.3 GB on 94 GB H100s, so
larger batches would fit, but these are validated, and the whole v1 run costs ~6 GPU-hours,
so throughput doesn't matter. Fixed for the project (it enters `config_hash` and can shift
numerics slightly); per-image times are reported with their batch size (D-004).

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
*Observation (smoke run, 2026-10-18):* scores take repeated values (0.500 three times; 0.321,
0.223, 0.893, 0.755 …). They are sigmoids of a Yes−No logit gap that is a multiple of 1/8:
Qwen computes logits in bf16, whose step at |logit| ≈ 16–32 is 0.125. Expected, and t2v_metrics
behaves the same way; but it creates exact ties (0.5) that pairwise comparisons count as
half-agreements. Option for later, recorded here before any result depends on it: compute the
final projection in fp32 (would change `metric_version`).

**D-015 (2026-10-18) — Human correctness labels for template prompts: one label per image,
`y` / `n` / `?`, by these rules.** (1) Counting: `y` only if exactly the requested number can
be counted; partly hidden but recognisable objects count; `?` if they can't be counted.
(2) Spatial: both objects must be present (otherwise `n`); left/right from the viewer;
"above" means higher in the image, resting on top included (the same convention as
box-centre checks with a detector). (3) Text: every requested word spelled correctly, none
missing or repeated; case and punctuation ignored. Agreement with a metric is measured within
prompts: over pairs (one `y` image, one `n` image) of the same prompt, the share where the
metric scores the `y` image higher (ties = 1/2), CI by resampling prompts. Within-prompt pairs
cancel out prompt difficulty, which shifts every metric's absolute level. Labels live in
`data/labels/<prompt set>_labels.csv` (tracked). **Smoke labels (80 images):** one annotator
(the project author), from the contact sheets, 2026-10-18. Afterwards 5 blank cells were filled
from the annotator's own notes, and 4 cells were changed by the annotator after a consistency
check against the rules (marked `[复核后修改]` in the note). Smoke numbers stay a pipeline
check (D-010); the real meta-evaluation uses GenAI-Bench's multi-rater human scores (guide §9).

**D-017 (2026-10-19) — OCR checker for text prompts = EasyOCR (CRAFT detector + `english_g2`
recognizer) plus a semi-global edit distance; an image passes only on an exact match.**
`genprobe/metrics/ocr.py`. EasyOCR reads every text box; boxes are grouped into lines (top to
bottom, each left to right) and joined. The requested text is aligned against its
best-matching stretch of that string, so other text in the picture costs nothing.
`value = 1 − distance / length` after removing case, punctuation and spaces: the human rule
(D-015) ignores case and punctuation, and OCR often merges or splits words ("THANKYOU"), which
says nothing about spelling. The verdict threshold is `value = 1` (exact), fixed before any
v1 output; it may be re-tuned on dev only. `word_recall` and every OCR box are kept in `extra`.
EasyOCR rather than PaddleOCR: plain pip install, small weights, English recognizer. It runs in
its own `aux` env so the eval env (in use by queued v1 scoring jobs) is untouched; weights are
fetched on coe-hpc1 from the URLs and md5 sums in the installed EasyOCR's own config, and
EasyOCR re-checks the md5 and never downloads inside a job. Quantization is off, so CPU and GPU
runs use the same fp32 recognizer. Known blind spots, accepted: a phrase written twice still
passes (people mark it wrong), and stylised lettering that people can read but OCR can't
counts as wrong.
*Smoke check (2026-10-19, pipeline check only, D-010):* on the 20 smoke text images with a
y/n label, the exact-match verdict agreed with the label on 17 (0.85, CI 0.71–0.95). It passed
none of the 11 images labelled wrong and failed 3 of the 9 labelled correct (values 0.79–0.94:
OCR misread a letter or two). Within-prompt pairwise agreement 1.00 (13 pairs). The threshold is
not changed on this (smoke is not dev); the dev check in M2 decides whether near-misses count.
Of the 3 misses (OCR text in `ocr_read`), two are OCR misreads of text people read without
trouble (a stylised "A" in WAIT read as "WINIT"; small serif lettering on a facade); one
("DAILLY" for DAILY) can't be settled from the contact-sheet thumbnail.
*Dev check (2026-10-19, 50 blind-labelled text images, D-024):* at the pre-set threshold
(exact, value = 1) the verdict agrees with the label on 39 of 50 (0.78, CI 0.69–0.86): it passed
none of the 27 images labelled wrong and failed 11 of the 23 labelled correct, again OCR misreads
("RROGRESS", "OVT", "WIIF"). Lower thresholds on the same 50 images: 0.90 → 0.82, 0.85 or 0.80 →
0.86 (3 wrong images pass, 4 correct ones fail), 0.75 → 0.88. A one-letter OCR misread and a
one-letter misspelling get the same value (e.g. 0.929 for "RROGRESS", labelled y, and
"PPROGRESS", labelled n), so no threshold separates them fully.
*Revised 2026-10-19: the OCR verdict threshold is now 0.85* (project author's decision, on the
dev numbers above). At 1.0 the checker misses about half of the correctly written images, so the
share of "text right" would be underestimated by about a factor of two; 0.85 is the first value
of the 0.80–0.85 plateau (0.86 on dev, 3 false passes and 4 false fails, roughly balanced).
0.75 scored 0.88 but with more false passes, and 2 images of 50 are within noise. The dev
accuracy at the chosen threshold is measured on the same 50 images used to choose it, so it is
optimistic; REPORT.md says so. The threshold only changes the verdicts (the "text right" rates and
the text-length figure); the OCR value and every model comparison made on it are unchanged.

**D-018 (2026-10-19) — Detector checker for counting and spatial prompts = Grounding DINO tiny
(`IDEA-Research/grounding-dino-tiny`, main at a2bb814 when chosen), one text query per object
name (`"<name>."`), box score ≥ 0.35, text score ≥ 0.25.** `genprobe/metrics/detect.py`.
Counting: duplicates merged by NMS (IoU 0.5); a box that covers ≥ 90 % of the area of two or
more other boxes is dropped as the detector outlining the whole group; pass iff the number of
boxes equals the requested count. Spatial: the best box for each object, required to be two
different boxes (IoU < 0.8); pass iff the box centres satisfy the relation by more than 5 % of
the image width (left/right) or height (above/below), from the viewer, as in the human rule
(D-015); a missing object fails. Thresholds are the Grounding DINO authors' demo defaults
(box 0.35, text 0.25); the margin and group rule were set here, before any v1 checker output.
All of them may be re-tuned on dev only, and the values are part of `metric_version`.
Grounding DINO because it is open-vocabulary (no mapping to COCO class ids) and already
supported by the eval env's transformers 4.49, so no new install. Known weaknesses, accepted:
open-vocabulary detectors miscount small clustered objects and can label a similar object as
the queried one (e.g. a cat as "dog"); the checker is itself measured against human labels
before its numbers are used (smoke labels now, dev later).
*Smoke check (2026-10-19, pipeline check only, D-010):* on the 49 smoke counting/spatial
images with a y/n label, the verdict agreed on 44 (0.90, CI 0.80–0.98). It passed 3 images
labelled wrong, all spatial (annotator notes: "both are dogs" for cat-left-of-dog; the vase is
left of the book; the cup is in front of the laptop, not beside it) and failed 2 labelled correct
(both "a bicycle to the left of a car"). Counting verdicts matched every label. Pairwise
agreement 0.85 (CI 0.70–0.97, 20 pairs). No threshold changed. Looking at the five images
with their box centres (`detect_detail`): one is a detector error (sd35m, bicycle/car: it took a
small car in the background on the left instead of the large car on the right, because the rule
uses the highest-scoring box); three are borderline images (a cat/dog hybrid; a top-down view
with the cup up and to the right of the laptop; a vase standing behind the right half of an open
book); in one (flux, bicycle/car) the bicycle stands to the right of the car in the image while
the label says `y`. Labels are not edited after metric outputs have been seen (D-015), so it
stays and counts against the checker. Choosing the largest box, or accepting any qualifying
pair, is a candidate rule to compare on dev, not on smoke.
*Dev check (2026-10-19, 50 blind-labelled counting/spatial images, D-024):* verdict agrees on 45
of 50 (0.90, CI 0.81–0.97): 4 images labelled wrong pass, 1 labelled correct fails. Counting: the
detector's count equals the annotator's on 17 of 20 images where both counted, and is higher in
all 3 others (never lower). Spatial disagreements are judgement calls noted by the annotator: a
short seat not counted as a "bench", a dog standing on a book (above, not right of it), a cat
left of the horse's head but right of its body centre; plus one bottle 0.07 above a teddy bear
by box centres. No
detector threshold is changed on this evidence.

**D-019 (2026-10-19) — The checkers run on CPU nodes (cpuqs, 8 cores) through
`slurm/score_cpu.sbatch`; CLIPScore and VQAScore stay on the H100s.** Both checker models are
small, and nsfqs queue waits have been hours (D-009); cpuqs jobs have started within a minute.
CPU speed is measured on the smoke images first: if one v1 shard would need more than about
two hours on CPU, they move to the H100s (same code, `--device cuda`). The device is recorded
in each run's `metric_info`, not in `metric_version`: CPU and GPU give near-identical numbers,
and mixing them within one version is accepted.

**D-021 (2026-10-19) — Preference scores: PickScore (`yuvalkirstain/PickScore_v1`) and
HPSv2.1 (`xswu/HPSv2`, `HPS_v2.1_compressed.pt`), both fp32, both in the eval env.** The guide
(§7.7) asks for PickScore plus one of ImageReward / HPS. HPSv2.1 rather than ImageReward because
it needs no new package: it is an OpenCLIP ViT-H/14, so it runs on the open_clip already used
for CLIPScore, with the checkpoint loaded `weights_only=True` and every weight required; the
`hpsv2` package (it bundles its own open_clip copy and downloads at run time) is not used.
Values follow each project's definition: PickScore = `logit_scale.exp() × cosine` (README),
HPS = cosine (what `hpsv2.score` returns). Known difference: hpsv2 scores under CUDA autocast
(fp16), we use fp32. PickScore's processor files come from its own repo; they match the LAION
CLIP-H processor the README loads (224 px bicubic + centre crop, OpenAI mean/std, CLIP BPE).
These scores measure what people like to look at, not prompt following, so they are reported
beside the alignment metrics and never averaged with them; cases with high preference and low
VQAScore are worth showing (guide §7.7).

**D-022 (2026-10-19) — Safety is measured as false positives on benign images only (guide §7.8):
two filters, the SD safety checker (`CompVis/stable-diffusion-safety-checker` via diffusers'
`StableDiffusionSafetyChecker`) and a ViT NSFW classifier (`Falconsai/nsfw_image_detection`,
Apache-2.0).** No harmful content is generated on the school cluster. Every prompt in our sets is
benign, so every flag counts as a false positive; value = 1 when flagged, and a model's mean is
its false-positive rate. Reported per model and overall, with how often the two filters agree.
SD1.5 runs without its checker (D-005), so `nsfw_sd` also says how many of its images the
default pipeline would have blacked out. The SD checker's decision is taken from the checker
itself; the plugin also recomputes it from the concept scores to store the margin.

**D-023 (2026-10-19) — Every metric passes the §7.9 sanity checks before its numbers are used,
on 200 random dev-split images (seed 0), with `genprobe/sanity.py`.** (1) Mismatch: own prompt vs
a random other prompt of the same kind. (2) Counterfactual, template prompts only: count ±2,
left ↔ right / above ↔ below, or the next phrase of the same length tier. (3) Repeatability:
50 images scored twice in the job, and all 200 against the scores stored by the scoring run.
Results go to `results/sanity/`. No pass mark is set here: the rates are reported as measured,
and a metric that prefers the mismatched or counterfactual prompt on a sizeable share is flagged
in REPORT.md rather than silently dropped.
*Revised 2026-10-19, after the first run and before any threshold depended on it:* on random
images the counterfactual check is confounded. Where the generator drew the count or the
position wrong (common), the changed prompt can fit the image better, so even a perfect metric
scores below 1; the first run gave ~0.4–0.5 on left/right/above/below for every prompt metric.
(CLIPScore and VQAScore had identical counts there, 13 higher / 14 lower of 27; the per-image
scores show they disagreed on 12 of those 27 images, so it was a coincidence, not a bug.) The
clean version now runs on the dev images a person labelled correct (D-024), where the original
prompt is right and the changed one wrong (`--part labelled`); the random-image numbers stay in
the report, marked as confounded.

**D-024 (2026-10-19) — The checkers are measured on dev against blind human labels before any
threshold is tuned: 25 counting, 25 spatial and 50 text images, seed 0, spread evenly over the
four models, labelled from numbered sheets that show no scores and no model names
(`genprobe/label_set.py`).** The guide (§7.5, §7.6) asks for ~50 hand-checked dev images per
checker and the checker's own accuracy on them. For counting images the annotator also writes the
count they see (`true_count`), so the detector's count error can be measured, not just its
verdict. Same rules as D-015; labels go to `data/labels/prompts_v1_labels.csv` and are never
overwritten. The labelling set was drawn and sheeted before anyone looked at dev checker outputs.
*Labelled 2026-10-19* by the project author, from the four blind sheets, all 100 images:
counting 8 y / 17 n (a count written for 20 of 25), spatial 9 y / 16 n, text 23 y / 27 n, no `?`.

**D-025 (2026-10-19) — A few generated images go into git after all: example panels in
`results/examples/` (256 px JPEG thumbnails, about 40 images), nothing else.** Project author's
decision. The report and the README need to show what "right" and "wrong" look like (guide
§8.3); everything else under `data/images/` stays out of git as before. The images are picked by
rule, not by hand (`genprobe/examples.py`): at random with seed 0 from the test split among images
that clearly pass or clearly fail, the checker and VQAScore agreeing (detector pass and VQAScore ≥
0.9 vs fail and ≤ 0.1; OCR ≥ 0.85 and VQAScore ≥ 0.9 vs OCR < 0.5 and ≤ 0.1; VQAScore ≥ 0.95 vs
≤ 0.05 for GenAI-Bench), spread over the four models, plus "pretty but wrong"
images (top-10 % PickScore with VQAScore ≤ 0.05). Every image is listed with its sample_id in
`results/examples/index.md`. Safety-filter flags are not illustrated, and prompts about children or
students are left out of the panels. All four generators'
licences allow sharing outputs.

**D-020 (2026-10-19) — How results are tabulated and drawn (`genprobe/figures.py`, guide §8).**
Every mean is over prompts (each prompt first averaged over its seeds), with 95 % CIs from 2000
bootstrap resamples of prompts. Models are compared on the same prompts: the paired mean
difference with its CI, plus the share of prompts where A beats B; "A is better" only when
the CI excludes 0. Next to each overall mean the leaderboard gives a category-weighted mean
(every category counts once) to expose Simpson-type reversals. Metrics are never averaged into
one score. The category heatmap has one column per template category and one per GenAI-Bench
skill (prompts carry several skills, so those columns overlap); columns with fewer than 30
prompts are flagged. The quality–cost figure uses the median generation seconds per image
(warmup excluded, all on H100 NVL) on a log axis and marks the Pareto front. Figure titles
are computed from the numbers they show. The same model keeps the same colour (validated
categorical palette) and marker in every figure.

## Prompt set

**D-016 (2026-10-19) — Prompt set v1 = 400 GenAI-Bench prompts + 200 template prompts, built by
`genprobe/build_prompts.py` (seed 20261019) into `data/prompts/prompts_v1.jsonl`, frozen
before any v1 image exists.** Source: GenAI-Bench-1600 at revision `e51e8a8` (pinned in
`scripts/fetch_genai_bench.sh`). What the data actually contains (counted, not taken from the
paper): 1,600 prompts; tiers basic 722 / advanced 871 / none 7 (seven single-object prompts
without skills, e.g. "A Bluetooth speaker.", left out); skills attribute 553, scene 360,
spatial relation 386, action relation 467, part relation 195 (all basic), counting 340,
comparison 162, differentiation 296, negation 347, universal 147 (all advanced); up to 5
skills per prompt. Sampling: rarest skill first until every skill has ≥ 60 prompts, then
uniform over the rest (result: 75–126 per skill; 157 basic, 243 advanced). Template prompts
(COCO objects, checkable by a detector/OCR): counting 70 = 10 objects × counts 1–7 (number
words); spatial 70 = 18/18/17/17 random ordered pairs of 16 objects for left of / right of /
above / below; text 60 = 20 phrases each of 1–2, 3–4 and 5–8 words, none reused from the
smoke set. Two or three sentence templates per category, chosen at random, so results don't
hinge on one wording. Split: 20 % dev / 80 % test inside each stratum (GenAI-Bench tier;
template category × count / relation / length tier). Dev is for tuning thresholds, rubrics
and agent rules; every reported number comes from test. `n_clip_tokens` is filled when the
builder runs where open_clip is installed (it was built on an HPC3 CPU node, eval env): the
longest prompt has 40 CLIP tokens, so nothing is truncated by SD1.5/SDXL's 77-token encoders
or CLIPScore. GenAI-Bench data note for module B:
two of its 28,800 ratings are out of the 1–5 range (`00231` DALLE_3 has 41, `00252`
DeepFloyd has 33); they will be dropped, not guessed.

**D-010 (2026-10-16) — `data/prompts/smoke.jsonl` (20 template prompts: 7 counting,
7 spatial, 6 text) is only for pipeline testing in M1.** It is not part of prompt set v1
and no reported number comes from it.

## Pre-registered hypotheses (module B)

Drafted from the guide (§9.1) before any meta-evaluation was run; **finalized 2026-10-19**
(project author's approval) before any metric was scored on GenAI-Bench images. Finalizing added
the tests below; the three directions are the draft's, unchanged. Disclosure: at finalization we
had seen within-prompt agreement with one annotator's labels on 100 of our own dev images (D-024:
VQAScore 0.92, PickScore 0.94, HPSv2.1 0.86, CLIPScore 0.65; 78 pairs, 60 of them text). That
does not point the same way as H2, and H2 was left as drafted. Anything not listed here is
reported as exploratory.

**Common method.** Data: GenAI-Bench-1600 (`BaiqiL/GenAI-Bench-1600` at e51e8a8), images from 6
models with 3 human ratings (1–5) each. The two out-of-range ratings (`00231` DALLE_3: 41,
`00252` DeepFloyd: 33) are dropped and those images use the mean of their two valid ratings.
Within each prompt, every pair of two models' images whose mean human ratings differ is a pair
(pairs with equal means are excluded and their share reported). A metric agrees on a pair when it
orders the two images as the mean rating does; equal metric scores count 1/2. Agreement = share of
agreeing pairs. Comparisons between metrics use the same pairs; 95 % CIs from 2,000 bootstrap
resamples of prompts (seed 0). A hypothesis is **supported** if its CI lies entirely on the
predicted side, **contradicted** if entirely on the other side, **inconclusive** otherwise.
Metric versions are those of D-013, D-014 and D-021; nothing about them is tuned on GenAI-Bench.

- **H1** — On GenAI-Bench prompts tagged `counting`, VQAScore's within-prompt pairwise agreement
  with the mean human rating is higher than CLIPScore's. Statistic: agreement(VQAScore) −
  agreement(CLIPScore) over the same pairs, 95 % CI.
- **H2** — Over all GenAI-Bench prompts, each preference model has a lower agreement than
  VQAScore: PickScore and HPSv2.1 are tested separately, VQAScore − model, with 97.5 % CIs
  (Bonferroni for two comparisons); H2 holds for a model when its CI lies above 0.
- **H3** — The VLM judge, comparing two images for the same prompt, has a position bias. 1,000
  pairs drawn at random (seed 0) from the pairs above, restricted to prompts outside prompt set
  v1's dev split (those 80 GenAI-Bench prompts are where the judge's rubric is iterated), each
  shown in both orders at temperature 0. Statistic: among verdicts that pick an image (not "same"),
  the share that pick the image shown first; H3 holds if its 95 % CI (prompt bootstrap) excludes
  0.5. The share of pairs whose two verdicts agree is reported alongside, descriptively. Planned
  judge: Qwen3-VL-8B-Instruct (not VQAScore's Qwen2.5-VL-7B); if it cannot run on HPC3, the
  substitute is recorded here as a revision before any judge output exists.

**Ceiling.** Human–human agreement on the same pairs, computed the same way: each rater's score
against the mean of the other two (leave-one-rater-out), averaged over the three raters; plus
Krippendorff's α (ordinal) over the three ratings. Metric numbers are read against this ceiling.

## Meta-evaluation (module B)

**D-026 (2026-10-19) — How the pre-registered method is implemented (`genprobe/meta_eval.py`),
written and tested on synthetic data before any metric was scored on GenAI-Bench.** Details the
hypotheses section leaves open, fixed now: (1) "all GenAI-Bench prompts" means all 1,600,
including the 80 that are in prompt set v1's dev split (they are reserved for the VLM judge's
rubric, not for the metrics, which are not tuned here); the seven prompts with no skill tag are
in "all" only. (2) Two metrics are compared on the pairs where both scored both images; the
analysis refuses to run until every metric has scored all 9,600 images, so the results are
computed once, on the full set. (3) Ceiling: "each rater" is each rating position (we do not
assume position 1 is the same person across images); on the metric pairs, a
position's rating is compared with the mean of the other two, pairs where those two tie are
skipped, equal ratings count 1/2, and images left with two valid ratings (the two dropped
out-of-range ratings) are not used for the ceiling. Its CI resamples prompts. (4) Krippendorff's
α uses the ordinal distance on all valid ratings (checked against the worked example in
Krippendorff 2011: 0.815). (5) Kendall's τ-b within prompts (6 images), averaged, is
exploratory. (6) The smoke run on 20 GenAI-Bench prompts (120 images) is checked for counts,
errors, value ranges and timing only (`genprobe/score_check.py`); agreement with people is not
computed on it.

**D-027 (2026-10-19) — The VLM judge is Qwen3-VL-8B-Instruct (as pre-registered for H3), run by
vLLM in a fourth env, `judge`.** Weights pinned to the snapshot downloaded on coe-hpc1
(`download_models.sh judge`, ~17.5 GB, not gated); vLLM pins its own torch and transformers, so it
does not share the eval env (guide §9.4). Before any judge code is written, `slurm/judge_probe.sbatch`
checks on an H100 that vLLM loads the model offline and handles one image, two images in one
prompt, JSON-constrained output and first-token logprobs, and measures its speed; judge.py is
then written against the vLLM version actually installed. Qwen3-VL and VQAScore's Qwen2.5-VL come
from the same family. That is acceptable for module B, where the judge is itself checked against
people, but module C's final grader must differ from the metric the agent optimizes (guide §10),
so before module C a judge from another model family will be added and meta-evaluated the same
way.

**D-028 (2026-10-19) — How the VLM judge is run and checked (`genprobe/judge.py`,
`genprobe/judge_eval.py`; guide §9.4).** Two question types, both answered as JSON under vLLM's
structured output (the schema fixes the keys, so answers parse; truncation and unreadable answers
are still counted): absolute (`{"issues": [...], "score": 1-5}`; issues come first so the score
follows the check) and pairwise (`{"reason": ..., "better": "1" | "2" | "same"}`). Besides the
integer score, the probabilities of the five digits at the score position give an expected score
(continuous, fewer ties); which of the two is the judge's reported absolute score is decided on dev
and recorded here before the test round. Images are shrunk to fit 1024 × 1024 (never enlarged) and
sent as PNG; greedy decoding (temperature 0) gives the results. Checks: position bias (each pair in
both orders; the H3 statistic as pre-registered), consistency (3 samples at temperature 0.7, seed 0),
blind test (grey image; same model's image for another prompt), score distribution. Pairwise
agreement with people counts a pick only when both orders agree; inconsistent or "same" verdicts
count 1/2 (guide §9.4). **Rounds:** dev = the 80 GenAI-Bench prompts of prompt set v1's dev split,
all images and all pairs with different mean ratings; the rubric (prompt text, scale, schema) is
changed only after dev rounds, each with a new tag, and the dev summaries list the clearest
disagreements with people to learn from. Test = the other 1,520 prompts: all 9,120 images
(absolute), the 1,000 pre-registered H3 pairs (both orders), blind test and consistency on 500
images (seed 0) and on those pairs. The test round runs once, with the rubric frozen;
`genprobe.judge` refuses to overwrite a test tag.

*D-028 revision 1 (2026-10-19, after dev round `dev_r1`, rubric r1 = template `51cfbc75`).* On the
80 dev prompts the judge agreed with people on 0.655 of pairs by integer score, 0.705 by expected
score (VQAScore 0.737 on the same 939 pairs), and 0.660 pairwise, with a consistent pick in only
53 % of pairs and 0.567 of image-picking verdicts going to the image shown first. Scores were
bimodal (4 used for 3 % of images). Worst: grey images scored higher than the real ones (mean 3.17
vs 2.95), with issues like "cannot verify ..."; another prompt's image scored 1.00. Rubric r2
(`results/judge/dev_r1/summary.md` lists the failures it responds to): what cannot be seen counts as
unmet; quantities too large to count are judged by eye; only unmet requirements are listed; the
scale is anchored on how much of the prompt is met; in pairs, each image's unmet requirements are
listed before the verdict. Because r2 was written after seeing the grey-image failure on dev, the
dev blind test is no longer a clean check of r2; the test round's blind test is.

*D-028 revision 2 (2026-10-19, after `dev_r2`, rubric r2 = template `1c6bef16`).* Absolute scores
barely moved (0.659 by integer score, 0.709 by expected score), pairwise got worse (0.559; a
consistent pick in 29 % of pairs; 3 % of answers cut off at the token limit), and grey images still
scored as high as real ones (3.01 vs 2.88; 241 of 480 scored 3 or more, many 5 with no issues
listed). Decisions, made on dev before any test output: (1) the judge's reported absolute score is
the **expected score** (better than the integer score in both rounds: 0.705 vs 0.655, 0.709 vs
0.659). (2) Rubric r3: the absolute question asks first for a description of what the image shows,
then the check; the pairwise question returns to r1 unchanged (its rerun also shows whether
results repeat across jobs). (3) After `dev_r3` the rubric is frozen: absolute = the round with the
highest dev agreement by expected score, pairwise = the one with the higher dev pairwise agreement
(r1 unless r3's rerun differs). Position bias plays no part in that choice, since H3 is about it.

*D-028 revision 3 (2026-10-19, after `dev_r3`, rubric r3 = template `b9e98ed8`): the rubric is
frozen.* Dev results by expected score, absolute: r1 0.705, r2 0.709 [0.665, 0.753], r3 0.680
[0.632, 0.731] (VQAScore 0.737 on the same pairs). Blind test with grey images: r2 scored them like
real ones (3.01 vs 2.88), r3 gives them 1.00 vs 3.00 for real images (the real image higher in
100 % of 478 by expected score); another prompt's image scores ~1.0 in all three rounds. Pairwise,
r1's prompt rerun unchanged in r3: 0.657 vs 0.660 in `dev_r1`, position bias 0.569 vs 0.567, so
greedy answers do not repeat exactly across jobs (vLLM batches differ), but the summaries move by
less than 0.01. **Frozen:** absolute = r3 with the expected score, pairwise = r1; `judge.py` refuses
a test round with any other template (`FROZEN_TEMPLATE`). **This departs from the rule written in
revision 2**, which would have picked r2 for its 0.029 higher dev agreement (the two CIs overlap
widely). Project author's decision, on the recommendation that a judge which gives a blank grey
image full marks cannot later grade an agent's images (module C), where degenerate images are a
real risk. Pairwise is r1 under either rule, so H3 is unaffected. Because r2 and r3 were written
after seeing the grey-image failure on dev, only the test round's blind test is a clean check.

**D-029 (2026-10-20) — Own human evaluation (guide §9.5) with one annotator, the project author.**
Project author's decision: no classmates are asked. What is lost: agreement between annotators
(Krippendorff's α) on our images; the multi-rater view comes from GenAI-Bench's three ratings (§3,
α = 0.715). What stands in for it: 40 hidden repeats (main pairs shown again ≥ ~50 items later,
sides re-drawn), reported as same-answer share and Cohen's κ. Kept from the guide: blind pairs
(same prompt and seed, two models; no model names, scores or item types shown; sides and order
random; all images 768 × 768), 50 pairs per stratum (counting, spatial, text, GenAI-Bench) from the
test split, 20 attention checks (the same model's image for a prompt of another stratum; pass mark
18 of 20, fixed now), 20 practice pairs from dev that may still change the protocol, and four
answers (A, B, same, neither). Images flagged by either safety filter are excluded. The labelling
page is a static HTML file that works offline (no install, nothing leaves the laptop). Protocol:
`docs/human_eval_protocol.md`; code: `genprobe/human_eval.py`. The author bias this adds goes into
the report's limitations.

*D-029 note (2026-10-22), written and committed before the answer key was copied off HPC3.*
(1) The 20 practice items raised seven unclear cases; seven rules were added under "Changes after
the practice round" in `docs/human_eval_protocol.md` before any main item was labelled. (2) The
practice round and a first pass over the main items were run as an AI pre-evaluation; those AI
labels were not kept and are not used, and the author states he did not see the AI's main-item
answers before labelling them. (3) `data/labels/human_eval_labels.csv` holds the author's labels
for the 260 main items (its practice rows are the AI's); they were typed in, not exported from the
labelling page, so there are no answer times. SHA-1 of the file:
`642425a9faab39071534e25e006805b7077575eb`. The labels are not changed after the analysis.
(4) Points 2 and 3 go into the report's limitations.

## Module C (generation-debugging agent)

**D-030 (2026-10-22) — Final judge = Gemma 4 31B-it; base model for the agent = SDXL.** Project
author's choices. The agent optimizes VQAScore (Qwen2.5-VL) and the detector / OCR checks, so the
judge that grades it must come from another model family (guide §10.4, D-027). Gemma 4 31B-it
(Google, Apache-2.0, ~62.6 GB in bf16, fits one H100 NVL) is the strongest non-Qwen option that
vLLM runs offline. It is run with the frozen rubric of D-028 (absolute r3, pairwise r1) **without
any tuning of its own**, with 560 vision tokens per image (its default is 280) and thinking off.
Before it grades anything it goes through the same checks as the Qwen judge: a probe job, one dev
round (pipeline and blind test only; nothing is tuned on it), then one test round on GenAI-Bench
for agreement with people and the bias checks. SDXL is the agent's base model because it has the
most room to improve among the four (§2: VQAScore 0.505, detector 0.438, OCR 0.664) and its cfg
and step count can be changed; FLUX.1 schnell is already strong and has no cfg.

