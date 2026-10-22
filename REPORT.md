# GenProbe report

_Status: M2 (benchmark on prompt set v1) and M3 (metric meta-evaluation: H1–H3, the VLM judge
and a human check on our own images) done; module C (the generation-debugging agent) next. Every number
here is copied from a file in `results/`, named next to it; nothing is reported before it
exists there. Choices behind the numbers are in `docs/decisions.md` (D-xxx)._

## 1. Setup

**Models** (D-006, D-007): Stable Diffusion 1.5, SDXL base 1.0, Stable Diffusion 3.5 Medium and
FLUX.1 [schnell], each at the settings of its own model card (SD1.5 50 steps / cfg 7.5 / 512 px;
SDXL 50 / 5.0 / 1024 px; SD3.5M 40 / 4.5 / 1024 px; FLUX 4 steps, no guidance, 1024 px), weights
pinned to a commit, all on one H100 NVL 94 GB card type.

**Prompts** (D-016): prompt set v1, 600 prompts = 400 from GenAI-Bench-1600 (sampled so every
skill has ≥ 60 prompts) + 200 template prompts with known answers (70 counting, 70 spatial,
60 text rendering). Fixed split, 20 % dev / 80 % test inside each stratum: 120 dev prompts for
choosing thresholds, 480 test prompts for every reported number. 4 seeds per prompt: 9,600
images, all generated (`results/prompts_v1_test/summary.md`, generation coverage 1920/1920
per model on test).

**Metrics.** Prompt following: CLIPScore (D-013), VQAScore with Qwen2.5-VL-7B (D-014), a
Grounding DINO detector check for counting / spatial prompts (D-018) and an OCR check for text
prompts (D-017). Preference: PickScore and HPSv2.1 (D-021). Safety: false-positive rate of two
filters on benign images (D-022). Cost: median generation seconds per image.

**Checking the checkers.** 100 dev images, labelled blind (no scores, no model names) by the
author (D-024), from `results/prompts_v1_dev/summary.md`:

| Checker | Images | Agrees with label | Notes |
| --- | --- | --- | --- |
| Detector (counting + spatial) | 50 | 0.90 [0.81, 0.97] | counts equal the annotator's on 85 % of 20 counted images, the 3 others over-count |
| OCR (text), threshold 0.85 | 50 | 0.86 [0.82, 0.91] | threshold chosen on these same 50 images (optimistic); at 1.0 it was 0.78 (D-017) |

**Sanity checks** (guide §7.9, D-023; `results/sanity/summary.md`, dev images):

- Mismatched prompt (own vs another prompt, 200 images): CLIPScore 200/200 prefer their own,
  VQAScore 198/200, PickScore 200/200, HPSv2.1 199/200.
- One-word counterfactuals on images labelled correct (count ±2, left ↔ right / above ↔ below,
  other text; 40 images): VQAScore prefers the true prompt on 40/40. CLIPScore, PickScore and
  HPSv2.1 on 34/40 each, with most misses on the 9 spatial flips (4/9, 3/9, 3/9): close to
  chance on left/right and above/below. Detector 16 of 17, OCR 23/23.
- Repeatability: rescoring gives identical values (max |Δ| ≤ 8 × 10⁻⁶).

## 2. Leaderboard and quality–cost frontier

Test split, 480 prompts × 4 seeds, means over prompts with 95 % bootstrap CIs over prompts
(`results/prompts_v1_test/summary.md`, `figures.md`):

| Model | VQAScore | Detector pass | OCR | PickScore | HPSv2.1 | s / image |
| --- | --- | --- | --- | --- | --- | --- |
| SD3.5 Medium | 0.680 [0.649, 0.709] | 0.641 | 0.855 | 22.56 | 0.286 | 4.60 |
| FLUX.1 schnell | 0.670 [0.640, 0.702] | 0.583 | 0.867 | 22.65 | 0.296 | 1.13 |
| SDXL | 0.505 [0.474, 0.535] | 0.438 | 0.664 | 22.38 | 0.278 | 3.05 |
| SD 1.5 | 0.365 [0.337, 0.391] | 0.359 | 0.425 | 21.15 | 0.249 | 0.53 |

- **SD3.5 Medium and FLUX.1 schnell tie on prompt following**: paired VQAScore difference
  +0.010 [−0.015, +0.033] over the same prompts; detector +0.058 [−0.009, +0.123]; OCR −0.013
  [−0.056, +0.030]. FLUX gets there 4.1× faster (`fig_frontier.png`). Every other pairwise
  VQAScore difference excludes 0.
- **Preference scores split them**: FLUX is ahead of SD3.5M on PickScore (−0.098 [−0.155,
  −0.043] for SD3.5M − FLUX) and HPSv2.1 (−0.010 [−0.012, −0.008]), while VQAScore calls them
  even: what people like to look at and whether the prompt was followed are different questions.
- **CLIPScore barely separates the models** (0.270–0.283) and ranks images very differently
  from VQAScore (Spearman ρ = 0.27 over 7,680 images).
- **Hardest skills** (`fig_category_heatmap.png`): negation (VQAScore 0.21–0.29 for all four),
  then comparison, differentiation and counting.
- **Text gets harder with length** (`fig_text_length.png`, `text_length.csv`): the share of
  images with the text right falls from 48 % (1–2 words) to 24 % (5–8 words) averaged over
  models — FLUX 73 % → 52 %, SD3.5M 69 % → 42 %, SDXL 47 % → 3 %, SD1.5 ≤ 5 % throughout.
- **Safety filters on benign images**: the SD safety checker flags 0.47 % of 7,680 test images,
  the ViT NSFW classifier 0.20 %, and no image is flagged by both (`figures.md`).
- **Cost**: about 7.7 GPU-hours to generate the 9,600 images (one shard re-run included) and
  about 2 for CLIPScore + VQAScore (`docs/hpc_notes.md`).

Example images, picked by rule: `results/examples/index.md`.

## 3. Metric meta-evaluation

**Data.** GenAI-Bench-1600 (D-016): 1,600 prompts, each rendered by six models other than §2's (DALL-E 3,
DeepFloyd IF XL, Midjourney 6, SD 2.1, SDXL base, SDXL Turbo), and each image rated 1–5 by three
people for how well it matches the prompt. 9,600 images, all scored by the four metrics with the
same versions as in §2; nothing was tuned on GenAI-Bench.

**Method** (pre-registered in `docs/decisions.md`, implementation D-026). Within each prompt,
every two images whose mean human ratings differ form a pair; a metric agrees on a pair when it
orders the two images as the mean rating does. 18,098 pairs from 1,558 prompts; 5,902 of 24,000
pairs (24.6 %) have equal means and are left out, and in 42 prompts all six images got the same
mean. 95 % CIs from 2,000 bootstrap resamples of prompts. The analysis code was committed (at
`8b0bdc9`) before any GenAI-Bench score existed and refuses to run on partial scores, so the
results below were computed once.

**People among themselves.** Krippendorff's α (ordinal) over the three ratings: 0.715. Each
rating against the mean of the other two agrees on 81 % of the pairs (equal ratings count 1/2):
the ceiling the metrics are read against.

**Pre-registered tests** (`results/meta_eval/hypotheses.md`):

| Test | Prompts | Agreement with people | Difference | CI | Verdict |
| --- | --- | --- | --- | --- | --- |
| H1: VQAScore > CLIPScore | counting (334) | 0.748 vs 0.577 | +0.171 | 95 %: [0.142, 0.199] | supported |
| H2: VQAScore > PickScore | all (1,558) | 0.733 vs 0.670 | +0.062 | 97.5 %: [0.046, 0.077] | supported |
| H2: VQAScore > HPSv2.1 | all (1,558) | 0.733 vs 0.665 | +0.067 | 97.5 %: [0.051, 0.083] | supported |
| H3: the VLM judge has a position bias | 1,000 pairs (639 prompts) | – | 0.581 pick the first image (0.5 = none) | 95 %: [0.562, 0.601] | supported |

H3 comes from `results/judge/test_r1/summary.md`: of the 1,565 pairwise verdicts that picked an
image, the share that picked the one shown first.

**By skill** (exploratory; `fig_meta_eval_heatmap.png`, `agreement_by_tag.csv`):

- VQAScore agrees with people on 71–76 % of pairs in every skill; the human ceiling there is
  78–83 %.
- CLIPScore stays at 54–62 % everywhere, 54 % on negation and on universal prompts (chance is 50 %).
- The two preference models are close to VQAScore on basic prompts (PickScore 0.72, HPSv2.1 0.72,
  VQAScore 0.74) and fall behind on advanced ones (0.63, 0.62 vs 0.73), most on negation (0.57,
  0.55 vs 0.71).
- Within-prompt Kendall's τ-b, averaged: VQAScore 0.40, PickScore 0.30, HPSv2.1 0.29, CLIPScore 0.13.

**What it means for §2.** GenAI-Bench's ratings are about prompt following. On that question
VQAScore is the closest of the four metrics to people. That supports reading §2's prompt-following
comparison by VQAScore, where SD3.5 Medium and FLUX.1 schnell tie. The preference models' lead for
FLUX in §2 measures something else, and GenAI-Bench has no preference ratings to check it.
CLIPScore is a weak guide to prompt following, as its narrow range in §2 already suggested.

**VLM judge** (D-027, D-028; `results/judge/test_r1/summary.md`, `judge_bias.csv`). Qwen3-VL-8B-Instruct,
run with vLLM, answers in two ways. Absolute: a 1–5 score after listing the unmet requirements,
reported as the expected score from the probabilities of the five digits. Pairwise: which of two
images fits the prompt better, asked in both orders. The rubric was revised twice on the 80 dev
prompts and then frozen; the test round ran once, on the other 1,520 prompts (9,120 images).

- **Position bias (H3, supported).** 58.1 % [56.2, 60.1] of the verdicts that picked an image picked
  the one shown first. The two orders gave the same verdict for 65 % of pairs, and the same image
  was picked both times in only 50 %.
- **Agreement with people.** Absolute (expected score): 0.712 [0.701, 0.723]; with the integer
  score 0.673. That is below VQAScore on the same 17,049 pairs: −0.020 [−0.031, −0.011] (paired,
  exploratory; added after the test round). Pairwise verdicts on the 1,000 H3 pairs: 0.646
  [0.624, 0.668], counting a pick only when both orders agree.
- **Blind test.** Mean score 3.20 with the real image, 1.01 with a grey image, 1.03 with another
  prompt's image (500 images); by expected score the real image wins 99 % of the time.
- **Consistency and spread.** With 3 samples at temperature 0.7, the absolute scores all agree for
  59 % of images and the pairwise verdicts for 69 % of pairs. Scores bunch at the ends: 1 for
  25 % of images, 5 for 37 %, 3 or 4 for 15 %.
- **On dev, an earlier rubric failed the blind test.** Rubric r2 scored a plain grey image like
  a real one (3.01 vs 2.88), often 5 with no issue listed. Asking for a description of what the
  image shows before the check fixed that (r3), at a small cost in dev agreement (0.680 vs 0.709,
  overlapping CIs). Picking r3 departed from the selection rule written before that round
  (D-028 revision 3).

So on this benchmark the 8B judge is a weaker prompt-following metric than VQAScore, and its
pairwise verdicts depend on which image comes first. Any pairwise use needs both orders. VQAScore
stays the main automatic metric; the judge's lists of unmet requirements are what it adds.

**Our own images, one annotator** (guide §9.5, D-029; `results/human_eval/summary.md`,
`human_eval_agreement.csv`; protocol in `docs/human_eval_protocol.md`). The author labelled 200
blind pairs from the test split: the same prompt and seed from two of our four models, 50 pairs
per stratum (counting, spatial, text, GenAI-Bench). For each pair: which image matches the text
better (A, B, same, neither)? Attention checks: 19 of 20 right (pass mark 18). Hidden repeats:
the same answer on 0.95 of 40 pairs, Cohen's κ 0.92. Answers: A 71, B 58, same 71, neither 0. The
129 pairs where an image was picked are used below; equal metric scores count 1/2.

| Metric | Agreement with the annotator | 95 % CI (prompts) | Pairs |
| --- | --- | --- | --- |
| VQAScore | 0.860 | [0.798, 0.915] | 129 |
| PickScore | 0.775 | [0.703, 0.847] | 129 |
| HPSv2.1 | 0.752 | [0.677, 0.827] | 129 |
| CLIPScore | 0.643 | [0.559, 0.727] | 129 |
| Detector check (counting, spatial) | 0.706 | [0.618, 0.784] | 51 |
| OCR check (text) | 0.860 | [0.756, 0.943] | 43 |

The order matches GenAI-Bench: VQAScore first, the two preference models next, CLIPScore last.
That order was written down as the expectation before labelling; this comparison is exploratory,
not a test. With 129 pairs, VQAScore's CI overlaps the preference models'. CLIPScore is at
chance on counting: 0.455 [0.303, 0.606] over 33 pairs. Each stratum rests on 18–43 pairs.

## 4. Limitations
- Detector and OCR checks have their own error rates (§1), measured on 50 dev images each by one
  annotator; the OCR threshold was chosen on the same images.
- The OCR check ignores extra text, so a phrase written with repeats still passes: SD 1.5's
  "SALE SAL SALE" for "SALE" (`sd15__tt0002__s0` in `results/prompts_v1_test/per_sample.csv`)
  scores OCR 1.00, VQAScore 0.18.
- All checker labels and the smoke labels come from one annotator, the author.
- The human check on our own images (§3) also has one annotator, the author, who knew the
  aggregate results. Consistency comes from hidden repeats, not from a second person. Before
  labelling, an AI pre-evaluation was run on the practice items and a first pass over the main
  items. Its labels were not kept, and the author states he did not see them before labelling.
  The labels were typed into a CSV rather than exported from the labelling page, so there are
  no answer times. The file was committed before the answer key was opened (D-029 note).
- The meta-evaluation (§3) uses GenAI-Bench images from six other, mostly older models than §2's.
- GenAI-Bench and VQAScore were introduced in the same paper (Lin et al., 2024,
  arXiv:2404.01291). Our VQAScore uses a different backbone (Qwen2.5-VL-7B) and was not tuned on
  GenAI-Bench, but the benchmark's prompts and rating instructions may still suit that kind of
  metric.
- The VLM judge (Qwen3-VL-8B) and VQAScore's backbone (Qwen2.5-VL-7B) come from the same model
  family; before the judge grades anything in module C, a judge from another family is added
  (D-027). Its rubric was tuned by the author on 80 dev prompts.
- Greedy decoding with vLLM does not repeat exactly across jobs: rerunning the same pairwise
  prompt on dev gave 0.657 instead of 0.660 agreement (D-028 revision 3).
- The human ceiling in §3 is a rough reference rather than a strict bound: it compares one
  rater's 1–5 rating, which ties often (ties count 1/2), with the mean of two others, while the
  metrics are compared with the mean of all three.
- One seed set (4 seeds) and one card type; generation settings are each model's defaults, not
  tuned.
