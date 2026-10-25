# GenProbe report

_Status: M2 (benchmark on prompt set v1), M3 (metric meta-evaluation: H1–H3, the VLM judge
and a human check on our own images), module C (the generation-debugging agent, §4) and module D
(re-ranking, video, preference tuning and multi-shot video, §5) done. Every number here is copied
from a file in `results/`, named next to it; nothing is reported before it exists there. Choices
behind the numbers are in `docs/decisions.md` (D-xxx)._

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

## 4. Generation-debugging agent (module C)

**Question.** With the GPU time that SDXL best-of-4 costs, does an agent that checks each image with
the metrics trusted in §1–3, diagnoses what is wrong and changes one thing at a time get more
prompts right than spending that time on more samples?

**Agent** (D-030, D-031 with revisions 1–2; `genprobe/agent/`). It starts from the SDXL seed-0 image
of prompt set v1. Its own checks: the detector for counting and spatial prompts, OCR ≥ 0.85 for
text, and VQAScore ≥ 0.9; a template prompt passes only when its checker passes and VQAScore ≥ 0.9,
a GenAI-Bench prompt on VQAScore alone. From the failed check it diagnoses wrong count, missing
object, wrong relation, misspelt or missing text, or low alignment, and tries that diagnosis's
actions in order: switch to the category's best model on dev (SD3.5 Medium for counting and
spatial, FLUX.1 schnell for text and GenAI-Bench), then a rule-based rewrite that states the
failing requirement, cfg 7.5 instead of SDXL's 5.0, or a new seed. When the list is used up, the
rest of the budget goes to new seeds. Each attempt changes one thing of the best image so far and
replaces it only if clearly better (pass, then the nearer miss, then VQAScore +0.02). It stops when
its own checks pass or nothing affordable is left. Every step is logged with its reason
(`results/agent/testrule/audit.jsonl`).

**Budget and baselines** (D-032). Per prompt, 4 × (median SDXL generation + median checks for the
category): 21.7 GPU-seconds on average over the test prompts. The agent is charged the measured
seconds of each attempt; model loading is not charged. Every group gives one image per prompt,
picked with the agent's own comparison:

- *single*: the SDXL seed-0 image;
- *best_of_4*: SDXL seeds 0–3;
- *routed_best_of_k*: the category's dev-best model (the agent's routing table) with as many of its
  seeds as the budget pays for (3 for counting and spatial, 4 for text and GenAI-Bench);
- *agent*: rules r2;
- *agent, SDXL only*: the same agent without the switch to another model (ablation).

**Grading.** The agent optimizes VQAScore (Qwen2.5-VL) and the checkers, so the grade comes from a
judge of another family: Gemma 4 31B-it, run with §3's frozen rubric without any tuning, always
given the original prompt. A prompt passes only on a score of 5 ("every requirement is met"). Before
it graded anything, it went through §3's checks on the 1,520 GenAI-Bench test prompts
(`results/judge/test_gemma/summary.md`):

- agreement with people by expected score 0.729 [0.719, 0.740], VQAScore 0.732 on the same 17,159
  pairs (paired −0.003 [−0.014, 0.008]); 0.709 [0.701, 0.718] by the integer score that decides a
  pass;
- no position bias: 0.490 [0.478, 0.503] of pairwise picks go to the image shown first;
- blind test: mean score 3.57 with the real image, 1.00 with a grey one, 1.02 with another prompt's;
- 3 samples at temperature 0.7 agree on 84 % of images.

**Protocol.** The rules were changed once on the 120 dev prompts (r1 → r2, after the judge graded
the first dev run) and then frozen. The 480 test prompts were run once (`testrule`, `testsdxl`) and
graded once.

**Results** (test, 480 prompts; `results/agent/testrule+testsdxl/judged.md`). Share of prompts the
judge passes; 95 % CIs over prompts are in the file for every cell:

| Group | All [95 % CI] | Counting (56) | Spatial (56) | Text (48) | GenAI-Bench (320) | GPU-s / prompt | GPU-s / success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.246 [0.21, 0.28] | 0.286 | 0.250 | 0.146 | 0.253 | 5.3 | 21.6 |
| best_of_4 | 0.367 [0.33, 0.41] | 0.589 | 0.554 | 0.188 | 0.322 | 21.7 | 59.3 |
| routed_best_of_k | **0.581** [0.54, 0.62] | 0.786 | 0.732 | 0.729 | 0.497 | 14.7 | 25.3 |
| agent | 0.546 [0.50, 0.59] | 0.732 | 0.732 | 0.604 | 0.472 | 11.5 | **21.1** |
| agent, SDXL only | 0.342 [0.30, 0.38] | 0.554 | 0.571 | 0.125 | 0.297 | 14.8 | 43.4 |

Paired differences on the same prompts:

| Comparison | All | Counting | Spatial | Text | GenAI-Bench |
| --- | --- | --- | --- | --- | --- |
| agent − best_of_4 | **+0.179** [0.14, 0.22] | +0.143 [−0.02, 0.29] | +0.179 [0.04, 0.32] | +0.417 [0.25, 0.58] | +0.150 [0.10, 0.20] |
| agent − routed_best_of_k | −0.035 [−0.08, 0.01] | −0.054 [−0.18, 0.07] | 0.000 [−0.12, 0.12] | −0.125 [−0.29, 0.04] | −0.025 [−0.07, 0.02] |
| agent − agent, SDXL only | **+0.204** [0.16, 0.25] | +0.179 [0.04, 0.34] | +0.161 [0.04, 0.29] | +0.479 [0.33, 0.63] | +0.175 [0.13, 0.22] |
| agent, SDXL only − best_of_4 | −0.025 [−0.06, 0.01] | −0.036 [−0.20, 0.12] | +0.018 [−0.12, 0.18] | −0.062 [−0.19, 0.06] | −0.025 [−0.06, 0.01] |

- **The agent beats spending the budget on more SDXL samples**: +0.179 [0.14, 0.22] over best-of-4,
  at about half its GPU time (11.5 vs 21.7 s per prompt).
- **It does not beat plain routing.** Best-of-k with the category's dev-best model passes 0.581, the
  agent 0.546 (−0.035 [−0.08, 0.01]); the gap is largest on text (0.604 vs 0.729). The agent is
  cheaper because it stops as soon as its own checks pass: 11.5 vs 14.7 GPU-s per prompt, 21.1 vs
  25.3 per success.
- **Diagnosis-driven repair adds nothing measurable.** Without the model switch, the same agent does
  no better than SDXL best-of-4 (−0.025 [−0.06, 0.01]), although its final image came from a rewrite
  in 26 prompts and from the higher cfg in 56 (from a new seed in 176; `results/agent/testsdxl/summary.md`).
  For the full agent, the final image came from the model switch in 166 prompts, a new seed in 102,
  a rewrite in 21 and the higher cfg in 14 (`results/agent/testrule/summary.md`).
- **Passing its own checks is not passing the judge.** By its own checks the agent passes more
  prompts than routed best-of-k (0.729 vs 0.700, `results/agent/testrule/summary.md`); by the judge,
  fewer. Of the 218 prompts whose final image the judge fails, the first image was diagnosed as low
  alignment in 140, had already passed the agent's checks in 44 (so the agent never tried), and had
  misspelt text in 14, a wrong count in 11, a wrong relation in 6, missing text in 2 and a missing
  object in 1.
- **Dev told the same story** (D-031 revisions 1–2; 120 prompts): agent r1 0.550 and r2 0.533 vs
  routed best-of-k 0.600; the SDXL-only agent 0.375 vs best-of-4 0.417.
- **Budget.** The full agent stayed within its budget on every prompt. The SDXL-only agent went over
  on 4 prompts, by at most 0.043 GPU-s, because it plans with median costs and is charged measured
  ones (`results/agent/testsdxl/audit.jsonl`).

**A person's check of the judge** (D-033; `results/judge_check/summary.md`; protocol in
`docs/judge_check_protocol.md`). The author labelled, blind to group, model and grade, the final
images of best-of-4, routed best-of-k and the agent for 50 random test prompts (138 distinct images):
does the image meet every requirement of the prompt (yes / no / unsure)? He got all 10 attention
checks right and gave the same answer to 12 of 15 hidden repeats (κ 0.60).

- On the 122 images answered yes or no, the judge and the person made the same call on 0.762
  [0.67, 0.85] (Cohen's κ 0.532 [0.36, 0.71]; D-033 fixed 0.4 as the floor below which the pass
  rates could not be taken at face value).
- The judge is the stricter of the two. The person also passed 0.893 [0.81, 0.96] of the judge's 56
  passes, but the judge passed only 0.685 [0.55, 0.82] of the person's 73 (23 images the person
  passed and the judge failed, 6 the other way). The absolute pass rates above are therefore low by
  this annotator's standard.
- The comparisons point the same way. On these 50 prompts the person passes the agent's image for
  0.56, routed best-of-k's for 0.58 and best-of-4's for 0.44 (unsure counts as not passed). Agent −
  routed best-of-k: −0.02 [−0.14, 0.10] by the person, −0.14 [−0.28, 0.00] by the judge; agent −
  best-of-4: +0.12 by both. Like the judge, the person sees no gain from the agent over routing.
  With 50 prompts the person's CIs are wide.
- Exploratory: of the judge's passes, the person did not pass 1 of 16 for best-of-4 (SDXL) but 7 of
  29 for routed best-of-k and 5 of 22 for the agent, so the judge may be a little more lenient with
  SD3.5 Medium and FLUX images. The numbers are small.

**What it means.** Here the useful part of "debugging" is knowing which model is good at which
kind of prompt, which §2-style per-category results on dev already tell. A fixed routing table
reaches the same pass rate (if anything slightly higher); the agent's only gain over it is cost,
from stopping early. The repairs open to SDXL (a rule-based rewrite, a higher cfg, a new seed) are
no better than more samples. Why the agent trails routing on text was not analysed. One untested
possibility: the agent keeps the first FLUX image that passes its checks, while best-of-k sees all
four and keeps the one with the highest OCR match. An LLM-planned agent (guide §10.3) was not built
(D-031 revision 2): dev suggested that the actions themselves, not how they are chosen, are the
bottleneck.

## 5. Module D: re-ranking, video, preference tuning and multi-shot video

### 5.1 Re-ranking curve

**Setup** (guide §11.1, D-034; `results/rerank/summary.md`, `fig_rerank.png`). For each of the 480
test prompts, nine seeds from SDXL and nine from the routed model of §4 (SD3.5 Medium for counting and
spatial, FLUX.1 schnell for text and GenAI-Bench). Seeds 0–3 are prompt set v1's; seeds 4–8 are new
(4,800 images). Of the first N seeds, the image with the highest VQAScore is picked, and Gemma 4 31B
grades the picks as in §4 (2,465 distinct images; pass = score 5). Cost per prompt: N generations, plus
N VQAScore runs when N > 1, at the median seconds of §4.

Share of prompts the judge passes [95 % CI], and GPU-seconds per prompt:

| Pick by VQAScore | N = 1 | N = 2 | N = 4 | N = 9 |
| --- | --- | --- | --- | --- |
| SDXL | 0.246 [0.21, 0.28], 3.1 s | 0.310 [0.27, 0.35], 7.8 s | 0.390 [0.35, 0.43], 15.6 s | 0.440 [0.40, 0.48], 35.1 s |
| routed model | 0.463 [0.42, 0.51], 1.9 s | 0.552 [0.51, 0.60], 5.6 s | 0.592 [0.55, 0.64], 11.2 s | 0.679 [0.64, 0.72], 25.1 s |

For reference, from §4 on the same prompts: the agent 0.546 at 11.5 s, routed best-of-k 0.581 at
14.7 s, SDXL best-of-4 0.367 at 21.7 s.

- **More samples help, but which model you sample matters more.** For SDXL, going from N = 1 to 9
  adds +0.194 [0.15, 0.23], each step smaller than the last. Even at N = 9 (0.440, 35 GPU-s), SDXL
  stays below one routed image (0.463, 1.9 GPU-s).
- **Re-ranking the routed model is the cheapest way to a given pass rate.**
  - At N = 2 it matches the agent for half the GPU time: agent − routed N = 2 is −0.006 [−0.05, 0.04]
    (5.6 vs 11.5 GPU-s).
  - At N = 4 and about the same cost it is a little ahead: −0.046 [−0.09, 0.00], the CI reaching 0.
  - At N = 9 it is clearly ahead (−0.133 [−0.18, −0.09]), at about twice the cost.
  - The agent does beat re-ranked SDXL at every N (agent − SDXL N = 9: +0.106 [0.06, 0.15]).
- **Picking with the agent's checks adds nothing over VQAScore alone.** On the same four SDXL images,
  best-of-4 by the agent's checks − by VQAScore is −0.023 [−0.05, 0.00]; routed best-of-k − routed
  N = 4 is −0.010 [−0.04, 0.02].
- **Picking inflates the picking metric's own view.** From N = 1 to 9 the share of SDXL picks with
  VQAScore ≥ 0.9 doubles (0.33 → 0.65), while the judge's pass rate rises by 0.19 (0.25 → 0.44). For
  the routed model the two go from 0.56 to 0.79 and from 0.46 to 0.68. This is the pattern D-034
  expected: the metric that does the picking overrates its picks, so another judge has to grade them.
- **The judge repeats itself.** 1,178 images were graded both here and in §4, in separate jobs: same
  score for 97.6 %, same pass / fail for 99.3 %.

**What it means.** With these models, a fixed budget is best spent by sending each kind of prompt to
the model that is best at it and drawing a few more samples, picked by VQAScore. §4's diagnosis-driven
repairs add nothing to that, and the routing alone does most of the work.

### 5.2 Which metrics agree with people on video

**Data and method** (D-035; `results/video_meta_eval/hypotheses.md`,
`fig_video_meta_eval_heatmap.png`). GenAI-Bench-Video: 800 prompts, each rendered by four older
text-to-video models (Floor33, Gen-2, Pika v1, ModelScope), and each video rated 1–5 by three people
for how well it matches the prompt. The method is §3's: within each prompt, every two videos whose
mean ratings differ form a pair. That gives 3,773 pairs from 775 prompts; 1,027 of the 4,800 pairs
(21.4 %) have equal means and are left out. People among themselves: Krippendorff's α 0.675; each
rating against the mean of the other two agrees on 0.80 of pairs. Scores compared:

- VQAScore with the whole video as input (Qwen2.5-VL-7B, 8 fps, at most 360 × 420 pixels per frame,
  t2v_metrics' settings);
- the image VQAScore on the middle frame;
- CLIPScore, PickScore and HPSv2.1, each averaged over 9 frames.

All were fixed and the analysis code committed before any video was scored.

| Test (pre-registered) | Agreement with people | Difference [97.5 % CI] | Verdict |
| --- | --- | --- | --- |
| H4: VQAScore (video) > CLIPScore (9 frames) | 0.675 vs 0.613 | +0.062 [0.035, 0.090] | supported |
| H5: VQAScore (video) > VQAScore (middle frame) | 0.675 vs 0.693 | −0.018 [−0.034, −0.002] | **contradicted** |

- **As on images, VQAScore beats CLIPScore (H4).**
- **Scoring the whole video did not help (H5).** One frame from the middle of the clip agreed with
  people slightly *more* often than the whole clip, so the expectation in the guide does not hold
  here. One untested possibility is resolution: given a video, the model sees each frame at no more
  than 360 × 420 pixels, much smaller than the single frame it gets as an image.
- **Exploratory: on video the preference models catch up.** PickScore and HPSv2.1 averaged over 9
  frames agree with people on 0.68 of pairs, level with the VQAScores (0.67–0.69). On images they
  trailed VQAScore by 6–7 points (§3). Every metric stays well below the people's 0.80.
- **Exploratory: model-level means can mislead.** ModelScope has the lowest mean human rating of the
  four (2.68), yet the highest mean whole-video VQAScore (0.436 vs 0.338–0.414). PickScore and HPSv2.1
  put it last, as people do. The table is in `hypotheses.md`.

### 5.3 Wan2.2 and LTX-Video beside the older video models

**Setup** (guide §11.2–11.3, D-036; `results/video/summary.md`). Wan2.2 TI2V-5B made one video
(seed 0) for each of 160 GenAI-Bench-Video prompts, drawn at random within the benchmark's tiers
(57 basic, 103 advanced). Settings are the model card's (1280 × 704, 50 steps, cfg 5.0, 24 fps)
except the length. The card's 121 frames took 198.7 s per video on an H100 NVL and 81 frames took
116.5 s, both above the 90 s that D-036 allowed, so the run used 49 frames (about 2 s): 65.4 s per
video, 2.9 GPU-hours for the 160. The four older models' videos of the same prompts are the ones
GenAI-Bench-Video distributes, at their own lengths and resolutions. LTX-Video 0.9.5 (2B, D-039)
made the same 160 videos as a fast comparison, at its card's settings (704 × 480, 50 steps, guidance
3.0, its negative prompt) and also 49 frames: 4.6 s per video. All six models get VQAScore on the
whole video (§5.2) and six VBench dimensions (custom-input mode). Exploratory throughout: no person
rated the Wan2.2 or LTX-Video videos.

Means over the 160 prompts [95 % CI]:

| Measure | Wan2.2 (49 frames) | LTX-Video (49 frames) | Floor33 | Gen-2 | Pika v1 | ModelScope |
| --- | --- | --- | --- | --- | --- | --- |
| VQAScore (whole video) | 0.613 [0.55, 0.67] | 0.413 [0.35, 0.47] | 0.342 [0.28, 0.40] | 0.399 [0.34, 0.47] | 0.407 [0.34, 0.47] | 0.453 [0.40, 0.51] |
| Subject consistency | 0.968 [0.96, 0.97] | 0.988 [0.98, 0.99] | 0.973 [0.97, 0.98] | 0.977 [0.97, 0.98] | 0.985 [0.98, 0.99] | 0.952 [0.94, 0.96] |
| Background consistency | 0.967 [0.96, 0.97] | 0.978 [0.98, 0.98] | 0.978 [0.98, 0.98] | 0.973 [0.97, 0.98] | 0.977 [0.97, 0.98] | 0.961 [0.96, 0.97] |
| Motion smoothness | 0.988 [0.99, 0.99] | 0.997 [1.00, 1.00] | 0.984 [0.98, 0.99] | 0.996 [1.00, 1.00] | 0.996 [1.00, 1.00] | 0.975 [0.97, 0.98] |
| Dynamic degree | 0.531 [0.46, 0.61] | 0.069 [0.03, 0.11] | 0.281 [0.21, 0.35] | 0.069 [0.03, 0.11] | 0.200 [0.14, 0.27] | 0.331 [0.26, 0.41] |
| Aesthetic quality | 0.541 [0.53, 0.55] | 0.511 [0.50, 0.53] | 0.650 [0.64, 0.66] | 0.654 [0.64, 0.67] | 0.592 [0.58, 0.61] | 0.452 [0.44, 0.47] |
| Imaging quality | 0.689 [0.67, 0.70] | 0.606 [0.59, 0.63] | 0.606 [0.59, 0.62] | 0.677 [0.66, 0.69] | 0.648 [0.63, 0.66] | 0.619 [0.61, 0.63] |

Differences below are Wan2.2 − another model, paired on prompts (all in `summary.md`).

- **By VQAScore, Wan2.2 follows the prompts better than every older model:** from +0.160
  [0.09, 0.23] against ModelScope to +0.271 [0.20, 0.34] against Floor33, every CI above 0. This is
  the metric that agreed best with people in §5.2, but its model-level means misled there, and the
  same pattern shows here: ModelScope, which people rated lowest in §5.2, has the highest VQAScore
  of the four older models on these prompts. So the gap is the metric's view and was not checked
  against people.
- **Wan2.2 moves much more.** Dynamic degree (the share of videos VBench counts as moving) is 0.53 for
  Wan2.2 against 0.07–0.33 for the others; Gen-2 and LTX-Video hardly move. The consistency and
  smoothness dimensions favour still videos: LTX-Video, Gen-2 and Pika v1, the three stillest, have
  the highest motion smoothness (0.996–0.997). Wan2.2 is a little below LTX-Video, Floor33, Gen-2
  and Pika v1 on subject consistency (−0.005 to −0.020) and above ModelScope (+0.016). That fits
  more motion rather than worse videos, but this was not checked.
- **Sharper but not prettier.** Wan2.2 has the highest imaging quality (MUSIQ: 0.689; Gen-2 is
  close, +0.012 [−0.00, 0.03]) but lower aesthetic quality (LAION predictor) than Floor33, Gen-2 and
  Pika v1 (−0.051 to −0.113), and higher only than ModelScope and LTX-Video. Why was not analysed.
- **LTX-Video is fast but barely moves.** At a fourteenth of Wan2.2's GPU time per video (4.6 s
  against 65.4 s), it scores 0.199 [0.13, 0.27] lower on VQAScore, level with the 2023 models
  (0.342–0.453). Its dynamic degree (0.069) is as low as Gen-2's, and, as with Gen-2, its stillness
  puts it at the top of the consistency and smoothness rows. It also renders at a smaller size
  (704 × 480), which can lower its imaging quality score (0.606, Wan2.2 +0.083 [0.07, 0.10]).

**What it means.** A 2025 open 5B model is well ahead of the 2023 models on the prompt-following
metric and on motion; the 2B LTX-Video, 14 times faster, is not. VBench's consistency and aesthetic scores do not put it ahead, and the
consistency and smoothness scores favour the stillest videos, so no single VBench dimension can be
read as overall quality.

### 5.4 Preference tuning on VQAScore (Diffusion-DPO)

**Setup** (guide §11.4, D-037; `results/dpo/`). SD 1.5 made four images (seeds 0–3) for each of the
1,200 GenAI-Bench-1600 prompts outside prompt set v1, so no dev or test prompt was trained on, and
none of the template prompts (counting, spatial, text) either. For each prompt, the image with the
highest VQAScore and the one with the lowest became a preference pair when their scores differed by
at least 0.1: 821 pairs, with 379 prompts dropped and a mean gap of 0.575 (`training.md`). diffusers'
Diffusion-DPO script trained a rank-8 LoRA on the pairs: learning rate 1e-5, batch 32, β = 2,500,
2,000 steps, which is about 78 passes over the 821 pairs.

**Training and checkpoint.** On its own training pairs the LoRA kept learning. Implicit accuracy (how
often it prefers the winner more than the base model does) rose from 0.61 over the first 250 steps to
0.78 over the last 250, and the loss fell from 0.68 to 0.54. On dev, the judge's pass rate picked the
2,000-step checkpoint: 0.183, against 0.150 for the base model and 0.133–0.150 at 250, 500 and 1,000
steps, all within each other's CIs (`dev_selection.md`).

**Test** (`summary.md`). The base model and the 2,000-step checkpoint were both regenerated in the
same environment, seeds 0–3 of the 480 test prompts. Gemma 4 31B grades seed 0 (pass = 5); the other
scores average all four seeds. The regenerated base model matches §2's SD 1.5 (VQAScore 0.364 vs
0.365, CLIPScore 0.270 for both).

| Measure | Base SD 1.5 | DPO | DPO − base [95 % CI] |
| --- | --- | --- | --- |
| Judge pass rate (**H6**, pre-registered) | 0.171 | 0.183 | +0.013 [−0.01, 0.04] |
| VQAScore (trained on) | 0.364 | 0.388 | +0.023 [0.01, 0.03] |
| CLIPScore | 0.270 | 0.273 | +0.003 [0.00, 0.00] |
| PickScore | 21.154 | 21.204 | +0.049 [0.03, 0.07] |
| HPSv2.1 | 0.249 | 0.252 | +0.003 [0.00, 0.00] |
| Detector check (counting, spatial) | 0.362 | 0.400 | +0.038 [0.01, 0.07] |
| OCR check (text) | 0.424 | 0.439 | +0.014 [−0.00, 0.03] |

- **H6 is inconclusive.** The judge's pass rate rose by 0.013, with a CI that reaches below 0. It is
  unchanged on counting, spatial and text prompts (text: 0 for both) and +0.019 [−0.01, 0.05] on the
  GenAI-Bench prompts.
- **No sign of reward hacking, but little reward either.** The LoRA fit its training pairs, yet on
  test prompts VQAScore, the metric it was trained on, rose by only 0.023. The scores it was not
  trained on did not fall: PickScore and HPSv2.1 rose slightly, and so did the detector check on
  counting and spatial prompts, which were never trained on.
- **A person sees almost no difference.**
  - Round 1 of the blind check failed its attention checks (0 of 6), so its labels are not used. On
    all six the annotator answered "neither", and the judge had failed every real image (scores 1–3),
    so by the decision rules "neither" was a fair answer: the checks tested SD 1.5, not attention
    (D-037 note).
  - Round 2 was fixed before it was drawn. It used 60 new prompts, and the attention checks used real
    images the judge had passed. It passed: 5 of 6 attention checks right, and all 6 hidden repeats
    got the same answer in both parts (`human_check_r2.md`).
  - Text: 53 of the 60 pairs were "same" (10) or "neither" (43). Of the 7 decided, the DPO image won
    4 (0.57 [0.14, 0.86]).
  - Looks: 52 of 60 were "same". Of the 8 decided, the DPO image won 5 (0.63 [0.25, 0.88]).

**What it means.** Trained on 821 preference pairs from SD 1.5's own samples, Diffusion-DPO fit those
pairs but changed the test images too little for the judge or a person to notice. By every measure
here it did no harm, and nothing points to the model gaming VQAScore. Using VQAScore at test time to
pick among samples (§5.1, on SDXL and the routed models) moved the pass rate far more.

### 5.5 Multi-shot videos from a brief

**Setup** (guide §11.2's creative workflow, D-038; `results/story/summary.md`, `dev_tau.md`).
- **Briefs.** 32 one-sentence briefs, written by Claude for this project before anything was
  generated, each with one main character and 3–4 events ("A small orange cat with a white chest wakes
  up on a sunny windowsill, stretches, jumps down to the kitchen floor and drinks from a blue bowl");
  8 dev and 24 test.
- **Shots.** Gemma 4 31B splits each brief once into 3–4 self-contained shot descriptions (78 test
  shots), which every arm uses. Every shot is 49 frames of Wan2.2 TI2V-5B (§5.3's settings).
- **Arms.**
  - A: text to video from the description.
  - B: a FLUX.1 schnell keyframe, then image to video from it.
  - C, the agent: B with checks. It redraws the keyframe (up to 3) until its VQAScore reaches 0.9,
    module C's threshold, and redoes the video (up to 3) until VQAScore on the whole video reaches
    τ_v. Only failing shots are redone.
  - D: B's keyframe, two videos per shot, keeping the one with the higher VQAScore.
- **τ_v**, fixed on dev: 0.6. On B's 25 dev shots it matched the judge's pass / fail with a balanced
  accuracy of 0.667, level with 0.7 and 0.8; ties went to the lower value.
- **Judge.** Gemma 4 31B grades each shot from its first, middle and last frame (pass = 5), and each
  brief from the middle frame of every shot: does the sequence tell the brief, and is the main
  character the same individual throughout (both 1–5).

| Arm (24 test briefs) | Shot pass rate | Story score (1–5) | Consistency score (1–5) | GPU-s per brief |
| --- | --- | --- | --- | --- |
| A text to video | 0.462 [0.34, 0.58] | 4.42 [4.12, 4.67] | 3.12 [2.67, 3.58] | 212 |
| B keyframe, then image to video | 0.500 [0.38, 0.62] | 4.71 [4.54, 4.88] | 3.62 [3.17, 4.04] | 215 |
| C agent (checks and redo) | 0.526 [0.42, 0.63] | 4.67 [4.46, 4.83] | 3.46 [3.00, 3.92] | 242 |
| D two videos per shot | 0.513 [0.41, 0.63] | 4.62 [4.42, 4.83] | 3.62 [3.12, 4.08] | 434 |

- **Both pre-registered tests are inconclusive.** The per-shot pass rate does not separate the arms.
  H7 (keyframes help), B − A: +0.038 [−0.15, 0.21]. H8 (checks and redo help), C − B: +0.026
  [−0.04, 0.09]. Both are 97.5 % CIs.
- **Exploratory: over a whole brief, keyframes look better.** These paired differences per brief
  were added after the results were seen.
  - Story score: B − A +0.29 [0.04, 0.54]; C − A +0.25 [0.04, 0.46].
  - Consistency score: B − A +0.50 [0.00, 1.04]; C − A +0.33 [−0.17, 0.83].
- **The agent's checks add cost and nothing measurable.**
  - Against keyframes alone (C − B): story −0.04 [−0.12, 0.00] and consistency −0.17 [−0.38, 0.00],
    for 13 % more GPU time (242 against 215 GPU-s per brief).
  - The agent redrew the keyframe for 21 % of shots and redid the video for 5 %. VQAScore on video
    passed 84 % of B's dev shots at τ_v = 0.6 and matched the judge only weakly, so the check found
    little to act on.
  - Two videos per shot (D) double the cost (434 GPU-s) for no gain. C − D, shot pass rate: +0.013
    [−0.06, 0.09].
- **A person leans the same way, without significance.** The author compared A and C on the 24 test
  briefs, blind to the arm, and the test results were withheld from him until he had finished
  (`human_check.md`). He passed the attention checks (4 of 4), and every hidden repeat (3 per part)
  got the same answer.
  - Story: 12 briefs were "same" and 1 "neither". Of the 11 decided, the agent's video won 7 (0.64
    [0.36, 0.91]). Where both he and the judge's story scores separated the two videos, they agreed on
    7 of 7.
  - Consistency: 10 briefs were "same". Of the 14 decided, the agent's video won 9 (0.64
    [0.36, 0.86]).
  - Most of C's shots are B's, so this mainly compares keyframes with plain text to video.

**What it means.** Starting each shot from a keyframe of a strong image model may help a
multi-shot video hold its story and its main character. The evidence is exploratory and at the level
of whole briefs, not shots. Checking each shot with VQAScore and redoing the failures does not add
to that: VQAScore on video is too lenient to tell good shots from bad ones here. This is the lesson
of §4 again: the choice of model and conditioning matters more than diagnosing and repairing.

## 6. Limitations
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
- The VLM judge of §3 (Qwen3-VL-8B) and VQAScore's backbone (Qwen2.5-VL-7B) come from the same
  model family, so module C is graded by Gemma 4 31B instead (D-030). The rubric both judges use
  was tuned by the author on 80 dev prompts, with Qwen3-VL-8B.
- Module C's result rests on one judge. Gemma 4 31B was checked against people on GenAI-Bench
  images from other models, and on the agent's own images against one annotator, the author, who
  knew module C's aggregate results (138 images, κ 0.53, §4). By his standard the judge is strict,
  so the absolute pass rates are low; the comparisons between groups point the same way for both.
- Pass means a judge score of 5, the strictest reading (project author's choice, fixed before any
  grade, D-032); other thresholds were not looked at.
- The agent's rules and routing table were chosen on the 120 dev prompts, and the rules were changed
  once after the judge graded a dev run, so only the test numbers are clean.
- Costs are GPU-seconds of generation and checking only: the agent is charged measured seconds per
  attempt, the baselines median seconds per image, and model loading is charged to no one. Routed
  best-of-k uses at most the 4 seeds prompt set v1 already has, although the budget would pay for
  more FLUX.1 schnell images, so that baseline is if anything a little weak.
- Greedy decoding with vLLM does not repeat exactly across jobs: rerunning the same pairwise
  prompt on dev gave 0.657 instead of 0.660 agreement (D-028 revision 3).
- The human ceiling in §3 is a rough reference rather than a strict bound: it compares one
  rater's 1–5 rating, which ties often (ties count 1/2), with the mean of two others, while the
  metrics are compared with the mean of all three.
- One seed set (4 seeds) and one card type; generation settings are each model's defaults, not
  tuned.
- Re-ranking costs (§5.1) are median seconds per image and leave out model loading. In practice the
  routed model means keeping two generators loaded, which a single-model baseline avoids.
- The video meta-evaluation (§5.2) uses four 2023 video models; how well the metrics rank today's
  models (§5.3) is not checked against people.
- DPO (§5.4) is one run: one seed, one β and learning rate, and 821 pairs from SD 1.5's own
  samples, far fewer than the training script was written for (D-037). The dev choice of checkpoint
  was between pass rates within each other's CIs.
- §5.4's blind check has one annotator, the author, who in round 2 knew the test results and his
  round 1 answers. Round 1 failed its attention checks because of a flaw in the protocol, and round 2
  changed only those checks. Few pairs were decided (7 and 8 of 60), so the person's preference
  shares have wide CIs.
- Wan2.2 ran at 49 frames (2 s), not its card's 121, to fit D-036's time rule, and LTX-Video at
  the same length but its card's smaller size. The older models' videos are used as distributed, at
  their own lengths and resolutions. VBench's motion, consistency and image-quality scores can depend
  on both, so §5.3's comparison is not like for like. LTX-Video's larger and newer releases were not
  run (D-039).
- §5.5 rests on 24 test briefs written by Claude, each with one main character, so its CIs are wide
  and its briefs may not be typical. The judge sees frames, not motion, and judges story and
  consistency from one frame per shot. Gemma's shot descriptions were shorter than its prompt asked
  (D-038 note). One seed per shot (two for D), 49 frames. The table of brief-level differences was
  added after the results were seen. The blind check has one annotator, the author, and few pairs
  were decided (11 and 14 of 24).
