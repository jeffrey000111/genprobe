# GenProbe report

_Status: M2 (benchmark on prompt set v1) done; M3 (metric meta-evaluation) next. Every number
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
Pre-registered hypotheses: see `docs/decisions.md`. — TODO (M3)

## 4. Limitations
- Detector and OCR checks have their own error rates (§1), measured on 50 dev images each by one
  annotator; the OCR threshold was chosen on the same images.
- The OCR check ignores extra text, so a phrase written with repeats still passes (SD 1.5's
  "SALE SAL SALE" in `results/examples/text.jpg` scores OCR 1.00; VQAScore gives it 0.18).
- All checker labels and the smoke labels come from one annotator, the author.
- Main meta-evaluation (M3) uses GenAI-Bench images from other models.
- One seed set (4 seeds) and one card type; generation settings are each model's defaults, not
  tuned.
