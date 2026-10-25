# DPO check protocol (D-037)

**Purpose.** The DPO model (§5 of the report) was trained to raise VQAScore. A model can learn to
please the metric it is trained on without getting better for people, or get worse to look at. This
check asks a person, blind to which model made which image, two questions about the same pairs.

**Who labels.** One annotator, the project author, blind to the model behind each image. He knows the
aggregate results if they exist by then.

## What is shown

Each item is one prompt (from prompt set v1's test split) and two images, A on the left and B on the
right: the base SD 1.5 image and the DPO image for the same prompt and seed (seed 0). Which side is
which is random; model names, scores and the judge's answers are never shown. Images are resized to
768 × 768.

| Part | Question | Items |
| --- | --- | --- |
| 1 | **Which image matches the text better?** A · B · same · neither | 60 pairs (15 per stratum: counting, spatial, text, GenAI-Bench), 6 attention checks, 6 hidden repeats |
| 2 | **Which image looks better?** A · B · same | the same 60 pairs, sides drawn again, 6 hidden repeats |

Items are drawn at random with seed 0 (`python -m genprobe.dpo_check make`).

## Rules

- **Part 1** uses the rules of `docs/human_eval_protocol.md` (decision rules and the changes after the
  practice round): list what each image gets wrong; fewer or smaller misses win; same when both meet
  everything or miss equally; neither only when both miss the point.
- **Part 2** looks at the picture only, never at the text: sharpness, natural shapes and faces, no
  distortions, extra limbs, smeared or melted parts, plausible lighting and composition. Same when you
  can't tell them apart within about 20 seconds. A plainer image that is clean beats a busier one
  with artefacts.

## Attention checks and consistency

- Attention checks (part 1): one of the two images belongs to a prompt of another stratum, so the
  answer is the other one. Passed with at least **5 of 6**, fixed before labelling; if failed, the
  labels are not used and that is reported.
- Hidden repeats: 6 per part, sides swapped; same-answer share and Cohen's κ per part.

## How to label

1. On HPC3: `bash scripts/make_dpo_check.sh`; copy only `data/dpo_check/dpo_check_package.zip` to the
   laptop. Don't open `key.json`.
2. Unzip, open `dpo_check/index.html` in Chrome. Keys 1 = A, 2 = B, 3 = same, 4 = neither (part 1
   only); ← → to move. Export the CSV ("导出 CSV") at the end.
3. Save it as `data/labels/dpo_check_labels.csv` and commit it before the key is opened. Labels are not
   changed after the analysis.
4. Then `bash scripts/make_dpo_check.sh analyze` on HPC3 → `results/dpo/human_check.md`.

## What is computed

Per part: the answers, and among pairs where one image was picked, the share where it was the DPO
image (0.5 = no difference), with a 95 % CI from 2,000 bootstrap resamples of prompts; per stratum;
the repeats. Part 1 says whether the judge's verdict on H6 holds for a person; part 2 whether DPO cost
image quality.

**Ethics.** Only the author labels; no personal data is collected.

## Round 2 (added 2026-10-24, before any round 2 item was drawn)

**Why.** Round 1's attention checks were 0 of 6 right, so by the rules above its labels are not used,
and that is reported. On all six the answer was "neither", and the judge had given every one of the
six real images 1–3 (none passed): the real image missed its own prompt too, so by the decision rules
"neither" was a fair answer. The checks tested the images, not the annotator's attention. This was a
flaw in the protocol, not in the labelling.

**What changes.** Only the attention checks:
- Their real image is one that Gemma 4 31B passed (score 5, seed 0, the test grades of D-037), so it
  matches its prompt and the right answer is unambiguous.
- Each comes from its own test prompt outside the 60 pairs; the decoy is again an image of a prompt
  of another stratum. Text prompts cannot supply a real image (the judge passed none of them).

**What stays.** 60 new pairs (15 per stratum), the two parts, the questions, the decision rules, 6
attention checks with the pass mark of 5 of 6, 6 hidden repeats per part, the analysis, and the rule
that labels are not used if the checks fail. Items are drawn with seed 1 from the test prompts round 1
never showed (its pairs, its checks' prompts and their decoys).

**Who labels.** The same annotator. Unlike in round 1, he knows the test results of D-037 (H6
inconclusive, the automatic metrics) and his own round 1 answers.

**How.** `bash scripts/make_dpo_check.sh make2` on HPC3; copy only
`data/dpo_check_r2/dpo_check_r2_package.zip`; the page exports `dpo_check_r2_labels.csv`, saved as
`data/labels/dpo_check_r2_labels.csv` and committed before the key is opened; then
`bash scripts/make_dpo_check.sh analyze2` → `results/dpo/human_check_r2.md`. Round 1's labels and
results stay as they are.
