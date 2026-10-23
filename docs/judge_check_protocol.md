# Judge check protocol (D-032, D-033)

**Purpose.** Module C's result (REPORT §4) rests on one judge, Gemma 4 31B, which passes an image
only when it gives it 5 ("every requirement is met"). The judge was checked against people on
GenAI-Bench images from other models (§3), not on the agent's own images. This check asks a person
the same question on those images: does the person pass the images the judge passes, and do the
two tell the same story about the agent, routed best-of-k and SDXL best-of-4?

**Who labels.** One annotator, the project author. He knows module C's aggregate results; the
labelling is blind to everything that would identify an image's group, model or grade.

## What is shown

Each item is one prompt (always the original prompt, never a rewrite) and one image, resized to
768 × 768. Group, model, file name, metric scores and the judge's answer are never shown.

| Part | Items | From | Analysed |
| --- | --- | --- | --- |
| Practice | 10 images | dev run `devrule2` | no |
| Main | the final images of best-of-4, routed best-of-k and the agent for 50 test prompts (33 GenAI-Bench, 6 counting, 6 spatial, 5 text, in proportion to the test split); an image two groups share is shown once | test run `testrule` | yes |
| Attention checks | 10: a test image shown with the prompt of another category | test run | pass / fail |
| Hidden repeats | 15 main images shown again, later in the order | test run | consistency |

Items and their order are drawn at random with seed 0 (`python -m genprobe.judge_check make`).

## The question

> **Does the image meet every requirement of the text?**
>
> **yes** (是): every requirement is visible and right ·
> **no** (否): at least one requirement is visibly wrong or missing ·
> **unsure** (拿不准): nothing is visibly wrong, but a requirement is hidden, cropped or too small
> to check, or you can't decide within about 30 seconds

Judge only whether the image does what the text says: the objects, how many there are, their
colours and other attributes, where they are relative to each other, what they are doing,
negations ("no", "without"), comparisons, and any text that should appear. **Do not** judge
beauty, realism, image quality or style, unless the text asks for them. "yes" means the image would
deserve the judge's 5: every requirement met, not just most of them.

## Decision rules

The rules of the own human evaluation apply (`docs/human_eval_protocol.md`, decision rules and
"Changes after the practice round"), read for one image instead of two:

1. **Counts are exact** for written numbers ("three", "six"); "a" / "an" means at least one.
   Objects cut off at the edge count if they are clearly the objects asked for.
2. **Left and right are as seen in the image** (the viewer's left). **Above / below** compares
   the centres of the two objects; resting on top counts, side by side at the same height doesn't.
3. **Requested text must be spelled exactly** (case and punctuation don't matter) and read as one
   phrase in reading order; extra text before, after or beside it is fine, text inserted inside it
   is a misspelling.
4. **Extra things not in the prompt are fine** unless they break a count, a negation or a
   comparison.
5. **Actions** are met only if most people would describe the image with that action. A negated
   action needs the subject present and visibly not doing it. Figurative wording takes its
   everyday meaning.
6. **Things on a screen, poster or photo are not real objects**, unless the prompt asks for a
   picture.
7. **Read the prompt the most natural way.** Don't hunt for a strict reading that the image fails,
   or a loose one it passes.
8. **Spend about 10–20 seconds per item.**

## Attention checks and consistency

- **Attention checks:** the right answer is "no" (the image belongs to another prompt). Passed with
  at least **9 of 10**, fixed before labelling. If the check fails, the main labels are not used and
  the failure is reported.
- **Hidden repeats:** the share of the 15 repeated images given the same answer both times, and
  Cohen's κ.

## How to label

1. On HPC3: `bash scripts/make_judge_check.sh` (CPU node) writes
   `data/judge_check/judge_check_package.zip` and `data/judge_check/key.json`. Copy only the zip to
   the laptop for now. **Don't open `key.json` before labelling is finished.**
2. Unzip; open `judge_check/index.html` in Chrome (works offline; progress is kept in that
   browser). Keys: 1 = yes, 2 = no, 3 = unsure; ← → to move.
3. Do the 10 practice items first; then the main part, with a break and a CSV export
   ("导出 CSV") about every 50 items and at the end.
4. Save the final export as `data/labels/judge_check_labels.csv` and commit it before the key is
   copied over. Labels are not changed once the analysis has been run.
5. Then, after `scripts/hpc_sync.sh push`, run `bash scripts/make_judge_check.sh analyze` on HPC3
   (a CPU node; it needs the key and the judge's grades) → `results/judge_check/summary.md`.

## What is computed (fixed in D-033)

1. **Judge vs person**, on main images answered yes or no: share with the same call, Cohen's κ,
   the share of the judge's passes the person also passes, the share of its fails the person also
   fails, and the share of the person's passes the judge also passes. 95 % CIs from 2,000 bootstrap
   resamples of prompts (seed 0). Per category: exploratory.
2. **Pass rates per group** on the 50 prompts, by the person (yes = pass; no and unsure = not) and
   by the judge on the same images, and the paired differences agent − routed best-of-k,
   agent − best-of-4 and routed best-of-k − best-of-4 by each.
3. **Per group, the share of the judge's passes the person does not pass**: is the judge more
   lenient with one group's images than another's?

**Ethics.** Only the author labels; no personal data is collected.
