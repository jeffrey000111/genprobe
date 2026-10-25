# Multi-shot video check protocol (D-038)

Written on 2026-10-25, before the test stage of the multi-shot videos was run and before any item
was drawn.

**Purpose.** The judge grades the multi-shot videos (§5.5 of the report) on frames. This check asks a
person, blind to the arm, whether the agent's videos (arm C: keyframes, checks and redo) tell their
brief better, and keep their main character more consistent, than plain text to video (arm A).

**Who labels.** One annotator, the project author, blind to the arm behind each video. He will know
the aggregate test results if they exist by then.

## What is shown

Each item is one test brief and two videos, A on the left and B on the right: the shots of arm A and
the shots of arm C for that brief, each joined in order into one video (re-encoded alike, 960 pixels
wide, no sound) that loops. Which side is which is random; arm names, scores and the judge's answers
are never shown.

| Part | Question | Items |
| --- | --- | --- |
| 1 | **Which video tells the brief better?** A · B · same · neither | the 24 test briefs, 4 attention checks, 3 hidden repeats |
| 2 | **In which video does the main character stay the same from shot to shot?** A · B · same | the same 24 pairs, sides drawn again, 3 hidden repeats |

Items are drawn at random with seed 0 (`python -m genprobe.story_check make`).

## Rules

- **Part 1:** for each video, note which events of the brief are missing or wrong (the event, its
  order, the main character, the setting). Fewer or smaller misses win. "Same" when both tell it
  equally well or equally badly; "neither" only when both miss most of the brief. Image quality and
  style don't count.
- **Part 2:** look only at the main character across the shots: the same kind, colours, clothing,
  markings. Changes in pose, size in the frame, lighting or angle don't count, and neither does how
  well the story is told. "Same" when you can't tell them apart within about 20 seconds.

## Attention checks and consistency

- Attention checks (part 1): one video is a video of this brief that the judge gave a story score of 5
  (from any arm), the other is the agent's video of another brief, so the answer is the first. Unlike
  the first round of the DPO check, the real video is one the judge passed, so "neither" is not a fair
  answer. Passed with at least **3 of 4**, fixed now; if failed, the labels are not used and that is
  reported. If fewer than 4 briefs have a video with story score 5, the highest-scored are used, and
  the report says so.
- Hidden repeats: 3 per part, sides swapped; same-answer share and Cohen's κ per part.

## How to label

1. On HPC3: `bash scripts/make_story_check.sh`; copy only `data/story_check/story_check_package.zip`
   to the laptop. Don't open `key.json`.
2. Unzip, open `story_check/index.html` in Chrome. Keys 1 = A, 2 = B, 3 = same, 4 = neither (part 1
   only), R = replay both, ← → to move. Export the CSV ("导出 CSV") at the end.
3. Save it as `data/labels/story_check_labels.csv` and commit it before the key is opened. Labels are
   not changed after the analysis.
4. Then `bash scripts/make_story_check.sh analyze` on HPC3 → `results/story/human_check.md`.

## What is computed

Per part: the answers; among briefs where one video was picked, the share where it was the agent's
(0.5 = no difference), with a 95 % CI from 2,000 bootstrap resamples of briefs; the repeats. For
part 1, exploratory: how often the person's pick agrees with the judge's story scores where both
separate the two videos. Nothing here is pre-registered as a hypothesis; H7 and H8 rest on the judge.

**Ethics.** Only the author labels; no personal data is collected.
