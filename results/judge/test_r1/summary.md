# VLM judge, round `test` (`test_r1`)

Made 2026-10-20T08:37:00.545+00:00 at commit `f47699e` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/qwen3-vl-8b@0c351dd/bf16/img1024/tmpl-b9e98ed8`; 16620 answers; generation took 1750.0 s.

## Parsing

- abs: 11620 answers, 99.8 % valid JSON, 27 unreadable, 27 cut off at the token limit
- pair: 5000 answers, 99.9 % valid JSON, 4 unreadable, 4 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 2325 (25 %), 2: 2007 (22 %), 3: 671 (7 %), 4: 714 (8 %), 5: 3377 (37 %), None: 26 (0 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.673 | [0.664, 0.682] | 17049 |
| judge expected score | 0.712 | [0.701, 0.723] | 17049 |
| vqascore (reference) | 0.732 | [0.722, 0.742] | 17159 |

Paired on the same 17049 pairs (exploratory): judge expected score 0.712 − VQAScore 0.733 = -0.020 [-0.031, -0.011].

## Pairwise verdicts

- Agreement with people: 0.646 [0.624, 0.668] over 1000 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 50 % of pairs.
- Same verdict in both orders: 0.652 [0.622, 0.683].
- Position bias: 0.581 [0.562, 0.601] of 1565 image-picking verdicts pick the image shown first (0.5 = no bias); **H3 supported** (pre-registered: holds if the 95 % CI excludes 0.5).

## Consistency (3 samples at temperature 0.7)

- abs: 500 items; all 3 samples agree on 59 %; the samples' majority equals the greedy answer on 74 %; mean SD of the sampled scores 0.33.
- pair: 1000 items; all 3 samples agree on 69 %; the samples' majority equals the greedy answer on 87 %.

## Blind test

- Real image vs grey image (498 images): mean score 3.20 vs 1.01; the real image scores higher in 88 % (ties 1/2; 99 % by expected score).
- Real image vs mismatch image (498 images): mean score 3.20 vs 1.03; the real image scores higher in 88 % (ties 1/2; 99 % by expected score).
