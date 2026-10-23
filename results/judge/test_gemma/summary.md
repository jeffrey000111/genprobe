# VLM judge, round `test` (`test_gemma`)

Made 2026-10-22T23:53:04.556+00:00 at commit `69cc57d` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/gemma-4-31b@842da37/bf16/vis560/img1024/tmpl-b9e98ed8`; 16620 answers; generation took 5214.4 s.

## Parsing

- abs: 11620 answers, 100.0 % valid JSON, 0 unreadable, 0 cut off at the token limit
- pair: 5000 answers, 100.0 % valid JSON, 1 unreadable, 1 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 641 (7 %), 2: 1604 (18 %), 3: 1692 (19 %), 4: 2164 (24 %), 5: 3019 (33 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.709 | [0.701, 0.718] | 17159 |
| judge expected score | 0.729 | [0.719, 0.740] | 17159 |
| vqascore (reference) | 0.732 | [0.722, 0.742] | 17159 |

Paired on the same 17159 pairs (exploratory): judge expected score 0.729 − VQAScore 0.732 = -0.003 [-0.014, 0.008].

## Pairwise verdicts

- Agreement with people: 0.697 [0.674, 0.719] over 1000 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 61 % of pairs.
- Same verdict in both orders: 0.818 [0.794, 0.842].
- Position bias: 0.490 [0.478, 0.503] of 1449 image-picking verdicts pick the image shown first (0.5 = no bias); **H3 inconclusive** (pre-registered: holds if the 95 % CI excludes 0.5).

## Consistency (3 samples at temperature 0.7)

- abs: 500 items; all 3 samples agree on 84 %; the samples' majority equals the greedy answer on 94 %; mean SD of the sampled scores 0.08.
- pair: 1000 items; all 3 samples agree on 84 %; the samples' majority equals the greedy answer on 93 %.

## Blind test

- Real image vs grey image (500 images): mean score 3.57 vs 1.00; the real image scores higher in 96 % (ties 1/2; 99 % by expected score).
- Real image vs mismatch image (500 images): mean score 3.57 vs 1.02; the real image scores higher in 96 % (ties 1/2; 99 % by expected score).
