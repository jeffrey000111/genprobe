# VLM judge, round `dev` (`dev_gemma`)

Made 2026-10-22T22:04:23.496+00:00 at commit `e2359f5` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/gemma-4-31b@842da37/bf16/vis560/img1024/tmpl-b9e98ed8`; 7575 answers; generation took 1662.7 s.

## Parsing

- abs: 2880 answers, 100.0 % valid JSON, 0 unreadable, 0 cut off at the token limit
- pair: 4695 answers, 100.0 % valid JSON, 0 unreadable, 0 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 36 (8 %), 2: 90 (19 %), 3: 90 (19 %), 4: 106 (22 %), 5: 158 (33 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.699 | [0.666, 0.733] | 939 |
| judge expected score | 0.694 | [0.647, 0.743] | 939 |
| vqascore (reference) | 0.737 | [0.694, 0.781] | 939 |

Paired on the same 939 pairs (exploratory): judge expected score 0.694 − VQAScore 0.737 = -0.043 [-0.098, 0.010].

## Pairwise verdicts

- Agreement with people: 0.711 [0.676, 0.746] over 939 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 57 % of pairs.
- Same verdict in both orders: 0.826 [0.794, 0.859].
- Position bias: 0.481 [0.463, 0.500] of 1294 image-picking verdicts pick the image shown first (0.5 = no bias) (dev round: descriptive only; H3 is tested on the test round).

## Consistency (3 samples at temperature 0.7)

- abs: 480 items; all 3 samples agree on 83 %; the samples' majority equals the greedy answer on 91 %; mean SD of the sampled scores 0.09.
- pair: 939 items; all 3 samples agree on 88 %; the samples' majority equals the greedy answer on 94 %.

## Blind test

- Real image vs grey image (480 images): mean score 3.54 vs 1.00; the real image scores higher in 96 % (ties 1/2; 100 % by expected score).
- Real image vs mismatch image (480 images): mean score 3.54 vs 1.01; the real image scores higher in 96 % (ties 1/2; 100 % by expected score).

## Clear disagreements with people (dev only, for the rubric)

Mean human ratings at least 2 points apart. Absolute scores in the wrong order: 2 pairs; pairwise picks of the worse image in both orders: 6. The largest gaps:

- "A playground with swings, but no slide.": people Midjourney_6 4.67 > SDXL_2_1 2.00; judge 3 vs 5; issues it saw: ['contains a slide'] vs []
- "Three vintage cars parked in a row, each a different color.": people DALLE_3 5.00 > SDXL_Turbo 2.33; judge 4 vs 5; issues it saw: ['more than three cars are present in the image'] vs []
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DeepFloyd_I_XL_v1 (4.67), the judge picks SDXL_Turbo (1.67) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_2_1 (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "Three vintage cars parked in a row, each a different color.": people prefer DALLE_3 (5.00), the judge picks SDXL_Turbo (2.33) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DALLE_3 (4.67), the judge picks Midjourney_6 (2.33) in both orders
- pairwise, "A classroom with books, but no book open.": people prefer SDXL_Turbo (4.00), the judge picks SDXL_2_1 (2.00) in both orders
- pairwise, "Five nervous deer.": people prefer DALLE_3 (5.00), the judge picks SDXL_2_1 (3.00) in both orders
