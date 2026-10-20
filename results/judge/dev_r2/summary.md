# VLM judge, round `dev` (`dev_r2`)

Made 2026-10-20T04:15:19.442+00:00 at commit `1fc0a2f` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/qwen3-vl-8b@0c351dd/bf16/img1024/tmpl-1c6bef16`; 7575 answers; generation took 644.6 s.

## Parsing

- abs: 2880 answers, 99.9 % valid JSON, 4 unreadable, 4 cut off at the token limit
- pair: 4695 answers, 96.8 % valid JSON, 149 unreadable, 149 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 140 (29 %), 2: 125 (26 %), 3: 25 (5 %), 4: 31 (6 %), 5: 158 (33 %), None: 1 (0 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.659 | [0.620, 0.695] | 935 |
| judge expected score | 0.709 | [0.665, 0.753] | 935 |
| vqascore (reference) | 0.737 | [0.694, 0.781] | 939 |

## Pairwise verdicts

- Agreement with people: 0.559 [0.535, 0.583] over 939 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 29 % of pairs.
- Same verdict in both orders: 0.491 [0.448, 0.532].
- Position bias: 0.468 [0.437, 0.500] of 1128 image-picking verdicts pick the image shown first (0.5 = no bias) (dev round: descriptive only; H3 is tested on the test round).

## Consistency (3 samples at temperature 0.7)

- abs: 480 items; all 3 samples agree on 67 %; the samples' majority equals the greedy answer on 84 %; mean SD of the sampled scores 0.29.
- pair: 939 items; all 3 samples agree on 43 %; the samples' majority equals the greedy answer on 66 %.

## Blind test

- Real image vs grey image (479 images): mean score 2.88 vs 3.01; the real image scores higher in 51 % (ties 1/2; 66 % by expected score).
- Real image vs mismatch image (479 images): mean score 2.88 vs 1.01; the real image scores higher in 85 % (ties 1/2; 97 % by expected score).

## Clear disagreements with people (dev only, for the rubric)

Mean human ratings at least 2 points apart. Absolute scores in the wrong order: 14 pairs; pairwise picks of the worse image in both orders: 10. The largest gaps:

- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > DeepFloyd_I_XL_v1 1.67; judge 3 vs 4; issues it saw: ["Cannot count stars to verify 'more visible stars than the number of lights'", "Cannot verify if stars are 'more' than lights as both are too numerous to count exactly"] vs ['Cannot verify if more stars are visible than city lights due to image resolution and lack of exact count']
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_Turbo 4.67 > DeepFloyd_I_XL_v1 1.67; judge 3 vs 4; issues it saw: ['Cannot verify if number of visible stars is more than number of lights in city', 'Cannot verify if stars are more than lights due to image resolution and scale'] vs ['Cannot verify if more stars are visible than city lights due to image resolution and lack of exact count']
- "A vintage car show with more convertibles than sedans.": people DeepFloyd_I_XL_v1 4.67 > SDXL_Turbo 1.67; judge 3 vs 4; issues it saw: ["No exact count of convertibles vs sedans is visible to confirm 'more convertibles than sedans'", "The text '189' on the car is not clearly readable as '189' in the image", "The text on the sign is not clearly readable as 'HISTORIC CARS'"] vs ["Not enough convertibles to clearly satisfy 'more convertibles than sedans'"]
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > DALLE_3 2.00; judge 3 vs 5; issues it saw: ["Cannot count stars to verify 'more visible stars than the number of lights'", "Cannot verify if stars are 'more' than lights as both are too numerous to count exactly"] vs []
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_Turbo 4.67 > DALLE_3 2.00; judge 3 vs 5; issues it saw: ['Cannot verify if number of visible stars is more than number of lights in city', 'Cannot verify if stars are more than lights due to image resolution and scale'] vs []
- "A vintage car show with more convertibles than sedans.": people DeepFloyd_I_XL_v1 4.67 > SDXL_Base 2.00; judge 3 vs 4; issues it saw: ["No exact count of convertibles vs sedans is visible to confirm 'more convertibles than sedans'", "The text '189' on the car is not clearly readable as '189' in the image", "The text on the sign is not clearly readable as 'HISTORIC CARS'"] vs ["Not enough convertibles to confirm 'more convertibles than sedans'"]
- "A monkey is stealing apples from a tourist's basket. The number of apples in the monkey's arms is less than the number of apples in the basket.": people DALLE_3 5.00 > SDXL_Turbo 2.67; judge 1 vs 2; issues it saw: ["The monkey is not stealing from a tourist's basket; the basket is not clearly associated with a tourist", 'The monkey is holding a basket, not apples in its arms', "The number of apples in the monkey's arms is not less than the number in the basket (it's holding a basket, not apples)"] vs ['The monkey is not stealing apples; it is sitting beside the basket and appears to be selecting one apple', 'The monkey has no apples in its arms', "The number of apples in the monkey's arms is not less than the number in the basket because there are zero in its arms"]
- "A vintage car show with more convertibles than sedans.": people DeepFloyd_I_XL_v1 4.67 > Midjourney_6 2.33; judge 3 vs 4; issues it saw: ["No exact count of convertibles vs sedans is visible to confirm 'more convertibles than sedans'", "The text '189' on the car is not clearly readable as '189' in the image", "The text on the sign is not clearly readable as 'HISTORIC CARS'"] vs ["No sedans are visible, but the prompt requires 'more convertibles than sedans' — since no sedans are present, the comparison cannot be verified as 'more'"]
- pairwise, "A swan with a silver anklet on a crystal lake.": people prefer DALLE_3 (4.67), the judge picks SDXL_Base (1.67) in both orders
- pairwise, "A red fox sneaking up on two unsuspecting rabbits.": people prefer DALLE_3 (4.67), the judge picks SDXL_2_1 (2.00) in both orders
- pairwise, "Three vintage cars parked in a row, each a different color.": people prefer DALLE_3 (5.00), the judge picks SDXL_Turbo (2.33) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DALLE_3 (4.67), the judge picks Midjourney_6 (2.33) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DeepFloyd_I_XL_v1 (4.67), the judge picks Midjourney_6 (2.33) in both orders
- pairwise, "The sun illuminates the garden, a shovel sticks in the soil, and gloves lay on the bench.": people prefer DALLE_3 (4.67), the judge picks SDXL_2_1 (2.33) in both orders
- pairwise, "The sun illuminates the garden, a shovel sticks in the soil, and gloves lay on the bench.": people prefer DALLE_3 (4.67), the judge picks SDXL_Turbo (2.33) in both orders
- pairwise, "A swan with a silver anklet on a crystal lake.": people prefer DALLE_3 (4.67), the judge picks SDXL_Turbo (2.67) in both orders

Grey images scored 3 or more: 241. Examples:

- "A Bengal tiger in the shade looks stronger than another Bengal tiger not in the shade.": score 5; issues []
- "A Bengal tiger in the shade looks stronger than another Bengal tiger not in the shade.": score 5; issues []
- "A Bengal tiger in the shade looks stronger than another Bengal tiger not in the shade.": score 5; issues []
- "A Bengal tiger in the shade looks stronger than another Bengal tiger not in the shade.": score 5; issues []
- "A Bengal tiger in the shade looks stronger than another Bengal tiger not in the shade.": score 5; issues []
- "A black cat wearing a pink bow tie sits on the windowsill and stares into the distance.": score 5; issues []
- "A black cat wearing a pink bow tie sits on the windowsill and stares into the distance.": score 5; issues []
- "A black cat wearing a pink bow tie sits on the windowsill and stares into the distance.": score 5; issues []
