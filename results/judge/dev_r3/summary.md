# VLM judge, round `dev` (`dev_r3`)

Made 2026-10-20T04:59:51.590+00:00 at commit `18dfd43` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/qwen3-vl-8b@0c351dd/bf16/img1024/tmpl-b9e98ed8`; 7575 answers; generation took 568.7 s.

## Parsing

- abs: 2880 answers, 99.8 % valid JSON, 5 unreadable, 5 cut off at the token limit
- pair: 4695 answers, 99.9 % valid JSON, 5 unreadable, 5 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 129 (27 %), 2: 108 (22 %), 3: 38 (8 %), 4: 41 (9 %), 5: 162 (34 %), None: 2 (0 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.646 | [0.606, 0.686] | 929 |
| judge expected score | 0.680 | [0.632, 0.731] | 929 |
| vqascore (reference) | 0.737 | [0.694, 0.781] | 939 |

## Pairwise verdicts

- Agreement with people: 0.657 [0.626, 0.690] over 939 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 53 % of pairs.
- Same verdict in both orders: 0.662 [0.620, 0.706].
- Position bias: 0.569 [0.535, 0.603] of 1516 image-picking verdicts pick the image shown first (0.5 = no bias) (dev round: descriptive only; H3 is tested on the test round).

## Consistency (3 samples at temperature 0.7)

- abs: 480 items; all 3 samples agree on 58 %; the samples' majority equals the greedy answer on 72 %; mean SD of the sampled scores 0.33.
- pair: 939 items; all 3 samples agree on 70 %; the samples' majority equals the greedy answer on 88 %.

## Blind test

- Real image vs grey image (478 images): mean score 3.00 vs 1.00; the real image scores higher in 87 % (ties 1/2; 100 % by expected score).
- Real image vs mismatch image (478 images): mean score 3.00 vs 1.01; the real image scores higher in 86 % (ties 1/2; 99 % by expected score).

## Clear disagreements with people (dev only, for the rubric)

Mean human ratings at least 2 points apart. Absolute scores in the wrong order: 17 pairs; pairwise picks of the worse image in both orders: 9. The largest gaps:

- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > DeepFloyd_I_XL_v1 1.67; judge 4 vs 5; issues it saw: ['Cannot count stars to verify if more stars are visible than city lights', 'No explicit comparison of star count vs. light count is visually verifiable'] vs []
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_Turbo 4.67 > DeepFloyd_I_XL_v1 1.67; judge 4 vs 5; issues it saw: ['The number of visible stars is not clearly more than the number of lights in the city; they appear roughly comparable or the stars are not visibly more numerous'] vs []
- "A vintage car show with more convertibles than sedans.": people DeepFloyd_I_XL_v1 4.67 > SDXL_Turbo 1.67; judge 2 vs 4; issues it saw: ['No sedans are visible to compare with convertibles', 'Cannot confirm if more convertibles than sedans are present as no sedans are visible'] vs ["Not enough convertibles to be 'more than sedans' — the count is roughly equal or sedans slightly more numerous"]
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > DALLE_3 2.00; judge 4 vs 5; issues it saw: ['Cannot count stars to verify if more stars are visible than city lights', 'No explicit comparison of star count vs. light count is visually verifiable'] vs []
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_Turbo 4.67 > DALLE_3 2.00; judge 4 vs 5; issues it saw: ['The number of visible stars is not clearly more than the number of lights in the city; they appear roughly comparable or the stars are not visibly more numerous'] vs []
- "A vintage car show with more convertibles than sedans.": people DeepFloyd_I_XL_v1 4.67 > SDXL_Base 2.00; judge 2 vs 4; issues it saw: ['No sedans are visible to compare with convertibles', 'Cannot confirm if more convertibles than sedans are present as no sedans are visible'] vs ["Not enough convertibles to be 'more than sedans' — the image shows roughly equal or slightly more sedans than convertibles.", 'Cannot verify exact counts of convertibles vs sedans due to depth and angle, but visual count suggests sedans are not outnumbered.']
- "Three vintage cars parked in a row, each a different color.": people DALLE_3 5.00 > SDXL_Turbo 2.33; judge 4 vs 5; issues it saw: ['Only three cars are in the immediate foreground row as requested, but many more are visible in the background and foreground', "The prompt asks for 'three vintage cars parked in a row', but the image shows more than three cars in the foreground row", 'The prompt does not specify background cars, but the image shows many more than three cars in the background row', 'The prompt does not specify the colors, but the image shows multiple colors including teal, yellow, beige, pink, green, and blue', 'The prompt does not specify spatial relations, but the image shows cars parked in rows with some in front of others', 'The prompt does not specify actions, but the image shows cars parked and stationary', 'The prompt does not specify materials, but the image shows metal and glass materials on the cars', 'The prompt does not specify text, but the image shows license plates with numbers and letters'] vs []
- "A monkey is stealing apples from a tourist's basket. The number of apples in the monkey's arms is less than the number of apples in the basket.": people DALLE_3 5.00 > SDXL_Base 2.67; judge 1 vs 2; issues it saw: ["The monkey is not stealing from a tourist's basket; there is no tourist visible.", 'The monkey is holding a basket, not apples in its arms.', "The number of apples in the monkey's arms is not less than the number in the basket, because it is holding a basket full of apples, not a small number of individual apples.", "The prompt says 'the number of apples in the monkey's arms is less than the number in the basket' — but the monkey is holding a basket, so the apples in its arms are not being compared to the apples in the basket as separate entities; the basket is being held, not apples.", "The prompt implies the monkey is stealing from a tourist's basket, but no tourist is visible."] vs ["The monkey is not stealing apples from a tourist's basket; it is riding a bike with a basket of apples and reaching for apples from a tree.", "The monkey's arms do not contain any apples; its hands are empty or one hand is reaching for an apple.", "The prompt says 'the number of apples in the monkey's arms is less than the number in the basket' — but there are zero apples in the arms, which is less than the number in the basket, so this part is technically met, but the context of stealing from a tourist's basket is not shown."]
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_Turbo (4.67), the judge picks DeepFloyd_I_XL_v1 (1.67) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DeepFloyd_I_XL_v1 (4.67), the judge picks SDXL_Turbo (1.67) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer Midjourney_6 (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_2_1 (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_Turbo (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DeepFloyd_I_XL_v1 (4.67), the judge picks Midjourney_6 (2.33) in both orders
- pairwise, "The sun illuminates the garden, a shovel sticks in the soil, and gloves lay on the bench.": people prefer DALLE_3 (4.67), the judge picks SDXL_Base (2.67) in both orders
- pairwise, "A girl in a smart blazer using a globe to explain continents to a girl in a school uniform and backpack.": people prefer DALLE_3 (5.00), the judge picks DeepFloyd_I_XL_v1 (3.00) in both orders
