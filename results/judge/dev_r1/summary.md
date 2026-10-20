# VLM judge, round `dev` (`dev_r1`)

Made 2026-10-20T03:28:59.660+00:00 at commit `596f309` by `python -m genprobe.judge_eval`. Judge `vllm==0.32.0/qwen3-vl-8b@0c351dd/bf16/img1024/tmpl-51cfbc75`; 7575 answers; generation took 526.4 s.

## Parsing

- abs: 2880 answers, 100.0 % valid JSON, 1 unreadable, 1 cut off at the token limit
- pair: 4695 answers, 99.9 % valid JSON, 6 unreadable, 6 cut off at the token limit

## Absolute scores

Distribution on real images: 1: 59 (12 %), 2: 158 (33 %), 3: 130 (27 %), 4: 13 (3 %), 5: 120 (25 %).

Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):

| Scorer | Agreement | CI | Pairs |
| --- | --- | --- | --- |
| judge score (1-5) | 0.655 | [0.622, 0.690] | 939 |
| judge expected score | 0.705 | [0.660, 0.750] | 939 |
| vqascore (reference) | 0.737 | [0.694, 0.781] | 939 |

## Pairwise verdicts

- Agreement with people: 0.660 [0.628, 0.693] over 939 pairs (a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in 53 % of pairs.
- Same verdict in both orders: 0.666 [0.626, 0.708].
- Position bias: 0.567 [0.533, 0.602] of 1521 image-picking verdicts pick the image shown first (0.5 = no bias) (dev round: descriptive only; H3 is tested on the test round).

## Consistency (3 samples at temperature 0.7)

- abs: 480 items; all 3 samples agree on 66 %; the samples' majority equals the greedy answer on 86 %; mean SD of the sampled scores 0.22.
- pair: 939 items; all 3 samples agree on 70 %; the samples' majority equals the greedy answer on 88 %.

## Blind test

- Real image vs grey image (480 images): mean score 2.95 vs 3.17; the real image scores higher in 50 % (ties 1/2; 58 % by expected score).
- Real image vs mismatch image (480 images): mean score 2.95 vs 1.00; the real image scores higher in 94 % (ties 1/2; 99 % by expected score).

## Clear disagreements with people (dev only, for the rubric)

Mean human ratings at least 2 points apart. Absolute scores in the wrong order: 11 pairs; pairwise picks of the worse image in both orders: 9. The largest gaps:

- "The girl with glasses is drawing, and the girl without glasses is singing.": people DALLE_3 5.00 > DeepFloyd_I_XL_v1 2.00; judge 2 vs 3; issues it saw: ['The girl without glasses is not singing; she is screaming into a microphone', 'The girl with glasses is not drawing; she is sketching a face with musical notes'] vs ['The girl without glasses is not singing; she is also drawing']
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > DALLE_3 2.00; judge 3 vs 4; issues it saw: ['Cannot verify exact count of stars vs city lights', "Cannot verify 'more visible stars than the number of lights' as counts are not discernible"] vs ['Cannot verify exact count of stars vs city lights', 'Cannot verify if more stars are visible than lights in city']
- "Three vintage cars parked in a row, each a different color.": people DALLE_3 5.00 > SDXL_Turbo 2.33; judge 3 vs 5; issues it saw: ['More than three vintage cars are visible', 'The cars are not all in a single row', 'The prompt does not specify colors, but the cars are different colors, which is fine, but the image shows many colors, not just three'] vs []
- "A monkey is stealing apples from a tourist's basket. The number of apples in the monkey's arms is less than the number of apples in the basket.": people DALLE_3 5.00 > SDXL_Turbo 2.67; judge 2 vs 3; issues it saw: ["The monkey is not stealing apples from a tourist's basket; it is holding a basket of apples while another basket sits nearby", 'The monkey is holding a basket, not apples in its arms', "The number of apples in the monkey's arms is not less than the number in the basket, since it is holding a basket of apples"] vs ["The monkey is not stealing apples from a tourist's basket; it is interacting with a basket of apples, but no tourist is visible", 'The monkey has no apples in its arms', 'The prompt requires a comparison of counts, but the monkey has zero apples in arms and the basket has many, so the comparison holds, but the absence of apples in arms is a key unmet requirement']
- "Three vintage cars parked in a row, each a different color.": people DALLE_3 5.00 > SDXL_2_1 2.67; judge 3 vs 5; issues it saw: ['More than three vintage cars are visible', 'The cars are not all in a single row', 'The prompt does not specify colors, but the cars are different colors, which is fine, but the image shows many colors, not just three'] vs []
- "A monkey is stealing apples from a tourist's basket. The number of apples in the monkey's arms is less than the number of apples in the basket.": people Midjourney_6 4.67 > SDXL_Turbo 2.67; judge 2 vs 3; issues it saw: ['The monkey is not stealing apples; it is holding one apple and sitting beside a basket full of apples', "The monkey is not interacting with a tourist's basket; the basket is just in front of it"] vs ["The monkey is not stealing apples from a tourist's basket; it is interacting with a basket of apples, but no tourist is visible", 'The monkey has no apples in its arms', 'The prompt requires a comparison of counts, but the monkey has zero apples in arms and the basket has many, so the comparison holds, but the absence of apples in arms is a key unmet requirement']
- "A starry night sky with more visible stars than the number of lights in the city below.": people SDXL_2_1 4.67 > SDXL_Base 2.67; judge 3 vs 4; issues it saw: ['Cannot verify exact count of stars vs city lights', "Cannot verify 'more visible stars than the number of lights' as counts are not discernible"] vs ['Cannot verify exact count of stars vs. lights due to image resolution and lack of precise enumeration']
- "The sun illuminates the garden, a shovel sticks in the soil, and gloves lay on the bench.": people DALLE_3 4.67 > DeepFloyd_I_XL_v1 2.67; judge 2 vs 3; issues it saw: ['no shovel is sticking in the soil', 'gloves are not on the bench, they are on the wooden surface next to the tools', 'the shovel is not sticking in the soil, it is resting on the soil'] vs ['shovel is not stuck in soil', 'gloves are on a bench, not in the soil', "sun is illuminating the scene but not clearly shown as 'illuminating the garden'"]
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_Turbo (4.67), the judge picks DeepFloyd_I_XL_v1 (1.67) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer Midjourney_6 (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_2_1 (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "A starry night sky with more visible stars than the number of lights in the city below.": people prefer SDXL_Turbo (4.67), the judge picks DALLE_3 (2.00) in both orders
- pairwise, "Some balls are on the table and on the floor, and the number of balls on the table is more than the number of balls on the floor.": people prefer Midjourney_6 (3.67), the judge picks DeepFloyd_I_XL_v1 (1.00) in both orders
- pairwise, "A vintage car show with more convertibles than sedans.": people prefer DeepFloyd_I_XL_v1 (4.67), the judge picks Midjourney_6 (2.33) in both orders
- pairwise, "The sun illuminates the garden, a shovel sticks in the soil, and gloves lay on the bench.": people prefer DALLE_3 (4.67), the judge picks SDXL_Base (2.67) in both orders
- pairwise, "A girl in a smart blazer using a globe to explain continents to a girl in a school uniform and backpack.": people prefer DALLE_3 (5.00), the judge picks DeepFloyd_I_XL_v1 (3.00) in both orders
