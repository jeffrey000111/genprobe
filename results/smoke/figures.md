# Figures: `results/smoke`

Made 2026-10-19T08:21:42.266+00:00 at commit `9971e8f` by `python -m genprobe.figures` from `per_sample.csv` and `data/prompts/smoke.jsonl`. Means are over prompts; CIs resample prompts (2000×).

- `fig_category_heatmap.png`: vqascore by prompt category: lowest is SD 1.5 on spatial (0.23)
- `fig_frontier.png`: Highest vqascore: FLUX.1 schnell (0.82, 1.13 s/image); cheapest: SD 1.5 (0.40, 0.53 s/image)
- `fig_text_length.png`: Exact text falls from 50% (1–2 words) to 0% (5–8 words), averaged over models

## Leaderboard

| Model | metric | mean [95% CI] | category-weighted | prompts | images |
| --- | --- | --- | --- | --- | --- |
| sd15 | clipscore | 0.245 [0.233, 0.257] | 0.245 | 20 | 20 |
| sdxl | clipscore | 0.258 [0.247, 0.273] | 0.260 | 20 | 20 |
| sd35m | clipscore | 0.271 [0.260, 0.283] | 0.272 | 20 | 20 |
| flux_schnell | clipscore | 0.266 [0.258, 0.274] | 0.266 | 20 | 20 |
| sd15 | vqascore | 0.401 [0.229, 0.585] | 0.399 | 20 | 20 |
| sdxl | vqascore | 0.623 [0.448, 0.793] | 0.630 | 20 | 20 |
| sd35m | vqascore | 0.668 [0.495, 0.828] | 0.683 | 20 | 20 |
| flux_schnell | vqascore | 0.817 [0.665, 0.945] | 0.825 | 20 | 20 |
| sd15 | ocr | 0.329 [0.265, 0.412] | 0.329 | 6 | 6 |
| sdxl | ocr | 0.589 [0.510, 0.679] | 0.589 | 6 | 6 |
| sd35m | ocr | 0.924 [0.841, 0.986] | 0.924 | 6 | 6 |
| flux_schnell | ocr | 0.854 [0.698, 0.990] | 0.854 | 6 | 6 |
| sd15 | detect | 0.286 [0.071, 0.571] | 0.286 | 14 | 14 |
| sdxl | detect | 0.571 [0.286, 0.857] | 0.571 | 14 | 14 |
| sd35m | detect | 0.357 [0.143, 0.643] | 0.357 | 14 | 14 |
| flux_schnell | detect | 0.643 [0.357, 0.857] | 0.643 | 14 | 14 |

## Paired differences (A − B over the same prompts)

| metric | A | B | mean diff [95% CI] | A higher on | prompts |
| --- | --- | --- | --- | --- | --- |
| clipscore | sd15 | sdxl | -0.014 [-0.028, -0.002] | 35% | 20 |
| clipscore | sd15 | sd35m | -0.026 [-0.040, -0.013] | 25% | 20 |
| clipscore | sd15 | flux_schnell | -0.021 [-0.031, -0.011] | 25% | 20 |
| clipscore | sdxl | sd35m | -0.012 [-0.022, -0.003] | 30% | 20 |
| clipscore | sdxl | flux_schnell | -0.007 [-0.017, +0.003] (CI includes 0) | 35% | 20 |
| clipscore | sd35m | flux_schnell | +0.005 [-0.004, +0.015] (CI includes 0) | 55% | 20 |
| vqascore | sd15 | sdxl | -0.222 [-0.414, -0.039] | 25% | 20 |
| vqascore | sd15 | sd35m | -0.267 [-0.481, -0.048] | 15% | 20 |
| vqascore | sd15 | flux_schnell | -0.416 [-0.634, -0.178] | 20% | 20 |
| vqascore | sdxl | sd35m | -0.045 [-0.245, +0.173] (CI includes 0) | 25% | 20 |
| vqascore | sdxl | flux_schnell | -0.194 [-0.362, -0.008] | 20% | 20 |
| vqascore | sd35m | flux_schnell | -0.149 [-0.353, +0.056] (CI includes 0) | 55% | 20 |
| ocr | sd15 | sdxl | -0.260 [-0.342, -0.180] | 0% | 6 |
| ocr | sd15 | sd35m | -0.596 [-0.672, -0.530] | 0% | 6 |
| ocr | sd15 | flux_schnell | -0.525 [-0.667, -0.383] | 0% | 6 |
| ocr | sdxl | sd35m | -0.336 [-0.425, -0.262] | 0% | 6 |
| ocr | sdxl | flux_schnell | -0.265 [-0.415, -0.115] | 0% | 6 |
| ocr | sd35m | flux_schnell | +0.070 [-0.062, +0.214] (CI includes 0) | 33% | 6 |
| detect | sd15 | sdxl | -0.286 [-0.500, -0.071] | 0% | 14 |
| detect | sd15 | sd35m | -0.071 [-0.286, +0.143] (CI includes 0) | 7% | 14 |
| detect | sd15 | flux_schnell | -0.357 [-0.643, -0.143] | 0% | 14 |
| detect | sdxl | sd35m | +0.214 [+0.000, +0.429] (CI includes 0) | 21% | 14 |
| detect | sdxl | flux_schnell | -0.071 [-0.286, +0.143] (CI includes 0) | 7% | 14 |
| detect | sd35m | flux_schnell | -0.286 [-0.500, -0.071] | 0% | 14 |
