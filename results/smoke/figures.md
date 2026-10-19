# Figures: `results/smoke`

Made 2026-10-19T19:31:41.443+00:00 at commit `124e661` by `python -m genprobe.figures` from `per_sample.csv` and `data/prompts/smoke.jsonl`. Means are over prompts; CIs resample prompts (2000×).

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
| sd15 | pickscore | 21.026 [20.404, 21.610] | 20.954 | 20 | 20 |
| sdxl | pickscore | 22.079 [21.709, 22.461] | 22.064 | 20 | 20 |
| sd35m | pickscore | 22.311 [21.947, 22.639] | 22.337 | 20 | 20 |
| flux_schnell | pickscore | 22.921 [22.530, 23.307] | 22.933 | 20 | 20 |
| sd15 | hpsv2 | 0.245 [0.223, 0.266] | 0.243 | 20 | 20 |
| sdxl | hpsv2 | 0.267 [0.255, 0.278] | 0.265 | 20 | 20 |
| sd35m | hpsv2 | 0.278 [0.269, 0.287] | 0.277 | 20 | 20 |
| flux_schnell | hpsv2 | 0.300 [0.288, 0.312] | 0.300 | 20 | 20 |
| sd15 | nsfw_sd | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| sdxl | nsfw_sd | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| sd35m | nsfw_sd | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| flux_schnell | nsfw_sd | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| sd15 | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| sdxl | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| sd35m | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |
| flux_schnell | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 20 | 20 |

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
| pickscore | sd15 | sdxl | -1.053 [-1.587, -0.525] | 25% | 20 |
| pickscore | sd15 | sd35m | -1.284 [-2.007, -0.614] | 30% | 20 |
| pickscore | sd15 | flux_schnell | -1.895 [-2.629, -1.234] | 10% | 20 |
| pickscore | sdxl | sd35m | -0.232 [-0.713, +0.221] (CI includes 0) | 50% | 20 |
| pickscore | sdxl | flux_schnell | -0.842 [-1.275, -0.403] | 20% | 20 |
| pickscore | sd35m | flux_schnell | -0.610 [-0.905, -0.317] | 20% | 20 |
| hpsv2 | sd15 | sdxl | -0.021 [-0.038, -0.004] | 25% | 20 |
| hpsv2 | sd15 | sd35m | -0.032 [-0.052, -0.014] | 25% | 20 |
| hpsv2 | sd15 | flux_schnell | -0.055 [-0.076, -0.034] | 15% | 20 |
| hpsv2 | sdxl | sd35m | -0.011 [-0.022, -0.001] | 35% | 20 |
| hpsv2 | sdxl | flux_schnell | -0.034 [-0.045, -0.022] | 10% | 20 |
| hpsv2 | sd35m | flux_schnell | -0.022 [-0.033, -0.011] | 10% | 20 |
| nsfw_sd | sd15 | sdxl | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_sd | sd15 | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_sd | sd15 | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_sd | sdxl | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_sd | sdxl | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_sd | sd35m | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sd15 | sdxl | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sd15 | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sd15 | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sdxl | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sdxl | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |
| nsfw_vit | sd35m | flux_schnell | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 20 |

## Safety filters on benign images (every flag is a false positive, D-022)

| Model | images | SD safety checker flags | NSFW ViT flags | both flag | filters agree |
| --- | --- | --- | --- | --- | --- |
| sd15 | 20 | 0.00% | 0.00% | 0.00% | 100.00% |
| sdxl | 20 | 0.00% | 0.00% | 0.00% | 100.00% |
| sd35m | 20 | 0.00% | 0.00% | 0.00% | 100.00% |
| flux_schnell | 20 | 0.00% | 0.00% | 0.00% | 100.00% |
| all | 80 | 0.00% | 0.00% | 0.00% | 100.00% |
