# Figures: `results/prompts_v1_test`

Made 2026-10-19T21:30:35.804+00:00 at commit `03a6e4b` by `python -m genprobe.figures` from `per_sample.csv` and `data/prompts/prompts_v1.jsonl`. Means are over prompts; CIs resample prompts (2000×).

- `fig_category_heatmap.png`: vqascore by prompt category: lowest is SD 1.5 on differentiation (0.19)
- `fig_frontier.png`: Top vqascore: SD3.5 Medium (0.68) and FLUX.1 schnell (0.67), difference not significant; FLUX.1 schnell is 4.1× faster
- `fig_text_length.png`: Text right falls from 48% (1–2 words) to 24% (5–8 words), averaged over models

## Leaderboard

| Model | metric | mean [95% CI] | category-weighted | prompts | images |
| --- | --- | --- | --- | --- | --- |
| sd15 | clipscore | 0.270 [0.267, 0.272] | 0.267 | 480 | 1920 |
| sdxl | clipscore | 0.279 [0.277, 0.282] | 0.279 | 480 | 1920 |
| sd35m | clipscore | 0.283 [0.280, 0.286] | 0.283 | 480 | 1920 |
| flux_schnell | clipscore | 0.280 [0.277, 0.284] | 0.279 | 480 | 1920 |
| sd15 | vqascore | 0.365 [0.337, 0.391] | 0.375 | 480 | 1920 |
| sdxl | vqascore | 0.505 [0.474, 0.535] | 0.541 | 480 | 1920 |
| sd35m | vqascore | 0.680 [0.649, 0.709] | 0.733 | 480 | 1920 |
| flux_schnell | vqascore | 0.670 [0.640, 0.702] | 0.702 | 480 | 1920 |
| sd15 | ocr | 0.425 [0.379, 0.471] | 0.425 | 48 | 192 |
| sdxl | ocr | 0.664 [0.624, 0.706] | 0.664 | 48 | 192 |
| sd35m | ocr | 0.855 [0.824, 0.882] | 0.855 | 48 | 192 |
| flux_schnell | ocr | 0.867 [0.833, 0.898] | 0.867 | 48 | 192 |
| sd15 | detect | 0.359 [0.295, 0.426] | 0.359 | 112 | 448 |
| sdxl | detect | 0.438 [0.368, 0.509] | 0.438 | 112 | 448 |
| sd35m | detect | 0.641 [0.569, 0.708] | 0.641 | 112 | 448 |
| flux_schnell | detect | 0.583 [0.513, 0.645] | 0.583 | 112 | 448 |
| sd15 | pickscore | 21.151 [21.064, 21.238] | 21.044 | 480 | 1920 |
| sdxl | pickscore | 22.384 [22.283, 22.483] | 22.291 | 480 | 1920 |
| sd35m | pickscore | 22.557 [22.463, 22.656] | 22.571 | 480 | 1920 |
| flux_schnell | pickscore | 22.654 [22.564, 22.745] | 22.710 | 480 | 1920 |
| sd15 | hpsv2 | 0.249 [0.246, 0.251] | 0.246 | 480 | 1920 |
| sdxl | hpsv2 | 0.278 [0.275, 0.281] | 0.277 | 480 | 1920 |
| sd35m | hpsv2 | 0.286 [0.283, 0.289] | 0.285 | 480 | 1920 |
| flux_schnell | hpsv2 | 0.296 [0.293, 0.298] | 0.296 | 480 | 1920 |
| sd15 | nsfw_sd | 0.004 [0.002, 0.007] | 0.004 | 480 | 1920 |
| sdxl | nsfw_sd | 0.003 [0.001, 0.006] | 0.002 | 480 | 1920 |
| sd35m | nsfw_sd | 0.006 [0.003, 0.010] | 0.006 | 480 | 1920 |
| flux_schnell | nsfw_sd | 0.006 [0.002, 0.011] | 0.007 | 480 | 1920 |
| sd15 | nsfw_vit | 0.001 [0.000, 0.002] | 0.000 | 480 | 1920 |
| sdxl | nsfw_vit | 0.004 [0.001, 0.008] | 0.002 | 480 | 1920 |
| sd35m | nsfw_vit | 0.003 [0.000, 0.007] | 0.002 | 480 | 1920 |
| flux_schnell | nsfw_vit | 0.001 [0.000, 0.003] | 0.001 | 480 | 1920 |

## Paired differences (A − B over the same prompts)

| metric | A | B | mean diff [95% CI] | A higher on | prompts |
| --- | --- | --- | --- | --- | --- |
| clipscore | sd15 | sdxl | -0.010 [-0.012, -0.007] | 39% | 480 |
| clipscore | sd15 | sd35m | -0.013 [-0.016, -0.011] | 34% | 480 |
| clipscore | sd15 | flux_schnell | -0.011 [-0.013, -0.008] | 39% | 480 |
| clipscore | sdxl | sd35m | -0.003 [-0.005, -0.002] | 42% | 480 |
| clipscore | sdxl | flux_schnell | -0.001 [-0.003, +0.001] (CI includes 0) | 50% | 480 |
| clipscore | sd35m | flux_schnell | +0.003 [+0.001, +0.004] | 56% | 480 |
| vqascore | sd15 | sdxl | -0.140 [-0.166, -0.116] | 32% | 480 |
| vqascore | sd15 | sd35m | -0.315 [-0.344, -0.286] | 16% | 480 |
| vqascore | sd15 | flux_schnell | -0.305 [-0.339, -0.274] | 17% | 480 |
| vqascore | sdxl | sd35m | -0.174 [-0.200, -0.150] | 24% | 480 |
| vqascore | sdxl | flux_schnell | -0.165 [-0.194, -0.136] | 26% | 480 |
| vqascore | sd35m | flux_schnell | +0.010 [-0.015, +0.033] (CI includes 0) | 48% | 480 |
| ocr | sd15 | sdxl | -0.239 [-0.286, -0.191] | 6% | 48 |
| ocr | sd15 | sd35m | -0.429 [-0.476, -0.376] | 4% | 48 |
| ocr | sd15 | flux_schnell | -0.442 [-0.494, -0.387] | 2% | 48 |
| ocr | sdxl | sd35m | -0.190 [-0.226, -0.152] | 4% | 48 |
| ocr | sdxl | flux_schnell | -0.203 [-0.246, -0.156] | 8% | 48 |
| ocr | sd35m | flux_schnell | -0.013 [-0.056, +0.030] (CI includes 0) | 35% | 48 |
| detect | sd15 | sdxl | -0.078 [-0.136, -0.020] | 26% | 112 |
| detect | sd15 | sd35m | -0.281 [-0.346, -0.214] | 11% | 112 |
| detect | sd15 | flux_schnell | -0.223 [-0.292, -0.152] | 17% | 112 |
| detect | sdxl | sd35m | -0.203 [-0.270, -0.134] | 12% | 112 |
| detect | sdxl | flux_schnell | -0.145 [-0.214, -0.074] | 20% | 112 |
| detect | sd35m | flux_schnell | +0.058 [-0.009, +0.123] (CI includes 0) | 38% | 112 |
| pickscore | sd15 | sdxl | -1.232 [-1.305, -1.158] | 4% | 480 |
| pickscore | sd15 | sd35m | -1.405 [-1.494, -1.322] | 2% | 480 |
| pickscore | sd15 | flux_schnell | -1.503 [-1.594, -1.421] | 2% | 480 |
| pickscore | sdxl | sd35m | -0.173 [-0.236, -0.113] | 43% | 480 |
| pickscore | sdxl | flux_schnell | -0.271 [-0.344, -0.204] | 36% | 480 |
| pickscore | sd35m | flux_schnell | -0.098 [-0.155, -0.043] | 40% | 480 |
| hpsv2 | sd15 | sdxl | -0.029 [-0.032, -0.027] | 15% | 480 |
| hpsv2 | sd15 | sd35m | -0.037 [-0.039, -0.034] | 9% | 480 |
| hpsv2 | sd15 | flux_schnell | -0.047 [-0.049, -0.044] | 4% | 480 |
| hpsv2 | sdxl | sd35m | -0.008 [-0.010, -0.006] | 36% | 480 |
| hpsv2 | sdxl | flux_schnell | -0.017 [-0.020, -0.015] | 21% | 480 |
| hpsv2 | sd35m | flux_schnell | -0.010 [-0.012, -0.008] | 31% | 480 |
| nsfw_sd | sd15 | sdxl | +0.002 [-0.002, +0.005] (CI includes 0) | 1% | 480 |
| nsfw_sd | sd15 | sd35m | -0.002 [-0.006, +0.002] (CI includes 0) | 1% | 480 |
| nsfw_sd | sd15 | flux_schnell | -0.002 [-0.007, +0.003] (CI includes 0) | 1% | 480 |
| nsfw_sd | sdxl | sd35m | -0.004 [-0.007, +0.000] (CI includes 0) | 0% | 480 |
| nsfw_sd | sdxl | flux_schnell | -0.003 [-0.007, +0.001] (CI includes 0) | 1% | 480 |
| nsfw_sd | sd35m | flux_schnell | +0.001 [-0.005, +0.006] (CI includes 0) | 2% | 480 |
| nsfw_vit | sd15 | sdxl | -0.003 [-0.006, -0.001] | 0% | 480 |
| nsfw_vit | sd15 | sd35m | -0.002 [-0.006, +0.000] (CI includes 0) | 0% | 480 |
| nsfw_vit | sd15 | flux_schnell | -0.001 [-0.003, +0.001] (CI includes 0) | 0% | 480 |
| nsfw_vit | sdxl | sd35m | +0.001 [-0.002, +0.004] (CI includes 0) | 1% | 480 |
| nsfw_vit | sdxl | flux_schnell | +0.003 [-0.001, +0.007] (CI includes 0) | 1% | 480 |
| nsfw_vit | sd35m | flux_schnell | +0.002 [-0.002, +0.007] (CI includes 0) | 0% | 480 |

## Safety filters on benign images (every flag is a false positive, D-022)

| Model | images | SD safety checker flags | NSFW ViT flags | both flag | filters agree |
| --- | --- | --- | --- | --- | --- |
| sd15 | 1920 | 0.42% | 0.05% | 0.00% | 99.53% |
| sdxl | 1920 | 0.26% | 0.36% | 0.00% | 99.38% |
| sd35m | 1920 | 0.62% | 0.26% | 0.00% | 99.11% |
| flux_schnell | 1920 | 0.57% | 0.10% | 0.00% | 99.32% |
| all | 7680 | 0.47% | 0.20% | 0.00% | 99.34% |
