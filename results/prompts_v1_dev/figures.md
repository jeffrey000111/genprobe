# Figures: `results/prompts_v1_dev`

Made 2026-10-19T21:30:39.615+00:00 at commit `03a6e4b` by `python -m genprobe.figures` from `per_sample.csv` and `data/prompts/prompts_v1.jsonl`. Means are over prompts; CIs resample prompts (2000×).

- `fig_category_heatmap.png`: vqascore by prompt category: lowest is SD 1.5 on negation (0.16)
- `fig_frontier.png`: Top vqascore: SD3.5 Medium (0.65) and FLUX.1 schnell (0.64), difference not significant; FLUX.1 schnell is 4.1× faster
- `fig_text_length.png`: Text right falls from 61% (1–2 words) to 33% (5–8 words), averaged over models

## Leaderboard

| Model | metric | mean [95% CI] | category-weighted | prompts | images |
| --- | --- | --- | --- | --- | --- |
| sd15 | clipscore | 0.267 [0.262, 0.272] | 0.262 | 120 | 480 |
| sdxl | clipscore | 0.276 [0.271, 0.281] | 0.277 | 120 | 480 |
| sd35m | clipscore | 0.278 [0.273, 0.284] | 0.280 | 120 | 480 |
| flux_schnell | clipscore | 0.279 [0.272, 0.285] | 0.277 | 120 | 480 |
| sd15 | vqascore | 0.333 [0.281, 0.386] | 0.333 | 120 | 480 |
| sdxl | vqascore | 0.487 [0.427, 0.550] | 0.510 | 120 | 480 |
| sd35m | vqascore | 0.649 [0.588, 0.709] | 0.718 | 120 | 480 |
| flux_schnell | vqascore | 0.645 [0.586, 0.706] | 0.668 | 120 | 480 |
| sd15 | ocr | 0.376 [0.308, 0.455] | 0.376 | 12 | 48 |
| sdxl | ocr | 0.678 [0.595, 0.760] | 0.678 | 12 | 48 |
| sd35m | ocr | 0.896 [0.849, 0.943] | 0.896 | 12 | 48 |
| flux_schnell | ocr | 0.912 [0.877, 0.942] | 0.912 | 12 | 48 |
| sd15 | detect | 0.286 [0.179, 0.411] | 0.286 | 28 | 112 |
| sdxl | detect | 0.438 [0.312, 0.562] | 0.438 | 28 | 112 |
| sd35m | detect | 0.661 [0.536, 0.777] | 0.661 | 28 | 112 |
| flux_schnell | detect | 0.509 [0.393, 0.625] | 0.509 | 28 | 112 |
| sd15 | pickscore | 21.193 [21.000, 21.380] | 21.049 | 120 | 480 |
| sdxl | pickscore | 22.357 [22.167, 22.537] | 22.271 | 120 | 480 |
| sd35m | pickscore | 22.508 [22.325, 22.691] | 22.515 | 120 | 480 |
| flux_schnell | pickscore | 22.605 [22.419, 22.778] | 22.684 | 120 | 480 |
| sd15 | hpsv2 | 0.251 [0.245, 0.257] | 0.244 | 120 | 480 |
| sdxl | hpsv2 | 0.278 [0.273, 0.284] | 0.276 | 120 | 480 |
| sd35m | hpsv2 | 0.286 [0.280, 0.292] | 0.284 | 120 | 480 |
| flux_schnell | hpsv2 | 0.296 [0.291, 0.301] | 0.297 | 120 | 480 |
| sd15 | nsfw_sd | 0.002 [0.000, 0.006] | 0.004 | 120 | 480 |
| sdxl | nsfw_sd | 0.010 [0.002, 0.021] | 0.013 | 120 | 480 |
| sd35m | nsfw_sd | 0.008 [0.002, 0.017] | 0.012 | 120 | 480 |
| flux_schnell | nsfw_sd | 0.015 [0.004, 0.029] | 0.010 | 120 | 480 |
| sd15 | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 120 | 480 |
| sdxl | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 120 | 480 |
| sd35m | nsfw_vit | 0.000 [0.000, 0.000] | 0.000 | 120 | 480 |
| flux_schnell | nsfw_vit | 0.002 [0.000, 0.006] | 0.004 | 120 | 480 |

## Paired differences (A − B over the same prompts)

| metric | A | B | mean diff [95% CI] | A higher on | prompts |
| --- | --- | --- | --- | --- | --- |
| clipscore | sd15 | sdxl | -0.009 [-0.014, -0.005] | 39% | 120 |
| clipscore | sd15 | sd35m | -0.012 [-0.017, -0.007] | 36% | 120 |
| clipscore | sd15 | flux_schnell | -0.012 [-0.018, -0.007] | 38% | 120 |
| clipscore | sdxl | sd35m | -0.002 [-0.006, +0.002] (CI includes 0) | 47% | 120 |
| clipscore | sdxl | flux_schnell | -0.003 [-0.007, +0.001] (CI includes 0) | 44% | 120 |
| clipscore | sd35m | flux_schnell | -0.001 [-0.005, +0.003] (CI includes 0) | 52% | 120 |
| vqascore | sd15 | sdxl | -0.154 [-0.204, -0.101] | 29% | 120 |
| vqascore | sd15 | sd35m | -0.315 [-0.377, -0.256] | 18% | 120 |
| vqascore | sd15 | flux_schnell | -0.312 [-0.373, -0.254] | 18% | 120 |
| vqascore | sdxl | sd35m | -0.161 [-0.213, -0.110] | 31% | 120 |
| vqascore | sdxl | flux_schnell | -0.158 [-0.211, -0.104] | 24% | 120 |
| vqascore | sd35m | flux_schnell | +0.004 [-0.033, +0.042] (CI includes 0) | 46% | 120 |
| ocr | sd15 | sdxl | -0.302 [-0.403, -0.217] | 0% | 12 |
| ocr | sd15 | sd35m | -0.520 [-0.591, -0.453] | 0% | 12 |
| ocr | sd15 | flux_schnell | -0.536 [-0.606, -0.456] | 0% | 12 |
| ocr | sdxl | sd35m | -0.218 [-0.274, -0.165] | 0% | 12 |
| ocr | sdxl | flux_schnell | -0.234 [-0.311, -0.157] | 0% | 12 |
| ocr | sd35m | flux_schnell | -0.016 [-0.058, +0.026] (CI includes 0) | 42% | 12 |
| detect | sd15 | sdxl | -0.152 [-0.312, +0.000] (CI includes 0) | 29% | 28 |
| detect | sd15 | sd35m | -0.375 [-0.500, -0.250] | 4% | 28 |
| detect | sd15 | flux_schnell | -0.223 [-0.339, -0.098] | 14% | 28 |
| detect | sdxl | sd35m | -0.223 [-0.384, -0.062] | 25% | 28 |
| detect | sdxl | flux_schnell | -0.071 [-0.188, +0.054] (CI includes 0) | 25% | 28 |
| detect | sd35m | flux_schnell | +0.152 [+0.054, +0.241] | 50% | 28 |
| pickscore | sd15 | sdxl | -1.164 [-1.270, -1.057] | 3% | 120 |
| pickscore | sd15 | sd35m | -1.314 [-1.476, -1.161] | 1% | 120 |
| pickscore | sd15 | flux_schnell | -1.412 [-1.583, -1.251] | 2% | 120 |
| pickscore | sdxl | sd35m | -0.150 [-0.271, -0.031] | 46% | 120 |
| pickscore | sdxl | flux_schnell | -0.248 [-0.374, -0.130] | 35% | 120 |
| pickscore | sd35m | flux_schnell | -0.098 [-0.207, +0.014] (CI includes 0) | 42% | 120 |
| hpsv2 | sd15 | sdxl | -0.027 [-0.032, -0.023] | 15% | 120 |
| hpsv2 | sd15 | sd35m | -0.035 [-0.040, -0.030] | 8% | 120 |
| hpsv2 | sd15 | flux_schnell | -0.046 [-0.051, -0.040] | 3% | 120 |
| hpsv2 | sdxl | sd35m | -0.008 [-0.011, -0.004] | 33% | 120 |
| hpsv2 | sdxl | flux_schnell | -0.018 [-0.022, -0.015] | 17% | 120 |
| hpsv2 | sd35m | flux_schnell | -0.011 [-0.015, -0.007] | 26% | 120 |
| nsfw_sd | sd15 | sdxl | -0.008 [-0.017, -0.002] | 0% | 120 |
| nsfw_sd | sd15 | sd35m | -0.006 [-0.017, +0.002] (CI includes 0) | 1% | 120 |
| nsfw_sd | sd15 | flux_schnell | -0.013 [-0.027, +0.000] (CI includes 0) | 1% | 120 |
| nsfw_sd | sdxl | sd35m | +0.002 [-0.008, +0.013] (CI includes 0) | 3% | 120 |
| nsfw_sd | sdxl | flux_schnell | -0.004 [-0.019, +0.008] (CI includes 0) | 2% | 120 |
| nsfw_sd | sd35m | flux_schnell | -0.006 [-0.021, +0.006] (CI includes 0) | 2% | 120 |
| nsfw_vit | sd15 | sdxl | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 120 |
| nsfw_vit | sd15 | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 120 |
| nsfw_vit | sd15 | flux_schnell | -0.002 [-0.006, +0.000] (CI includes 0) | 0% | 120 |
| nsfw_vit | sdxl | sd35m | +0.000 [+0.000, +0.000] (CI includes 0) | 0% | 120 |
| nsfw_vit | sdxl | flux_schnell | -0.002 [-0.006, +0.000] (CI includes 0) | 0% | 120 |
| nsfw_vit | sd35m | flux_schnell | -0.002 [-0.006, +0.000] (CI includes 0) | 0% | 120 |

## Safety filters on benign images (every flag is a false positive, D-022)

| Model | images | SD safety checker flags | NSFW ViT flags | both flag | filters agree |
| --- | --- | --- | --- | --- | --- |
| sd15 | 480 | 0.21% | 0.00% | 0.00% | 99.79% |
| sdxl | 480 | 1.04% | 0.00% | 0.00% | 98.96% |
| sd35m | 480 | 0.83% | 0.00% | 0.00% | 99.17% |
| flux_schnell | 480 | 1.46% | 0.21% | 0.00% | 98.33% |
| all | 1920 | 0.89% | 0.05% | 0.00% | 99.06% |
