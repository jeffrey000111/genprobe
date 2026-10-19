# Summary: `data/prompts/prompts_v1.jsonl` (split: test)

Generated 2026-10-19T21:17:11.156+00:00 at commit `41a867f` by `python -m genprobe.summarize`. 480 prompts; means are over prompts, 95% CIs from 2000 bootstrap resamples of prompts.

## Generation coverage

| Model | expected images | ok | blank | error | no record |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 1920 | 1920 | 0 | 0 | 0 |
| sd15 | 1920 | 1920 | 0 | 0 | 0 |
| sd35m | 1920 | 1920 | 0 | 0 | 0 |
| sdxl | 1920 | 1920 | 0 | 0 | 0 |

## Metric scores by model

| Model | clipscore | vqascore | ocr | detect | pickscore | hpsv2 | nsfw_sd | nsfw_vit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.280 [0.277, 0.284] (n=1920) | 0.670 [0.640, 0.702] (n=1920) | 0.867 [0.834, 0.897] (n=192) | 0.583 [0.516, 0.647] (n=448) | 22.654 [22.564, 22.745] (n=1920) | 0.296 [0.293, 0.298] (n=1920) | 0.006 [0.002, 0.011] (n=1920) | 0.001 [0.000, 0.003] (n=1920) |
| sd15 | 0.270 [0.267, 0.272] (n=1920) | 0.365 [0.337, 0.391] (n=1920) | 0.425 [0.379, 0.471] (n=192) | 0.359 [0.295, 0.426] (n=448) | 21.151 [21.064, 21.238] (n=1920) | 0.249 [0.246, 0.251] (n=1920) | 0.004 [0.002, 0.007] (n=1920) | 0.001 [0.000, 0.002] (n=1920) |
| sd35m | 0.283 [0.280, 0.286] (n=1920) | 0.680 [0.649, 0.709] (n=1920) | 0.855 [0.824, 0.882] (n=192) | 0.641 [0.569, 0.708] (n=448) | 22.557 [22.463, 22.656] (n=1920) | 0.286 [0.283, 0.289] (n=1920) | 0.006 [0.003, 0.010] (n=1920) | 0.003 [0.000, 0.007] (n=1920) |
| sdxl | 0.279 [0.277, 0.282] (n=1920) | 0.505 [0.474, 0.535] (n=1920) | 0.664 [0.624, 0.706] (n=192) | 0.438 [0.368, 0.509] (n=448) | 22.384 [22.283, 22.483] (n=1920) | 0.278 [0.275, 0.281] (n=1920) | 0.003 [0.001, 0.006] (n=1920) | 0.004 [0.001, 0.008] (n=1920) |

### clipscore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.276 (776) | 0.299 (504) | 0.244 (224) | 0.275 (224) | 0.300 (192) |
| sd15 | 0.268 (776) | 0.285 (504) | 0.247 (224) | 0.254 (224) | 0.282 (192) |
| sd35m | 0.276 (776) | 0.299 (504) | 0.253 (224) | 0.277 (224) | 0.311 (192) |
| sdxl | 0.274 (776) | 0.294 (504) | 0.253 (224) | 0.267 (224) | 0.309 (192) |

### vqascore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.513 (776) | 0.877 (504) | 0.666 (224) | 0.558 (224) | 0.897 (192) |
| sd15 | 0.279 (776) | 0.489 (504) | 0.459 (224) | 0.298 (224) | 0.350 (192) |
| sd35m | 0.488 (776) | 0.867 (504) | 0.703 (224) | 0.700 (224) | 0.909 (192) |
| sdxl | 0.351 (776) | 0.699 (504) | 0.427 (224) | 0.443 (224) | 0.783 (192) |

### ocr by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | – (0) | – (0) | – (0) | – (0) | 0.867 (192) |
| sd15 | – (0) | – (0) | – (0) | – (0) | 0.425 (192) |
| sd35m | – (0) | – (0) | – (0) | – (0) | 0.855 (192) |
| sdxl | – (0) | – (0) | – (0) | – (0) | 0.664 (192) |

### detect by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | – (0) | – (0) | 0.567 (224) | 0.598 (224) | – (0) |
| sd15 | – (0) | – (0) | 0.379 (224) | 0.339 (224) | – (0) |
| sd35m | – (0) | – (0) | 0.558 (224) | 0.723 (224) | – (0) |
| sdxl | – (0) | – (0) | 0.366 (224) | 0.509 (224) | – (0) |

### pickscore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 22.362 (776) | 23.031 (504) | 22.467 (224) | 22.835 (224) | 22.856 (192) |
| sd15 | 21.248 (776) | 21.240 (504) | 21.416 (224) | 21.467 (224) | 19.850 (192) |
| sd35m | 22.328 (776) | 22.955 (504) | 22.485 (224) | 22.493 (224) | 22.592 (192) |
| sdxl | 22.288 (776) | 22.848 (504) | 21.980 (224) | 22.774 (224) | 21.566 (192) |

### hpsv2 by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.290 (776) | 0.303 (504) | 0.296 (224) | 0.309 (224) | 0.282 (192) |
| sd15 | 0.250 (776) | 0.255 (504) | 0.258 (224) | 0.252 (224) | 0.218 (192) |
| sd35m | 0.281 (776) | 0.299 (504) | 0.284 (224) | 0.288 (224) | 0.271 (192) |
| sdxl | 0.275 (776) | 0.287 (504) | 0.272 (224) | 0.295 (224) | 0.254 (192) |

### nsfw_sd by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.005 (776) | 0.004 (504) | 0.009 (224) | 0.000 (224) | 0.016 (192) |
| sd15 | 0.005 (776) | 0.002 (504) | 0.000 (224) | 0.004 (224) | 0.010 (192) |
| sd35m | 0.006 (776) | 0.006 (504) | 0.009 (224) | 0.000 (224) | 0.010 (192) |
| sdxl | 0.005 (776) | 0.000 (504) | 0.000 (224) | 0.000 (224) | 0.005 (192) |

### nsfw_vit by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.003 (776) | 0.000 (504) | 0.000 (224) | 0.000 (224) | 0.000 (192) |
| sd15 | 0.000 (776) | 0.002 (504) | 0.000 (224) | 0.000 (224) | 0.000 (192) |
| sd35m | 0.001 (776) | 0.008 (504) | 0.000 (224) | 0.000 (224) | 0.000 (192) |
| sdxl | 0.005 (776) | 0.006 (504) | 0.000 (224) | 0.000 (224) | 0.000 (192) |

## Metric sanity checks

| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` | 7680 / 7680 | 0 | 0.138 | 0.438 | 0.037 | 0 |
| vqascore | `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8` | 7680 / 7680 | 0 | 0.000 | 1.000 | 0.424 | 0 |
| ocr | `easyocr==1.7.2/craft+english_g2/semiglobal-nospace-v1/bs16` | 768 / 768 | 0 | 0.000 | 1.000 | 0.263 | 0 |
| detect | `transformers==4.49.0/grounding-dino-tiny@a2bb814/box0.35-text0.25-nms0.5-group0.9-margin0.05/bs16` | 1792 / 1792 | 0 | 0.000 | 1.000 | 0.500 | 0 |
| pickscore | `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16` | 7680 / 7680 | 0 | 17.399 | 27.350 | 1.308 | 0 |
| hpsv2 | `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16` | 7680 / 7680 | 0 | 0.141 | 0.412 | 0.039 | 0 |
| nsfw_sd | `diffusers==0.39.0/sd-safety-checker@cb41f3a/fp32/bs32` | 7680 / 7680 | 0 | 0.000 | 1.000 | 0.068 | 0 |
| nsfw_vit | `transformers==4.49.0/falconsai-nsfw-vit@96cb0d0/fp32/argmax/bs32` | 7680 / 7680 | 0 | 0.000 | 1.000 | 0.044 | 0 |

Rank agreement between metrics over the same images (Spearman). This says how often the metrics agree with each other, not which one agrees with people.

- clipscore~vqascore: ρ = 0.273 (n = 7680)
- clipscore~ocr: ρ = 0.159 (n = 768)
- clipscore~detect: ρ = 0.136 (n = 1792)
- clipscore~pickscore: ρ = 0.268 (n = 7680)
- clipscore~hpsv2: ρ = 0.280 (n = 7680)
- clipscore~nsfw_sd: ρ = 0.021 (n = 7680)
- clipscore~nsfw_vit: ρ = 0.001 (n = 7680)
- vqascore~ocr: ρ = 0.677 (n = 768)
- vqascore~detect: ρ = 0.700 (n = 1792)
- vqascore~pickscore: ρ = 0.419 (n = 7680)
- vqascore~hpsv2: ρ = 0.356 (n = 7680)
- vqascore~nsfw_sd: ρ = -0.008 (n = 7680)
- vqascore~nsfw_vit: ρ = 0.007 (n = 7680)
- ocr~detect: ρ = – (n = 0)
- ocr~pickscore: ρ = 0.643 (n = 768)
- ocr~hpsv2: ρ = 0.594 (n = 768)
- ocr~nsfw_sd: ρ = 0.022 (n = 768)
- ocr~nsfw_vit: ρ = – (n = 768)
- detect~pickscore: ρ = 0.349 (n = 1792)
- detect~hpsv2: ρ = 0.276 (n = 1792)
- detect~nsfw_sd: ρ = 0.052 (n = 1792)
- detect~nsfw_vit: ρ = – (n = 1792)
- pickscore~hpsv2: ρ = 0.678 (n = 7680)
- pickscore~nsfw_sd: ρ = -0.004 (n = 7680)
- pickscore~nsfw_vit: ρ = 0.019 (n = 7680)
- hpsv2~nsfw_sd: ρ = -0.017 (n = 7680)
- hpsv2~nsfw_vit: ρ = -0.002 (n = 7680)
- nsfw_sd~nsfw_vit: ρ = -0.003 (n = 7680)

## Generation cost

Median seconds per image from the manifests, warmup batches excluded (D-004). GPU-hours = median × 2400 images / 3600.

| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for 2400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | NVIDIA H100 NVL | 1 | 4 | 1920 (1915) | 1.13 | 36.3 | 0 | 0 | 0.75 |
| sd15 | NVIDIA H100 NVL | 4 | 50 | 1920 (1904) | 0.53 | 4.3 | 0 | 0 | 0.35 |
| sd35m | NVIDIA H100 NVL | 2 | 40 | 1920 (1912) | 4.60 | 20.7 | 0 | 0 | 3.07 |
| sdxl | NVIDIA H100 NVL | 2 | 50 | 1920 (1912) | 3.05 | 15.8 | 0 | 0 | 2.03 |
