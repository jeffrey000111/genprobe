# Summary: `data/prompts/smoke.jsonl` (split: all)

Generated 2026-10-19T19:31:37.568+00:00 at commit `124e661` by `python -m genprobe.summarize`. 20 prompts; means are over prompts, 95% CIs from 2000 bootstrap resamples of prompts.

## Generation coverage

| Model | expected images | ok | blank | error | no record |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 20 | 20 | 0 | 0 | 0 |
| sd15 | 20 | 20 | 0 | 0 | 0 |
| sd35m | 20 | 20 | 0 | 0 | 0 |
| sdxl | 20 | 20 | 0 | 0 | 0 |

## Metric scores by model

| Model | clipscore | vqascore | ocr | detect | pickscore | hpsv2 | nsfw_sd | nsfw_vit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.266 [0.258, 0.274] (n=20) | 0.817 [0.665, 0.945] (n=20) | 0.854 [0.698, 0.990] (n=6) | 0.643 [0.357, 0.857] (n=14) | 22.921 [22.530, 23.307] (n=20) | 0.300 [0.288, 0.312] (n=20) | 0.000 [0.000, 0.000] (n=20) | 0.000 [0.000, 0.000] (n=20) |
| sd15 | 0.245 [0.233, 0.257] (n=20) | 0.401 [0.229, 0.585] (n=20) | 0.329 [0.265, 0.412] (n=6) | 0.286 [0.071, 0.571] (n=14) | 21.026 [20.404, 21.610] (n=20) | 0.245 [0.223, 0.266] (n=20) | 0.000 [0.000, 0.000] (n=20) | 0.000 [0.000, 0.000] (n=20) |
| sd35m | 0.271 [0.260, 0.283] (n=20) | 0.668 [0.495, 0.828] (n=20) | 0.924 [0.841, 0.986] (n=6) | 0.357 [0.143, 0.643] (n=14) | 22.311 [21.947, 22.639] (n=20) | 0.278 [0.269, 0.287] (n=20) | 0.000 [0.000, 0.000] (n=20) | 0.000 [0.000, 0.000] (n=20) |
| sdxl | 0.258 [0.247, 0.273] (n=20) | 0.623 [0.448, 0.793] (n=20) | 0.589 [0.510, 0.679] (n=6) | 0.571 [0.286, 0.857] (n=14) | 22.079 [21.709, 22.461] (n=20) | 0.267 [0.255, 0.278] (n=20) | 0.000 [0.000, 0.000] (n=20) | 0.000 [0.000, 0.000] (n=20) |

### clipscore by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.255 (7) | 0.260 (7) | 0.284 (6) |
| sd15 | 0.247 (7) | 0.231 (7) | 0.258 (6) |
| sd35m | 0.260 (7) | 0.257 (7) | 0.301 (6) |
| sdxl | 0.250 (7) | 0.243 (7) | 0.286 (6) |

### vqascore by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.704 (7) | 0.790 (7) | 0.979 (6) |
| sd15 | 0.625 (7) | 0.225 (7) | 0.346 (6) |
| sd35m | 0.491 (7) | 0.567 (7) | 0.992 (6) |
| sdxl | 0.603 (7) | 0.517 (7) | 0.770 (6) |

### ocr by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | – (0) | – (0) | 0.854 (6) |
| sd15 | – (0) | – (0) | 0.329 (6) |
| sd35m | – (0) | – (0) | 0.924 (6) |
| sdxl | – (0) | – (0) | 0.589 (6) |

### detect by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.429 (7) | 0.857 (7) | – (0) |
| sd15 | 0.286 (7) | 0.286 (7) | – (0) |
| sd35m | 0.143 (7) | 0.571 (7) | – (0) |
| sdxl | 0.429 (7) | 0.714 (7) | – (0) |

### pickscore by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 22.697 (7) | 22.933 (7) | 23.169 (6) |
| sd15 | 21.753 (7) | 21.597 (7) | 19.512 (6) |
| sd35m | 22.164 (7) | 21.979 (7) | 22.869 (6) |
| sdxl | 22.541 (7) | 21.878 (7) | 21.775 (6) |

### hpsv2 by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.290 (7) | 0.321 (7) | 0.289 (6) |
| sd15 | 0.272 (7) | 0.264 (7) | 0.192 (6) |
| sd35m | 0.281 (7) | 0.285 (7) | 0.267 (6) |
| sdxl | 0.275 (7) | 0.280 (7) | 0.241 (6) |

### nsfw_sd by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sd15 | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sd35m | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sdxl | 0.000 (7) | 0.000 (7) | 0.000 (6) |

### nsfw_vit by category (mean, n images)

| Model | counting | spatial | text |
| --- | --- | --- | --- |
| flux_schnell | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sd15 | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sd35m | 0.000 (7) | 0.000 (7) | 0.000 (6) |
| sdxl | 0.000 (7) | 0.000 (7) | 0.000 (6) |

## Metric sanity checks

| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` | 80 / 80 | 0 | 0.197 | 0.335 | 0.027 | 0 |
| vqascore | `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8` | 80 / 80 | 0 | 0.000 | 1.000 | 0.407 | 0 |
| ocr | `easyocr==1.7.2/craft+english_g2/semiglobal-nospace-v1/bs16` | 24 / 24 | 0 | 0.240 | 1.000 | 0.268 | 0 |
| detect | `transformers==4.49.0/grounding-dino-tiny@a2bb814/box0.35-text0.25-nms0.5-group0.9-margin0.05/bs16` | 56 / 56 | 0 | 0.000 | 1.000 | 0.499 | 0 |
| pickscore | `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16` | 80 / 80 | 0 | 18.513 | 24.475 | 1.202 | 0 |
| hpsv2 | `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16` | 80 / 80 | 0 | 0.153 | 0.351 | 0.038 | 0 |
| nsfw_sd | `diffusers==0.39.0/sd-safety-checker@cb41f3a/fp32/bs32` | 80 / 80 | 0 | 0.000 | 0.000 | 0.000 | 0 |
| nsfw_vit | `transformers==4.49.0/falconsai-nsfw-vit@96cb0d0/fp32/argmax/bs32` | 80 / 80 | 0 | 0.000 | 0.000 | 0.000 | 0 |

Rank agreement between metrics over the same images (Spearman). This says how often the metrics agree with each other, not which one agrees with people.

- clipscore~vqascore: ρ = 0.229 (n = 80)
- clipscore~ocr: ρ = 0.430 (n = 24)
- clipscore~detect: ρ = 0.016 (n = 56)
- clipscore~pickscore: ρ = 0.232 (n = 80)
- clipscore~hpsv2: ρ = 0.103 (n = 80)
- clipscore~nsfw_sd: ρ = – (n = 80)
- clipscore~nsfw_vit: ρ = – (n = 80)
- vqascore~ocr: ρ = 0.873 (n = 24)
- vqascore~detect: ρ = 0.716 (n = 56)
- vqascore~pickscore: ρ = 0.534 (n = 80)
- vqascore~hpsv2: ρ = 0.448 (n = 80)
- vqascore~nsfw_sd: ρ = – (n = 80)
- vqascore~nsfw_vit: ρ = – (n = 80)
- ocr~detect: ρ = – (n = 0)
- ocr~pickscore: ρ = 0.913 (n = 24)
- ocr~hpsv2: ρ = 0.773 (n = 24)
- ocr~nsfw_sd: ρ = – (n = 24)
- ocr~nsfw_vit: ρ = – (n = 24)
- detect~pickscore: ρ = 0.476 (n = 56)
- detect~hpsv2: ρ = 0.496 (n = 56)
- detect~nsfw_sd: ρ = – (n = 56)
- detect~nsfw_vit: ρ = – (n = 56)
- pickscore~hpsv2: ρ = 0.689 (n = 80)
- pickscore~nsfw_sd: ρ = – (n = 80)
- pickscore~nsfw_vit: ρ = – (n = 80)
- hpsv2~nsfw_sd: ρ = – (n = 80)
- hpsv2~nsfw_vit: ρ = – (n = 80)
- nsfw_sd~nsfw_vit: ρ = – (n = 80)

## Agreement with human labels

Labels: `data/labels/smoke_labels.csv` (?=11, n=38, y=31; `?` excluded). Within each prompt, every pair of one image labelled correct and one labelled wrong: how often the metric scores the correct one higher (ties = 1/2). 95% CI from resampling prompts. One annotator, small sample: a sanity check, not a result (D-010, D-015).

| Metric | all | counting | spatial | text | mean score if correct / wrong |
| --- | --- | --- | --- | --- | --- |
| clipscore | 0.73 [0.56, 0.89] (33 pairs) | 0.62 [0.33, 1.00] (8 pairs) | 0.92 [0.75, 1.00] (12 pairs) | 0.62 [0.38, 0.92] (13 pairs) | 0.265 / 0.257 |
| vqascore | 0.97 [0.90, 1.00] (33 pairs) | 0.88 [0.67, 1.00] (8 pairs) | 1.00 [1.00, 1.00] (12 pairs) | 1.00 [1.00, 1.00] (13 pairs) | 0.904 / 0.438 |
| ocr | 1.00 [1.00, 1.00] (13 pairs) | – | – | 1.00 [1.00, 1.00] (13 pairs) | 0.960 / 0.505 |
| detect | 0.85 [0.70, 0.97] (20 pairs) | 1.00 [1.00, 1.00] (8 pairs) | 0.75 [0.57, 0.95] (12 pairs) | – | 0.909 / 0.111 |
| pickscore | 0.91 [0.78, 1.00] (33 pairs) | 1.00 [1.00, 1.00] (8 pairs) | 0.92 [0.80, 1.00] (12 pairs) | 0.85 [0.54, 1.00] (13 pairs) | 22.989 / 21.499 |
| hpsv2 | 0.85 [0.68, 1.00] (33 pairs) | 0.62 [0.33, 1.00] (8 pairs) | 1.00 [1.00, 1.00] (12 pairs) | 0.85 [0.54, 1.00] (13 pairs) | 0.297 / 0.256 |
| nsfw_sd | 0.50 [0.50, 0.50] (33 pairs) | 0.50 [0.50, 0.50] (8 pairs) | 0.50 [0.50, 0.50] (12 pairs) | 0.50 [0.50, 0.50] (13 pairs) | 0.000 / 0.000 |
| nsfw_vit | 0.50 [0.50, 0.50] (33 pairs) | 0.50 [0.50, 0.50] (8 pairs) | 0.50 [0.50, 0.50] (12 pairs) | 0.50 [0.50, 0.50] (13 pairs) | 0.000 / 0.000 |

Checker verdicts against the same labels, image by image (an image passes when the checker's value ≥ its threshold). Accuracy CI from resampling prompts.

| Checker | threshold | images | accuracy | passes & correct | passes & wrong | fails & correct | fails & wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ocr | 1.0 | 20 | 0.85 [0.71, 0.95] | 6 | 0 | 3 | 11 |
| detect | 1.0 | 49 | 0.90 [0.80, 0.98] | 20 | 3 | 2 | 24 |

Largest clipscore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sd35m__sm002__s0` > `sd15__sm002__s0` by 0.0338
- `sd15__sm016__s0` > `sd35m__sm016__s0` by 0.032
- `sdxl__sm019__s0` > `sd35m__sm019__s0` by 0.0242
- `sd15__sm016__s0` > `flux_schnell__sm016__s0` by 0.0127
- `sdxl__sm015__s0` > `flux_schnell__sm015__s0` by 0.0109

Largest vqascore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sd35m__sm005__s0` > `sdxl__sm005__s0` by 0.005

Largest pickscore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sdxl__sm011__s0` > `sd35m__sm011__s0` by 0.2661
- `flux_schnell__sm019__s0` > `sd35m__sm019__s0` by 0.1208
- `sdxl__sm019__s0` > `sd35m__sm019__s0` by 0.1154

Largest hpsv2 mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sd35m__sm002__s0` > `flux_schnell__sm002__s0` by 0.0336
- `sd35m__sm002__s0` > `sdxl__sm002__s0` by 0.0201
- `flux_schnell__sm019__s0` > `sd35m__sm019__s0` by 0.0179
- `sdxl__sm019__s0` > `sd35m__sm019__s0` by 0.0112
- `sd35m__sm005__s0` > `sdxl__sm005__s0` by 0.0059

## Generation cost

Median seconds per image from the manifests, warmup batches excluded (D-004). GPU-hours = median × 2400 images / 3600.

| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for 2400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | NVIDIA H100 NVL | 1 | 4 | 20 (19) | 1.13 | 36.3 | 0 | 0 | 0.75 |
| sd15 | NVIDIA H100 NVL | 4 | 50 | 20 (16) | 0.53 | 4.3 | 0 | 0 | 0.35 |
| sd35m | NVIDIA H100 NVL | 2 | 40 | 20 (18) | 4.65 | 20.7 | 0 | 0 | 3.10 |
| sdxl | NVIDIA H100 NVL | 2 | 30 | 20 (18) | 1.90 | 15.8 | 0 | 0 | 1.27 |
