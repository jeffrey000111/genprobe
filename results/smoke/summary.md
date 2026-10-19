# Summary: `data/prompts/smoke.jsonl` (split: all)

Generated 2026-10-19T08:08:53.975+00:00 at commit `9546265` by `python -m genprobe.summarize`. 20 prompts; means are over prompts, 95% CIs from 2000 bootstrap resamples of prompts.

## Metric scores by model

| Model | clipscore | vqascore | ocr | detect |
| --- | --- | --- | --- | --- |
| flux_schnell | 0.266 [0.258, 0.274] (n=20) | 0.817 [0.665, 0.945] (n=20) | 0.854 [0.698, 0.990] (n=6) | 0.643 [0.357, 0.857] (n=14) |
| sd15 | 0.245 [0.233, 0.257] (n=20) | 0.401 [0.229, 0.585] (n=20) | 0.329 [0.265, 0.412] (n=6) | 0.286 [0.071, 0.571] (n=14) |
| sd35m | 0.271 [0.260, 0.283] (n=20) | 0.668 [0.495, 0.828] (n=20) | 0.924 [0.841, 0.986] (n=6) | 0.357 [0.143, 0.643] (n=14) |
| sdxl | 0.258 [0.247, 0.273] (n=20) | 0.623 [0.448, 0.793] (n=20) | 0.589 [0.510, 0.679] (n=6) | 0.571 [0.286, 0.857] (n=14) |

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

## Metric sanity checks

| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` | 80 / 80 | 0 | 0.197 | 0.335 | 0.027 | 0 |
| vqascore | `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8` | 80 / 80 | 0 | 0.000 | 1.000 | 0.407 | 0 |
| ocr | `easyocr==1.7.2/craft+english_g2/semiglobal-nospace-v1/bs16` | 24 / 24 | 0 | 0.240 | 1.000 | 0.268 | 0 |
| detect | `transformers==4.49.0/grounding-dino-tiny@a2bb814/box0.35-text0.25-nms0.5-group0.9-margin0.05/bs16` | 56 / 56 | 0 | 0.000 | 1.000 | 0.499 | 0 |

Rank agreement between metrics over the same images (Spearman). This says how often the metrics agree with each other, not which one agrees with people.

- clipscore~vqascore: ρ = 0.229 (n = 80)
- clipscore~ocr: ρ = 0.430 (n = 24)
- clipscore~detect: ρ = 0.016 (n = 56)
- vqascore~ocr: ρ = 0.873 (n = 24)
- vqascore~detect: ρ = 0.716 (n = 56)
- ocr~detect: ρ = – (n = 0)

## Agreement with human labels

Labels: `data/labels/smoke_labels.csv` (?=11, n=38, y=31; `?` excluded). Within each prompt, every pair of one image labelled correct and one labelled wrong: how often the metric scores the correct one higher (ties = 1/2). 95% CI from resampling prompts. One annotator, small sample: a sanity check, not a result (D-010, D-015).

| Metric | all | counting | spatial | text | mean score if correct / wrong |
| --- | --- | --- | --- | --- | --- |
| clipscore | 0.73 [0.56, 0.89] (33 pairs) | 0.62 [0.33, 1.00] (8 pairs) | 0.92 [0.75, 1.00] (12 pairs) | 0.62 [0.38, 0.92] (13 pairs) | 0.265 / 0.257 |
| vqascore | 0.97 [0.90, 1.00] (33 pairs) | 0.88 [0.67, 1.00] (8 pairs) | 1.00 [1.00, 1.00] (12 pairs) | 1.00 [1.00, 1.00] (13 pairs) | 0.904 / 0.438 |
| ocr | 1.00 [1.00, 1.00] (13 pairs) | – | – | 1.00 [1.00, 1.00] (13 pairs) | 0.960 / 0.505 |
| detect | 0.85 [0.70, 0.97] (20 pairs) | 1.00 [1.00, 1.00] (8 pairs) | 0.75 [0.57, 0.95] (12 pairs) | – | 0.909 / 0.111 |

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

Largest detect mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sdxl__sm011__s0` > `sd35m__sm011__s0` by 0.0
- `sd15__sm011__s0` > `sd35m__sm011__s0` by 0.0
- `sdxl__sm011__s0` > `flux_schnell__sm011__s0` by 0.0
- `sd15__sm011__s0` > `flux_schnell__sm011__s0` by 0.0
- `sdxl__sm009__s0` > `flux_schnell__sm009__s0` by 0.0

## Generation cost

Median seconds per image from the manifests, warmup batches excluded (D-004). GPU-hours = median × 2400 images / 3600.

| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for 2400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | NVIDIA H100 NVL | 1 | 4 | 20 (19) | 1.13 | 36.3 | 0 | 0 | 0.75 |
| sd15 | NVIDIA H100 NVL | 4 | 50 | 20 (16) | 0.53 | 4.3 | 0 | 0 | 0.35 |
| sd35m | NVIDIA H100 NVL | 2 | 40 | 20 (18) | 4.65 | 20.7 | 0 | 0 | 3.10 |
| sdxl | NVIDIA H100 NVL | 2 | 30 | 20 (18) | 1.90 | 15.8 | 0 | 0 | 1.27 |
