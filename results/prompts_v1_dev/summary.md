# Summary: `data/prompts/prompts_v1.jsonl` (split: dev)

Generated 2026-10-19T21:16:38.660+00:00 at commit `41a867f` by `python -m genprobe.summarize`. 120 prompts; means are over prompts, 95% CIs from 2000 bootstrap resamples of prompts.

## Generation coverage

| Model | expected images | ok | blank | error | no record |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 480 | 480 | 0 | 0 | 0 |
| sd15 | 480 | 480 | 0 | 0 | 0 |
| sd35m | 480 | 480 | 0 | 0 | 0 |
| sdxl | 480 | 480 | 0 | 0 | 0 |

## Metric scores by model

| Model | clipscore | vqascore | ocr | detect | pickscore | hpsv2 | nsfw_sd | nsfw_vit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.279 [0.272, 0.285] (n=480) | 0.645 [0.586, 0.706] (n=480) | 0.912 [0.875, 0.945] (n=48) | 0.509 [0.393, 0.625] (n=112) | 22.605 [22.419, 22.778] (n=480) | 0.296 [0.291, 0.301] (n=480) | 0.015 [0.004, 0.029] (n=480) | 0.002 [0.000, 0.006] (n=480) |
| sd15 | 0.267 [0.262, 0.272] (n=480) | 0.333 [0.281, 0.386] (n=480) | 0.376 [0.308, 0.455] (n=48) | 0.286 [0.179, 0.411] (n=112) | 21.193 [21.000, 21.380] (n=480) | 0.251 [0.245, 0.257] (n=480) | 0.002 [0.000, 0.006] (n=480) | 0.000 [0.000, 0.000] (n=480) |
| sd35m | 0.278 [0.273, 0.284] (n=480) | 0.649 [0.588, 0.709] (n=480) | 0.896 [0.849, 0.943] (n=48) | 0.661 [0.536, 0.777] (n=112) | 22.508 [22.325, 22.691] (n=480) | 0.286 [0.280, 0.292] (n=480) | 0.008 [0.002, 0.017] (n=480) | 0.000 [0.000, 0.000] (n=480) |
| sdxl | 0.276 [0.271, 0.281] (n=480) | 0.487 [0.427, 0.550] (n=480) | 0.678 [0.595, 0.760] (n=48) | 0.438 [0.312, 0.562] (n=112) | 22.357 [22.167, 22.537] (n=480) | 0.278 [0.273, 0.284] (n=480) | 0.010 [0.002, 0.021] (n=480) | 0.000 [0.000, 0.000] (n=480) |

### clipscore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.280 (196) | 0.290 (124) | 0.239 (56) | 0.283 (56) | 0.293 (48) |
| sd15 | 0.271 (196) | 0.278 (124) | 0.245 (56) | 0.250 (56) | 0.266 (48) |
| sd35m | 0.273 (196) | 0.288 (124) | 0.254 (56) | 0.278 (56) | 0.305 (48) |
| sdxl | 0.272 (196) | 0.284 (124) | 0.254 (56) | 0.272 (56) | 0.303 (48) |

### vqascore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.539 (196) | 0.802 (124) | 0.566 (56) | 0.491 (56) | 0.945 (48) |
| sd15 | 0.282 (196) | 0.441 (124) | 0.318 (56) | 0.280 (56) | 0.343 (48) |
| sd35m | 0.454 (196) | 0.798 (124) | 0.648 (56) | 0.737 (56) | 0.956 (48) |
| sdxl | 0.399 (196) | 0.605 (124) | 0.409 (56) | 0.391 (56) | 0.747 (48) |

### ocr by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | – (0) | – (0) | – (0) | – (0) | 0.912 (48) |
| sd15 | – (0) | – (0) | – (0) | – (0) | 0.376 (48) |
| sd35m | – (0) | – (0) | – (0) | – (0) | 0.896 (48) |
| sdxl | – (0) | – (0) | – (0) | – (0) | 0.678 (48) |

### detect by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | – (0) | – (0) | 0.500 (56) | 0.518 (56) | – (0) |
| sd15 | – (0) | – (0) | 0.268 (56) | 0.304 (56) | – (0) |
| sd35m | – (0) | – (0) | 0.589 (56) | 0.732 (56) | – (0) |
| sdxl | – (0) | – (0) | 0.393 (56) | 0.482 (56) | – (0) |

### pickscore by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 22.369 (196) | 22.793 (124) | 22.351 (56) | 23.101 (56) | 22.805 (48) |
| sd15 | 21.235 (196) | 21.513 (124) | 21.417 (56) | 21.502 (56) | 19.579 (48) |
| sd35m | 22.344 (196) | 22.815 (124) | 22.405 (56) | 22.504 (56) | 22.507 (48) |
| sdxl | 22.294 (196) | 22.739 (124) | 22.022 (56) | 22.812 (56) | 21.488 (48) |

### hpsv2 by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.294 (196) | 0.297 (124) | 0.288 (56) | 0.319 (56) | 0.288 (48) |
| sd15 | 0.257 (196) | 0.259 (124) | 0.244 (56) | 0.258 (56) | 0.203 (48) |
| sd35m | 0.284 (196) | 0.296 (124) | 0.272 (56) | 0.294 (56) | 0.274 (48) |
| sdxl | 0.278 (196) | 0.287 (124) | 0.264 (56) | 0.294 (56) | 0.256 (48) |

### nsfw_sd by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.026 (196) | 0.008 (124) | 0.018 (56) | 0.000 (56) | 0.000 (48) |
| sd15 | 0.000 (196) | 0.000 (124) | 0.000 (56) | 0.018 (56) | 0.000 (48) |
| sd35m | 0.005 (196) | 0.000 (124) | 0.054 (56) | 0.000 (56) | 0.000 (48) |
| sdxl | 0.005 (196) | 0.008 (124) | 0.018 (56) | 0.036 (56) | 0.000 (48) |

### nsfw_vit by category (mean, n images)

| Model | advanced | basic | counting | spatial | text |
| --- | --- | --- | --- | --- | --- |
| flux_schnell | 0.000 (196) | 0.000 (124) | 0.000 (56) | 0.000 (56) | 0.021 (48) |
| sd15 | 0.000 (196) | 0.000 (124) | 0.000 (56) | 0.000 (56) | 0.000 (48) |
| sd35m | 0.000 (196) | 0.000 (124) | 0.000 (56) | 0.000 (56) | 0.000 (48) |
| sdxl | 0.000 (196) | 0.000 (124) | 0.000 (56) | 0.000 (56) | 0.000 (48) |

## Metric sanity checks

| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` | 1920 / 1920 | 0 | 0.181 | 0.412 | 0.036 | 0 |
| vqascore | `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8` | 1920 / 1920 | 0 | 0.000 | 1.000 | 0.419 | 0 |
| ocr | `easyocr==1.7.2/craft+english_g2/semiglobal-nospace-v1/bs16` | 192 / 192 | 0 | 0.000 | 1.000 | 0.271 | 0 |
| detect | `transformers==4.49.0/grounding-dino-tiny@a2bb814/box0.35-text0.25-nms0.5-group0.9-margin0.05/bs16` | 448 / 448 | 0 | 0.000 | 1.000 | 0.499 | 0 |
| pickscore | `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16` | 1920 / 1920 | 0 | 18.068 | 25.470 | 1.282 | 0 |
| hpsv2 | `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16` | 1920 / 1920 | 0 | 0.144 | 0.374 | 0.039 | 0 |
| nsfw_sd | `diffusers==0.39.0/sd-safety-checker@cb41f3a/fp32/bs32` | 1920 / 1920 | 0 | 0.000 | 1.000 | 0.094 | 0 |
| nsfw_vit | `transformers==4.49.0/falconsai-nsfw-vit@96cb0d0/fp32/argmax/bs32` | 1920 / 1920 | 0 | 0.000 | 1.000 | 0.023 | 0 |

Rank agreement between metrics over the same images (Spearman). This says how often the metrics agree with each other, not which one agrees with people.

- clipscore~vqascore: ρ = 0.334 (n = 1920)
- clipscore~ocr: ρ = 0.207 (n = 192)
- clipscore~detect: ρ = 0.207 (n = 448)
- clipscore~pickscore: ρ = 0.277 (n = 1920)
- clipscore~hpsv2: ρ = 0.371 (n = 1920)
- clipscore~nsfw_sd: ρ = 0.012 (n = 1920)
- clipscore~nsfw_vit: ρ = -0.019 (n = 1920)
- vqascore~ocr: ρ = 0.782 (n = 192)
- vqascore~detect: ρ = 0.603 (n = 448)
- vqascore~pickscore: ρ = 0.339 (n = 1920)
- vqascore~hpsv2: ρ = 0.275 (n = 1920)
- vqascore~nsfw_sd: ρ = -0.016 (n = 1920)
- vqascore~nsfw_vit: ρ = 0.027 (n = 1920)
- ocr~detect: ρ = – (n = 0)
- ocr~pickscore: ρ = 0.762 (n = 192)
- ocr~hpsv2: ρ = 0.723 (n = 192)
- ocr~nsfw_sd: ρ = – (n = 192)
- ocr~nsfw_vit: ρ = 0.094 (n = 192)
- detect~pickscore: ρ = 0.219 (n = 448)
- detect~hpsv2: ρ = 0.164 (n = 448)
- detect~nsfw_sd: ρ = 0.075 (n = 448)
- detect~nsfw_vit: ρ = – (n = 448)
- pickscore~hpsv2: ρ = 0.684 (n = 1920)
- pickscore~nsfw_sd: ρ = 0.006 (n = 1920)
- pickscore~nsfw_vit: ρ = 0.038 (n = 1920)
- hpsv2~nsfw_sd: ρ = -0.022 (n = 1920)
- hpsv2~nsfw_vit: ρ = -0.004 (n = 1920)
- nsfw_sd~nsfw_vit: ρ = -0.002 (n = 1920)

## Agreement with human labels

Labels: `data/labels/prompts_v1_labels.csv` (n=60, y=40; `?` excluded). Within each prompt, every pair of one image labelled correct and one labelled wrong: how often the metric scores the correct one higher (ties = 1/2). 95% CI from resampling prompts. One annotator, small sample: a sanity check, not a result (D-010, D-015).

| Metric | all | advanced | basic | counting | spatial | text | mean score if correct / wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | 0.65 [0.43, 0.88] (78 pairs) | – | – | 0.62 [0.00, 0.83] (8 pairs) | 1.00 [1.00, 1.00] (10 pairs) | 0.60 [0.37, 0.88] (60 pairs) | 0.274 / 0.260 |
| vqascore | 0.92 [0.85, 1.00] (78 pairs) | – | – | 1.00 [1.00, 1.00] (8 pairs) | 1.00 [1.00, 1.00] (10 pairs) | 0.90 [0.81, 1.00] (60 pairs) | 0.874 / 0.390 |
| ocr | 0.97 [0.95, 1.00] (60 pairs) | – | – | – | – | 0.97 [0.95, 1.00] (60 pairs) | 0.919 / 0.490 |
| detect | 0.89 [0.77, 1.00] (18 pairs) | – | – | 0.81 [0.75, 1.00] (8 pairs) | 0.95 [0.75, 1.00] (10 pairs) | – | 0.941 / 0.121 |
| pickscore | 0.94 [0.86, 0.99] (78 pairs) | – | – | 0.88 [0.50, 1.00] (8 pairs) | 0.80 [0.67, 1.00] (10 pairs) | 0.97 [0.90, 1.00] (60 pairs) | 22.606 / 21.342 |
| hpsv2 | 0.86 [0.74, 0.96] (78 pairs) | – | – | 0.75 [0.00, 1.00] (8 pairs) | 0.80 [0.67, 1.00] (10 pairs) | 0.88 [0.75, 0.98] (60 pairs) | 0.283 / 0.253 |

Checker verdicts against the same labels, image by image (an image passes when the checker's value ≥ its threshold). Accuracy CI from resampling prompts.

| Checker | threshold | images | accuracy | passes & correct | passes & wrong | fails & correct | fails & wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ocr | 0.85 | 50 | 0.86 [0.82, 0.91] | 19 | 3 | 4 | 24 |
| detect | 1.0 | 50 | 0.90 [0.81, 0.97] | 16 | 4 | 1 | 29 |

Detector count vs the count the annotator wrote down (20 counting images): same number on 85%, mean |difference| 0.20; more than the annotator on 3, fewer on 0.

Largest clipscore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sdxl__tt0039__s0` > `flux_schnell__tt0039__s0` by 0.0778
- `sdxl__tt0039__s0` > `sd35m__tt0039__s3` by 0.0775
- `sd15__tt0017__s3` > `flux_schnell__tt0017__s2` by 0.0773
- `sd15__tt0013__s2` > `flux_schnell__tt0013__s2` by 0.0689
- `sd15__tt0013__s2` > `flux_schnell__tt0013__s1` by 0.0573

Largest vqascore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sdxl__tt0010__s1` > `sd35m__tt0010__s3` by 0.2193
- `sd35m__tt0013__s0` > `flux_schnell__tt0013__s0` by 0.0635
- `sd15__tt0013__s2` > `flux_schnell__tt0013__s0` by 0.0337
- `sdxl__tt0010__s1` > `flux_schnell__tt0010__s0` by 0.0168
- `sdxl__tt0010__s1` > `flux_schnell__tt0010__s2` by 0.0031

Largest pickscore mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `sdxl__ts0027__s2` > `sd35m__ts0027__s1` by 1.0434
- `flux_schnell__tc0027__s0` > `sd35m__tc0027__s2` by 0.483
- `sdxl__tt0055__s2` > `sd35m__tt0055__s0` by 0.4362
- `sdxl__tt0031__s3` > `sdxl__tt0031__s2` by 0.3146
- `sdxl__ts0027__s2` > `sd35m__ts0027__s2` by 0.279

Largest hpsv2 mistakes (image labelled wrong scored above one labelled correct, same prompt):

- `flux_schnell__tc0027__s0` > `sd35m__tc0027__s2` by 0.0726
- `sdxl__tt0055__s2` > `sd35m__tt0055__s0` by 0.0352
- `sd35m__tt0039__s1` > `flux_schnell__tt0039__s1` by 0.0197
- `sdxl__ts0027__s2` > `sd35m__ts0027__s2` by 0.0127
- `sd35m__tt0039__s1` > `sd35m__tt0039__s3` by 0.0095

## Generation cost

Median seconds per image from the manifests, warmup batches excluded (D-004). GPU-hours = median × 2400 images / 3600.

| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for 2400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | NVIDIA H100 NVL | 1 | 4 | 480 (480) | 1.13 | 36.3 | 0 | 0 | 0.75 |
| sd15 | NVIDIA H100 NVL | 4 | 50 | 480 (480) | 0.53 | 4.3 | 0 | 0 | 0.35 |
| sd35m | NVIDIA H100 NVL | 2 | 40 | 480 (480) | 4.60 | 20.7 | 0 | 0 | 3.07 |
| sdxl | NVIDIA H100 NVL | 2 | 50 | 480 (480) | 3.05 | 15.8 | 0 | 0 | 2.03 |
