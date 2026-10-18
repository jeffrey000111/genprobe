# Summary: `data/prompts/smoke.jsonl`

Generated 2026-10-18T09:18:54.252+00:00 at commit `7ec27f3` by `python -m genprobe.summarize`. 20 prompts; means are over prompts, 95% CIs from 2000 bootstrap resamples of prompts.

## Metric scores by model

| Model | clipscore | vqascore |
| --- | --- | --- |
| flux_schnell | 0.266 [0.258, 0.274] (n=20) | – [–, –] (n=0) |
| sd15 | 0.245 [0.233, 0.257] (n=20) | – [–, –] (n=0) |
| sd35m | 0.271 [0.260, 0.283] (n=20) | – [–, –] (n=0) |
| sdxl | 0.258 [0.247, 0.273] (n=20) | – [–, –] (n=0) |

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
| flux_schnell | – (0) | – (0) | – (0) |
| sd15 | – (0) | – (0) | – (0) |
| sd35m | – (0) | – (0) | – (0) |
| sdxl | – (0) | – (0) | – (0) |

## Metric sanity checks

| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clipscore | `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` | 80 / 80 | 0 | 0.197 | 0.335 | 0.027 | 0 |
| vqascore | `None` | 0 / 80 | 0 | – | – | – | 0 |

Rank agreement between metrics over the same images (Spearman). This says how often the metrics agree with each other, not which one agrees with people.

- clipscore~vqascore: ρ = – (n = 0)

## Generation cost

Median seconds per image from the manifests, warmup batches excluded (D-004). GPU-hours = median × 2400 images / 3600.

| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for 2400 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flux_schnell | NVIDIA H100 NVL | 1 | 4 | 20 (19) | 1.13 | 36.3 | 0 | 0 | 0.75 |
| sd15 | NVIDIA H100 NVL | 4 | 50 | 20 (16) | 0.53 | 4.3 | 0 | 0 | 0.35 |
| sd35m | NVIDIA H100 NVL | 2 | 40 | 20 (18) | 4.65 | 20.7 | 0 | 0 | 3.10 |
| sdxl | NVIDIA H100 NVL | 2 | 30 | 20 (18) | 1.90 | 15.8 | 0 | 0 | 1.27 |
