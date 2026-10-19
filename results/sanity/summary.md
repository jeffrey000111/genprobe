# Metric sanity checks (guide §7.9)

Written 2026-10-19T21:16:37.085+00:00 at commit `41a867f` by `python -m genprobe.sanity --report`. Dev-split images only. Rates: share of images where the own prompt scores higher than the changed one (ties count 1/2); ↑ = / ↓ give the raw counts. Binary checkers tie often (both 0): read the ↓ column, which should be near 0.

## Mismatch and repeatability (200 random dev images)

| Metric | images | mismatched prompt | repeat in job | repeat vs scoring run |
| --- | --- | --- | --- | --- |
| clipscore | 200 | 1.00 (200↑ 0= 0↓ of 200) | 100% identical, max |Δ| 0 (n=50) | 98% identical, max |Δ| 6e-08 (n=200) |
| detect | 200 | 0.72 (90↑ 107= 3↓ of 200) | 100% identical, max |Δ| 0 (n=50) | 100% identical, max |Δ| 0 (n=200) |
| hpsv2 | 200 | 0.99 (199↑ 0= 1↓ of 200) | 96% identical, max |Δ| 6e-08 (n=50) | 96% identical, max |Δ| 6e-08 (n=200) |
| nsfw_sd | 200 | – | 100% identical, max |Δ| 0 (n=50) | 100% identical, max |Δ| 0 (n=200) |
| nsfw_vit | 200 | – | 100% identical, max |Δ| 0 (n=50) | 100% identical, max |Δ| 0 (n=200) |
| ocr | 192 | 0.93 (178↑ 1= 13↓ of 192) | 100% identical, max |Δ| 0 (n=50) | 100% identical, max |Δ| 0 (n=192) |
| pickscore | 200 | 1.00 (200↑ 0= 0↓ of 200) | 96% identical, max |Δ| 7.6e-06 (n=50) | 52% identical, max |Δ| 7.6e-06 (n=200) |
| vqascore | 200 | 0.99 (198↑ 0= 2↓ of 200) | 100% identical, max |Δ| 0 (n=50) | 100% identical, max |Δ| 0 (n=200) |

## One-word counterfactuals

**On images labelled correct** (the clean test: the original prompt is right, the changed one wrong):

| Metric | all | count ±2 | left/right, above/below | other text |
| --- | --- | --- | --- | --- |
| clipscore | 0.85 (34↑ 0= 6↓ of 40) | 0.88 (7↑ 0= 1↓ of 8) | 0.44 (4↑ 0= 5↓ of 9) | 1.00 (23↑ 0= 0↓ of 23) |
| detect | 0.94 (16↑ 0= 1↓ of 17) | 1.00 (8↑ 0= 0↓ of 8) | 0.89 (8↑ 0= 1↓ of 9) | – |
| hpsv2 | 0.85 (34↑ 0= 6↓ of 40) | 1.00 (8↑ 0= 0↓ of 8) | 0.33 (3↑ 0= 6↓ of 9) | 1.00 (23↑ 0= 0↓ of 23) |
| ocr | 1.00 (23↑ 0= 0↓ of 23) | – | – | 1.00 (23↑ 0= 0↓ of 23) |
| pickscore | 0.85 (34↑ 0= 6↓ of 40) | 1.00 (8↑ 0= 0↓ of 8) | 0.33 (3↑ 0= 6↓ of 9) | 1.00 (23↑ 0= 0↓ of 23) |
| vqascore | 1.00 (40↑ 0= 0↓ of 40) | 1.00 (8↑ 0= 0↓ of 8) | 1.00 (9↑ 0= 0↓ of 9) | 1.00 (23↑ 0= 0↓ of 23) |

On the random template images of the sample (confounded: where the generator got the count or position wrong, the changed prompt can fit the image better, so even a perfect metric scores below 1):

| Metric | all | count ±2 | left/right, above/below | other text |
| --- | --- | --- | --- | --- |
| clipscore | 0.72 (43↑ 0= 17↓ of 60) | 0.77 (10↑ 0= 3↓ of 13) | 0.48 (13↑ 0= 14↓ of 27) | 1.00 (20↑ 0= 0↓ of 20) |
| detect | 0.67 (92↑ 82= 26↓ of 200) | 0.65 (43↑ 56= 10↓ of 109) | 0.68 (49↑ 26= 16↓ of 91) | – |
| hpsv2 | 0.70 (42↑ 0= 18↓ of 60) | 0.69 (9↑ 0= 4↓ of 13) | 0.48 (13↑ 0= 14↓ of 27) | 1.00 (20↑ 0= 0↓ of 20) |
| ocr | 0.97 (185↑ 1= 6↓ of 192) | – | – | 0.97 (185↑ 1= 6↓ of 192) |
| pickscore | 0.68 (41↑ 0= 19↓ of 60) | 0.77 (10↑ 0= 3↓ of 13) | 0.41 (11↑ 0= 16↓ of 27) | 1.00 (20↑ 0= 0↓ of 20) |
| vqascore | 0.72 (43↑ 0= 17↓ of 60) | 0.77 (10↑ 0= 3↓ of 13) | 0.48 (13↑ 0= 14↓ of 27) | 1.00 (20↑ 0= 0↓ of 20) |

Versions: clipscore `open_clip==3.3.0/ViT-L-14:openai/cos/bs32` on cuda; detect `transformers==4.49.0/grounding-dino-tiny@a2bb814/box0.35-text0.25-nms0.5-group0.9-margin0.05/bs16` on cpu; hpsv2 `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16` on cuda; nsfw_sd `diffusers==0.39.0/sd-safety-checker@cb41f3a/fp32/bs32` on cuda; nsfw_vit `transformers==4.49.0/falconsai-nsfw-vit@96cb0d0/fp32/argmax/bs32` on cuda; ocr `easyocr==1.7.2/craft+english_g2/semiglobal-nospace-v1/bs16` on cpu; pickscore `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16` on cuda; vqascore `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8` on cuda
