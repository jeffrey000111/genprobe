# Human evaluation on our own images (one annotator)

Made 2026-10-22T20:52:40.877+00:00 at commit `e5289d1` by `python -m genprobe.human_eval analyze`; protocol in `docs/human_eval_protocol.md` (D-029). Labelled: 260 of 260 main-part items.

- Attention checks: 19 of 20 right (pass mark 18 of 20): **passed**.
- Hidden repeats: same answer on 0.950 of 40 pairs; Cohen's κ 0.921 (stands in for agreement between annotators).
- Answers on the 200 main pairs: A 71, B 58, same 71; 129 pairs where one image was picked are used below.

## Does each metric prefer the image the annotator picked?

Share of decided pairs where the metric scores the picked image higher (ties 1/2); 95 % CI over prompts. Detector and OCR only apply to their own prompt types.

| Metric | Stratum | Agreement | CI | Pairs |
| --- | --- | --- | --- | --- |
| vqascore | all | 0.860 | [0.798, 0.915] | 129 |
| vqascore | counting | 0.818 | [0.667, 0.939] | 33 |
| vqascore | spatial | 1.000 | [1.000, 1.000] | 18 |
| vqascore | text | 0.837 | [0.721, 0.933] | 43 |
| vqascore | genai | 0.857 | [0.743, 0.971] | 35 |
| clipscore | all | 0.643 | [0.559, 0.727] | 129 |
| clipscore | counting | 0.455 | [0.303, 0.606] | 33 |
| clipscore | spatial | 0.667 | [0.444, 0.889] | 18 |
| clipscore | text | 0.674 | [0.537, 0.814] | 43 |
| clipscore | genai | 0.771 | [0.629, 0.914] | 35 |
| pickscore | all | 0.775 | [0.703, 0.847] | 129 |
| pickscore | counting | 0.758 | [0.606, 0.880] | 33 |
| pickscore | spatial | 0.722 | [0.500, 0.889] | 18 |
| pickscore | text | 0.860 | [0.750, 0.953] | 43 |
| pickscore | genai | 0.714 | [0.571, 0.857] | 35 |
| hpsv2 | all | 0.752 | [0.677, 0.827] | 129 |
| hpsv2 | counting | 0.758 | [0.606, 0.879] | 33 |
| hpsv2 | spatial | 0.667 | [0.444, 0.889] | 18 |
| hpsv2 | text | 0.814 | [0.690, 0.911] | 43 |
| hpsv2 | genai | 0.714 | [0.571, 0.857] | 35 |
| detect | all | 0.706 | [0.618, 0.784] | 51 |
| detect | counting | 0.667 | [0.561, 0.773] | 33 |
| detect | spatial | 0.778 | [0.639, 0.917] | 18 |
| ocr | all | 0.860 | [0.756, 0.943] | 43 |
| ocr | text | 0.860 | [0.756, 0.943] | 43 |
