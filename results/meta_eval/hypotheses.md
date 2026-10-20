# Metric meta-evaluation on GenAI-Bench

Made 2026-10-20T00:59:22.480+00:00 at commit `8b0bdc9` by `python -m genprobe.meta_eval`. 1600 prompts, 9600 images. 18098 within-prompt image pairs whose mean human ratings differ; 5902 of 24000 pairs (24.6 %) have equal means and are excluded. 2 out-of-range ratings dropped. Krippendorff's α (ordinal) over the human ratings: 0.715.

Coverage: clipscore 9600/9600 images scored; vqascore 9600/9600 images scored; pickscore 9600/9600 images scored; hpsv2 9600/9600 images scored.

## Pre-registered hypotheses (docs/decisions.md)

A hypothesis is supported when its CI lies above 0, contradicted when below, inconclusive otherwise.

| Test | Comparison | Agreement a / b | Difference | CI | Pairs | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | vqascore - clipscore, prompts tagged 'counting' | 0.748 / 0.577 | 0.171 | 95 %: [0.142, 0.199] | 3852 | **supported** |
| H2a | vqascore - pickscore, all prompts | 0.733 / 0.670 | 0.062 | 97.5 %: [0.046, 0.077] | 18098 | **supported** |
| H2b | vqascore - hpsv2, all prompts | 0.733 / 0.665 | 0.067 | 97.5 %: [0.051, 0.083] | 18098 | **supported** |

## Agreement with people by tag

Exploratory, apart from the cells the tests above use. Share of within-prompt image pairs ordered like the mean human rating, 95 % CI from resampling prompts. Human ceiling: each rating against the mean of the other two. GenAI-Bench prompts carry several tags, so rows overlap.

| Tag | Prompts | Human ceiling | clipscore | vqascore | pickscore | hpsv2 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1558 | 0.81 | 0.58 [0.57, 0.59] | 0.73 [0.72, 0.74] | 0.67 [0.66, 0.68] | 0.67 [0.65, 0.68] |
| basic | 699 | 0.80 | 0.59 [0.57, 0.61] | 0.74 [0.73, 0.75] | 0.72 [0.70, 0.73] | 0.72 [0.71, 0.73] |
| advanced | 853 | 0.81 | 0.57 [0.55, 0.58] | 0.73 [0.71, 0.74] | 0.63 [0.62, 0.65] | 0.62 [0.61, 0.64] |
| action_relation | 455 | 0.81 | 0.59 [0.57, 0.61] | 0.74 [0.72, 0.76] | 0.71 [0.70, 0.73] | 0.72 [0.70, 0.74] |
| attribute | 534 | 0.81 | 0.60 [0.58, 0.62] | 0.75 [0.74, 0.77] | 0.73 [0.71, 0.74] | 0.72 [0.71, 0.74] |
| comparison | 159 | 0.82 | 0.59 [0.56, 0.62] | 0.72 [0.69, 0.75] | 0.65 [0.62, 0.68] | 0.64 [0.61, 0.67] |
| counting | 334 | 0.82 | 0.58 [0.55, 0.60] | 0.75 [0.73, 0.77] | 0.67 [0.65, 0.69] | 0.65 [0.62, 0.67] |
| differentiation | 292 | 0.83 | 0.55 [0.53, 0.58] | 0.73 [0.71, 0.75] | 0.65 [0.63, 0.67] | 0.66 [0.64, 0.69] |
| negation | 337 | 0.79 | 0.54 [0.52, 0.56] | 0.71 [0.68, 0.73] | 0.57 [0.55, 0.60] | 0.55 [0.52, 0.57] |
| part_relation | 192 | 0.82 | 0.62 [0.59, 0.65] | 0.76 [0.73, 0.78] | 0.73 [0.70, 0.75] | 0.72 [0.69, 0.74] |
| scene | 348 | 0.78 | 0.58 [0.56, 0.60] | 0.74 [0.72, 0.76] | 0.73 [0.71, 0.75] | 0.72 [0.70, 0.74] |
| spatial_relation | 377 | 0.80 | 0.59 [0.57, 0.61] | 0.75 [0.73, 0.77] | 0.71 [0.69, 0.72] | 0.72 [0.70, 0.74] |
| universal | 143 | 0.79 | 0.54 [0.50, 0.58] | 0.71 [0.67, 0.74] | 0.65 [0.62, 0.69] | 0.62 [0.58, 0.66] |
| untagged | 6 | 0.79 | 0.58 [0.47, 0.68] | 0.61 [0.45, 0.78] | 0.46 [0.31, 0.60] | 0.51 [0.42, 0.62] |

Kendall's τ-b between metric and mean rating within each prompt, averaged over prompts (exploratory): clipscore 0.133; vqascore 0.403; pickscore 0.296; hpsv2 0.287.

Metric versions: clipscore `open_clip==3.3.0/ViT-L-14:openai/cos/bs32`; vqascore `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8`; pickscore `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16`; hpsv2 `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16`.
