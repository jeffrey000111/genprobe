# Metric meta-evaluation on GenAI-Bench-Video

Made 2026-10-24T01:55:06.600+00:00 at commit `44329ac` by `python -m genprobe.video_meta_eval` (D-035). Data `zhiqiulin/GenAI-Bench-800@22bb97341224f58e0b080111b9c2f775999ffdaa`: 800 prompts, 3200 videos. 3773 within-prompt video pairs whose mean human ratings differ; 1027 of 4800 pairs (21.4 %) have equal means and are excluded. 0 out-of-range ratings dropped. Krippendorff's α (ordinal): 0.675.

## Pre-registered hypotheses (docs/decisions.md, D-035)

Supported when the CI lies above 0, contradicted when below, inconclusive otherwise.

| Test | Comparison | Agreement a / b | Difference | CI | Pairs | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| H4 | vqascore_video - clipscore_frames, all prompts | 0.675 / 0.613 | 0.062 | 97.5 %: [0.035, 0.090] | 3773 | **supported** |
| H5 | vqascore_video - vqascore_mid, all prompts | 0.675 / 0.693 | -0.018 | 97.5 %: [-0.034, -0.002] | 3773 | **contradicted** |

## Agreement with people by tag (exploratory apart from the tests above)

| Tag | Prompts | Human ceiling | vqascore_video | vqascore_mid | clipscore_frames | pickscore_frames | hpsv2_frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 775 | 0.80 | 0.67 [0.66, 0.69] | 0.69 [0.67, 0.71] | 0.61 [0.59, 0.63] | 0.68 [0.67, 0.70] | 0.68 [0.66, 0.70] |
| basic | 279 | 0.80 | 0.71 [0.68, 0.74] | 0.72 [0.69, 0.75] | 0.64 [0.61, 0.67] | 0.70 [0.67, 0.72] | 0.69 [0.66, 0.72] |
| advanced | 496 | 0.79 | 0.65 [0.63, 0.67] | 0.67 [0.65, 0.70] | 0.60 [0.57, 0.62] | 0.68 [0.65, 0.70] | 0.67 [0.65, 0.69] |
| action_relation | 165 | 0.80 | 0.70 [0.66, 0.74] | 0.71 [0.67, 0.75] | 0.64 [0.60, 0.68] | 0.70 [0.66, 0.73] | 0.69 [0.65, 0.73] |
| attribute | 194 | 0.80 | 0.73 [0.70, 0.76] | 0.73 [0.70, 0.76] | 0.63 [0.60, 0.67] | 0.67 [0.64, 0.71] | 0.67 [0.64, 0.71] |
| comparison | 99 | 0.77 | 0.60 [0.54, 0.65] | 0.63 [0.57, 0.69] | 0.61 [0.55, 0.66] | 0.66 [0.61, 0.70] | 0.71 [0.66, 0.76] |
| counting | 218 | 0.79 | 0.63 [0.59, 0.66] | 0.66 [0.62, 0.69] | 0.58 [0.55, 0.62] | 0.68 [0.64, 0.71] | 0.68 [0.64, 0.71] |
| differentiation | 212 | 0.78 | 0.61 [0.57, 0.65] | 0.65 [0.62, 0.69] | 0.61 [0.57, 0.65] | 0.68 [0.64, 0.71] | 0.68 [0.65, 0.72] |
| negation | 167 | 0.80 | 0.66 [0.62, 0.70] | 0.69 [0.65, 0.73] | 0.58 [0.54, 0.62] | 0.68 [0.64, 0.72] | 0.65 [0.61, 0.69] |
| part_relation | 49 | 0.79 | 0.71 [0.65, 0.77] | 0.71 [0.65, 0.78] | 0.63 [0.56, 0.70] | 0.68 [0.61, 0.75] | 0.67 [0.59, 0.75] |
| scene | 134 | 0.78 | 0.74 [0.70, 0.78] | 0.74 [0.69, 0.78] | 0.64 [0.60, 0.69] | 0.71 [0.67, 0.75] | 0.71 [0.67, 0.75] |
| spatial_relation | 141 | 0.81 | 0.68 [0.64, 0.72] | 0.71 [0.66, 0.75] | 0.62 [0.58, 0.66] | 0.69 [0.65, 0.73] | 0.70 [0.66, 0.74] |
| universal | 79 | 0.81 | 0.74 [0.68, 0.79] | 0.72 [0.66, 0.78] | 0.59 [0.52, 0.66] | 0.70 [0.64, 0.74] | 0.70 [0.64, 0.75] |

Kendall's τ-b within prompts, averaged (exploratory): vqascore_video 0.311; vqascore_mid 0.344; clipscore_frames 0.202; pickscore_frames 0.325; hpsv2_frames 0.316.

## By model (means over its videos)

| Model | Videos | Mean human rating | vqascore_video | vqascore_mid | clipscore_frames | pickscore_frames | hpsv2_frames |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Floor33 | 800 | 2.92 | 0.338 | 0.335 | 0.247 | 21.169 | 0.251 |
| Gen2 | 800 | 3.12 | 0.413 | 0.394 | 0.255 | 21.941 | 0.261 |
| Pika_v1 | 800 | 3.01 | 0.414 | 0.414 | 0.251 | 21.025 | 0.239 |
| Modelscope | 800 | 2.68 | 0.436 | 0.392 | 0.263 | 20.416 | 0.210 |

Coverage: vqascore_video 3200/3200; vqascore_mid 3200/3200; clipscore_frames 3200/3200; pickscore_frames 3200/3200; hpsv2_frames 3200/3200.

Metric versions: vqascore_video `vqascore_video-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/fps8/maxpix151200/reader-decord/qwen-vl-utils==0.0.14/bs1`; vqascore_mid `vqascore-port-of-t2v_metrics-qwen2vl/transformers==4.49.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-af8bcca9/bs8`; clipscore_frames `open_clip==3.3.0/ViT-L-14:openai/cos/bs32`; pickscore_frames `transformers==4.49.0/pickscore-v1@a4e4367/fp32/bs16`; hpsv2_frames `open_clip==3.3.0/ViT-H-14/hps-v2.1@697403c/fp32/cos/bs16`.
