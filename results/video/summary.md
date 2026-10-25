# Generated videos: Wan2.2 TI2V-5B, LTX-Video and GenAI-Bench-Video's models on the same prompts

Made 2026-10-25T23:22:29.137+00:00 at commit `fec8bfe` by `python -m genprobe.video_eval analyze` (D-036). 160 prompts; means over prompts with 95 % CIs from resampling prompts. Exploratory.

| Measure | wan22_f49 | ltxv095_f49 | Floor33 | Gen2 | Pika_v1 | Modelscope |
| --- | --- | --- | --- | --- | --- | --- |
| vqascore_video | 0.613 [0.55, 0.67] | 0.413 [0.35, 0.47] | 0.342 [0.28, 0.40] | 0.399 [0.34, 0.47] | 0.407 [0.34, 0.47] | 0.453 [0.40, 0.51] |
| subject_consistency | 0.968 [0.96, 0.97] | 0.988 [0.98, 0.99] | 0.973 [0.97, 0.98] | 0.977 [0.97, 0.98] | 0.985 [0.98, 0.99] | 0.952 [0.94, 0.96] |
| background_consistency | 0.967 [0.96, 0.97] | 0.978 [0.98, 0.98] | 0.978 [0.98, 0.98] | 0.973 [0.97, 0.98] | 0.977 [0.97, 0.98] | 0.961 [0.96, 0.97] |
| motion_smoothness | 0.988 [0.99, 0.99] | 0.997 [1.00, 1.00] | 0.984 [0.98, 0.99] | 0.996 [1.00, 1.00] | 0.996 [1.00, 1.00] | 0.975 [0.97, 0.98] |
| dynamic_degree | 0.531 [0.46, 0.61] | 0.069 [0.03, 0.11] | 0.281 [0.21, 0.35] | 0.069 [0.03, 0.11] | 0.200 [0.14, 0.27] | 0.331 [0.26, 0.41] |
| aesthetic_quality | 0.541 [0.53, 0.55] | 0.511 [0.50, 0.53] | 0.650 [0.64, 0.66] | 0.654 [0.64, 0.67] | 0.592 [0.58, 0.61] | 0.452 [0.44, 0.47] |
| imaging_quality | 0.689 [0.67, 0.70] | 0.606 [0.59, 0.63] | 0.606 [0.59, 0.62] | 0.677 [0.66, 0.69] | 0.648 [0.63, 0.66] | 0.619 [0.61, 0.63] |

| Our run | Videos | Size | Frames | Mean GPU-s per video |
| --- | --- | --- | --- | --- |
| wan22_f49 | 160 | 1280 x 704 | 49 | 65.4 |
| ltxv095_f49 | 160 | 704 x 480 | 49 | 4.6 |

## wan22_f49 minus each other model (same prompts)

| Measure | ltxv095_f49 | Floor33 | Gen2 | Pika_v1 | Modelscope |
| --- | --- | --- | --- | --- | --- |
| vqascore_video | 0.199 [0.13, 0.27] | 0.271 [0.20, 0.34] | 0.214 [0.14, 0.29] | 0.206 [0.13, 0.27] | 0.160 [0.09, 0.23] |
| subject_consistency | -0.020 [-0.03, -0.02] | -0.005 [-0.01, 0.00] | -0.009 [-0.01, -0.00] | -0.017 [-0.02, -0.01] | 0.016 [0.01, 0.02] |
| background_consistency | -0.011 [-0.02, -0.01] | -0.012 [-0.02, -0.01] | -0.006 [-0.01, -0.00] | -0.010 [-0.01, -0.01] | 0.006 [0.00, 0.01] |
| motion_smoothness | -0.009 [-0.01, -0.01] | 0.004 [0.00, 0.01] | -0.008 [-0.01, -0.01] | -0.008 [-0.01, -0.01] | 0.013 [0.01, 0.02] |
| dynamic_degree | 0.463 [0.38, 0.54] | 0.250 [0.16, 0.34] | 0.463 [0.38, 0.54] | 0.331 [0.24, 0.42] | 0.200 [0.11, 0.29] |
| aesthetic_quality | 0.031 [0.02, 0.04] | -0.109 [-0.12, -0.09] | -0.113 [-0.13, -0.10] | -0.051 [-0.07, -0.03] | 0.089 [0.07, 0.10] |
| imaging_quality | 0.083 [0.07, 0.10] | 0.084 [0.07, 0.10] | 0.012 [-0.00, 0.03] | 0.041 [0.02, 0.06] | 0.070 [0.05, 0.09] |

VQAScore on the whole video: D-035's metric (how far it agrees with people is in `results/video_meta_eval/`). The six VBench dimensions measure consistency, motion and image quality, not prompt following.
