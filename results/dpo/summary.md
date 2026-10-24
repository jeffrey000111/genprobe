# Diffusion-DPO on SD 1.5 with VQAScore preferences (guide §11.4, D-037)

Made 2026-10-24T13:21:22.224+00:00 at commit `aae6d90` by `python -m genprobe.dpo analyze`. Test prompts of prompt set v1 (480); judge Gemma 4 31B (grades `grade_dpo_test_full_gemma4_31b`), pass = score 5, seed 0; automatic metrics averaged over seeds 0-3 per prompt; 95 % CIs from resampling prompts.

**H6 (pre-registered): DPO raises the judge pass rate.** dpo_s2000 − dpo_base = 0.013 [-0.01, 0.04]: **inconclusive**.

| Judge pass rate | all | counting | genai | spatial | text |
| --- | --- | --- | --- | --- | --- |
| dpo_base | 0.171 [0.14, 0.20] | 0.411 [0.29, 0.54] | 0.156 [0.12, 0.19] | 0.161 [0.07, 0.27] | 0.000 [0.00, 0.00] |
| dpo_s2000 | 0.183 [0.15, 0.22] | 0.411 [0.29, 0.54] | 0.175 [0.14, 0.22] | 0.161 [0.07, 0.27] | 0.000 [0.00, 0.00] |
| dpo_s2000 − dpo_base | 0.013 [-0.01, 0.04] | 0.000 [-0.07, 0.07] | 0.019 [-0.01, 0.05] | 0.000 [-0.07, 0.07] | 0.000 [0.00, 0.00] |

Judge expected score, dpo_s2000 − dpo_base: 0.059 [-0.01, 0.12].

## Reward-hacking checks: the trained-on metric against the others

| Metric | dpo_base | dpo_s2000 | dpo_s2000 − dpo_base |
| --- | --- | --- | --- |
| vqascore (trained on) | 0.364 [0.34, 0.39] | 0.388 [0.36, 0.42] | 0.023 [0.01, 0.03] |
| clipscore | 0.270 [0.27, 0.27] | 0.273 [0.27, 0.28] | 0.003 [0.00, 0.00] |
| pickscore | 21.154 [21.07, 21.24] | 21.204 [21.11, 21.30] | 0.049 [0.03, 0.07] |
| hpsv2 | 0.249 [0.25, 0.25] | 0.252 [0.25, 0.25] | 0.003 [0.00, 0.00] |
| detect | 0.362 [0.30, 0.43] | 0.400 [0.34, 0.46] | 0.038 [0.01, 0.07] |
| ocr | 0.424 [0.38, 0.47] | 0.439 [0.39, 0.49] | 0.014 [-0.00, 0.03] |

Detector and OCR apply to counting / spatial and text prompts only.
