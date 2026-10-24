# Diffusion-DPO on SD 1.5 with VQAScore preferences (guide §11.4, D-037)

Made 2026-10-24T09:09:57.155+00:00 at commit `c1fd61e` by `python -m genprobe.dpo analyze`. Test prompts of prompt set v1 (8); judge Gemma 4 31B (grades `grade_dpo_test_smoke_gemma4_31b`), pass = score 5, seed 0; automatic metrics averaged over seeds 0-3 per prompt; 95 % CIs from resampling prompts.

**H6 (pre-registered): DPO raises the judge pass rate.** smk_s10 − smk_base = 0.125 [0.00, 0.38]: **inconclusive**.

| Judge pass rate | all | counting | genai | spatial | text |
| --- | --- | --- | --- | --- | --- |
| smk_base | 0.375 [0.12, 0.75] | 1.000 [1.00, 1.00] | 0.500 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| smk_s10 | 0.500 [0.12, 0.88] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| smk_s10 − smk_base | 0.125 [0.00, 0.38] | 0.000 [0.00, 0.00] | 0.500 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |

Judge expected score, smk_s10 − smk_base: 0.126 [-0.00, 0.38].

## Reward-hacking checks: the trained-on metric against the others

| Metric | smk_base | smk_s10 | smk_s10 − smk_base |
| --- | --- | --- | --- |
| vqascore (trained on) | 0.616 [0.37, 0.83] | 0.614 [0.37, 0.83] | -0.002 [-0.01, 0.00] |
| clipscore | 0.260 [0.24, 0.28] | 0.258 [0.24, 0.27] | -0.002 [-0.00, -0.00] |
| pickscore | 21.280 [20.82, 21.74] | 21.271 [20.81, 21.73] | -0.009 [-0.02, -0.00] |
| hpsv2 | 0.250 [0.23, 0.26] | 0.250 [0.23, 0.26] | -0.000 [-0.00, 0.00] |
| detect | 0.500 [0.12, 0.88] | 0.500 [0.12, 0.88] | 0.000 [0.00, 0.00] |
| ocr | 0.714 [0.68, 0.75] | 0.679 [0.61, 0.75] | -0.036 [-0.07, 0.00] |

Detector and OCR apply to counting / spatial and text prompts only.
