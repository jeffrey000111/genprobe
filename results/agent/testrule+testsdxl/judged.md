# Agent run(s) `testrule`, `testsdxl` graded by the independent judge

Made 2026-10-23T03:56:40.057+00:00 at commit `01855d0` by `python -m genprobe.agent.compare`. Judge `vllm==0.32.0/gemma-4-31b@842da37/bf16/vis560/img1024/tmpl-b9e98ed8` (grades `grade_testrule_gemma4_31b`, `grade_testsdxl_gemma4_31b`), original prompts, pass = judge score == 5. 480 prompts; 95 % CIs from resampling prompts.

## Pass rate

| Group | all | counting | genai | spatial | text | GPU-s / prompt | GPU-s / success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.246 [0.21, 0.28] | 0.286 [0.18, 0.41] | 0.253 [0.21, 0.30] | 0.250 [0.14, 0.38] | 0.146 [0.06, 0.25] | 5.3 | 21.6 |
| best_of_4 | 0.367 [0.33, 0.41] | 0.589 [0.46, 0.71] | 0.322 [0.27, 0.38] | 0.554 [0.43, 0.68] | 0.188 [0.08, 0.31] | 21.7 | 59.3 |
| routed_best_of_k | 0.581 [0.54, 0.62] | 0.786 [0.68, 0.89] | 0.497 [0.44, 0.55] | 0.732 [0.61, 0.84] | 0.729 [0.60, 0.85] | 14.7 | 25.3 |
| agent_rule@testrule | 0.546 [0.50, 0.59] | 0.732 [0.61, 0.84] | 0.472 [0.42, 0.53] | 0.732 [0.61, 0.84] | 0.604 [0.46, 0.75] | 11.5 | 21.1 |
| agent_rule_sdxl@testsdxl | 0.342 [0.30, 0.38] | 0.554 [0.43, 0.68] | 0.297 [0.25, 0.35] | 0.571 [0.45, 0.70] | 0.125 [0.04, 0.23] | 14.8 | 43.4 |

## Agent minus baseline (same prompts)

| Comparison | all | counting | genai | spatial | text |
| --- | --- | --- | --- | --- | --- |
| agent_rule@testrule - agent_rule_sdxl@testsdxl | 0.204 [0.16, 0.25] | 0.179 [0.04, 0.34] | 0.175 [0.13, 0.22] | 0.161 [0.04, 0.29] | 0.479 [0.33, 0.63] |
| agent_rule@testrule - best_of_4 | 0.179 [0.14, 0.22] | 0.143 [-0.02, 0.29] | 0.150 [0.10, 0.20] | 0.179 [0.04, 0.32] | 0.417 [0.25, 0.58] |
| agent_rule@testrule - routed_best_of_k | -0.035 [-0.08, 0.01] | -0.054 [-0.18, 0.07] | -0.025 [-0.07, 0.02] | 0.000 [-0.12, 0.12] | -0.125 [-0.29, 0.04] |
| agent_rule@testrule - single | 0.300 [0.26, 0.34] | 0.446 [0.30, 0.59] | 0.219 [0.17, 0.26] | 0.482 [0.34, 0.62] | 0.458 [0.29, 0.62] |
| agent_rule_sdxl@testsdxl - best_of_4 | -0.025 [-0.06, 0.01] | -0.036 [-0.20, 0.12] | -0.025 [-0.06, 0.01] | 0.018 [-0.12, 0.18] | -0.062 [-0.19, 0.06] |
| agent_rule_sdxl@testsdxl - routed_best_of_k | -0.240 [-0.29, -0.19] | -0.232 [-0.39, -0.07] | -0.200 [-0.26, -0.14] | -0.161 [-0.30, -0.02] | -0.604 [-0.75, -0.46] |
| agent_rule_sdxl@testsdxl - single | 0.096 [0.06, 0.13] | 0.268 [0.12, 0.41] | 0.044 [0.02, 0.07] | 0.321 [0.20, 0.46] | -0.021 [-0.12, 0.08] |

agent_rule@testrule: prompts whose final image still fails, by the diagnosis of the first image: low_alignment 140, missing_object 1, missing_text 2, misspelled_text 14, passed own checks 44, wrong_count 11, wrong_relation 6.

agent_rule_sdxl@testsdxl: prompts whose final image still fails, by the diagnosis of the first image: low_alignment 198, missing_object 4, missing_text 2, misspelled_text 36, passed own checks 44, wrong_count 20, wrong_relation 12.
