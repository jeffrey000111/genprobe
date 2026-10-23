# Agent run(s) `devrule2`, `devsdxl2` graded by the independent judge

Made 2026-10-23T00:54:39.373+00:00 at commit `f2718a2` by `python -m genprobe.agent.compare`. Judge `vllm==0.32.0/gemma-4-31b@842da37/bf16/vis560/img1024/tmpl-b9e98ed8` (grades `grade_devrule2_gemma4_31b`, `grade_devsdxl2_gemma4_31b`), original prompts, pass = judge score == 5. 120 prompts; 95 % CIs from resampling prompts.

## Pass rate

| Group | all | counting | genai | spatial | text | GPU-s / prompt | GPU-s / success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.242 [0.17, 0.32] | 0.286 [0.07, 0.57] | 0.250 [0.16, 0.35] | 0.286 [0.07, 0.50] | 0.083 [0.00, 0.25] | 5.3 | 21.9 |
| best_of_4 | 0.417 [0.33, 0.50] | 0.643 [0.36, 0.86] | 0.400 [0.29, 0.50] | 0.500 [0.21, 0.79] | 0.167 [0.00, 0.42] | 20.7 | 49.7 |
| routed_best_of_k | 0.600 [0.52, 0.68] | 0.786 [0.57, 1.00] | 0.512 [0.40, 0.62] | 0.643 [0.36, 0.86] | 0.917 [0.75, 1.00] | 14.0 | 23.3 |
| agent_rule@devrule2 | 0.533 [0.45, 0.62] | 0.786 [0.57, 1.00] | 0.425 [0.31, 0.54] | 0.786 [0.57, 1.00] | 0.667 [0.42, 0.92] | 11.7 | 21.9 |
| agent_rule_sdxl@devsdxl2 | 0.375 [0.29, 0.46] | 0.571 [0.29, 0.79] | 0.350 [0.25, 0.45] | 0.571 [0.29, 0.79] | 0.083 [0.00, 0.25] | 13.8 | 36.8 |

## Agent minus baseline (same prompts)

| Comparison | all | counting | genai | spatial | text |
| --- | --- | --- | --- | --- | --- |
| agent_rule@devrule2 - agent_rule_sdxl@devsdxl2 | 0.158 [0.07, 0.24] | 0.214 [-0.07, 0.50] | 0.075 [-0.01, 0.16] | 0.214 [0.00, 0.43] | 0.583 [0.33, 0.83] |
| agent_rule@devrule2 - best_of_4 | 0.117 [0.03, 0.21] | 0.143 [-0.21, 0.50] | 0.025 [-0.07, 0.12] | 0.286 [-0.07, 0.57] | 0.500 [0.25, 0.75] |
| agent_rule@devrule2 - routed_best_of_k | -0.067 [-0.16, 0.03] | 0.000 [-0.36, 0.36] | -0.087 [-0.20, 0.03] | 0.143 [-0.14, 0.43] | -0.250 [-0.50, -0.08] |
| agent_rule@devrule2 - single | 0.292 [0.22, 0.38] | 0.500 [0.21, 0.79] | 0.175 [0.10, 0.26] | 0.500 [0.21, 0.79] | 0.583 [0.33, 0.83] |
| agent_rule_sdxl@devsdxl2 - best_of_4 | -0.042 [-0.12, 0.03] | -0.071 [-0.29, 0.14] | -0.050 [-0.14, 0.04] | 0.071 [-0.29, 0.36] | -0.083 [-0.25, 0.00] |
| agent_rule_sdxl@devsdxl2 - routed_best_of_k | -0.225 [-0.33, -0.12] | -0.214 [-0.50, 0.07] | -0.163 [-0.29, -0.03] | -0.071 [-0.36, 0.21] | -0.833 [-1.00, -0.58] |
| agent_rule_sdxl@devsdxl2 - single | 0.133 [0.07, 0.20] | 0.286 [0.07, 0.50] | 0.100 [0.04, 0.17] | 0.286 [0.07, 0.50] | 0.000 [0.00, 0.00] |

agent_rule@devrule2: prompts whose final image still fails, by the diagnosis of the first image: low_alignment 42, misspelled_text 3, passed own checks 8, wrong_count 2, wrong_relation 1.

agent_rule_sdxl@devsdxl2: prompts whose final image still fails, by the diagnosis of the first image: low_alignment 48, misspelled_text 10, passed own checks 8, wrong_count 6, wrong_relation 3.
