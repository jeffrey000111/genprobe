# Agent run `devrule1` graded by the independent judge

Made 2026-10-22T23:47:08.201+00:00 at commit `69cc57d` by `python -m genprobe.agent.compare`. Judge `vllm==0.32.0/gemma-4-31b@842da37/bf16/vis560/img1024/tmpl-b9e98ed8` (grades `grade_devrule1_gemma4_31b`), original prompts, pass = judge score == 5. 120 prompts; 95 % CIs from resampling prompts.

## Pass rate

| Group | all | counting | genai | spatial | text | GPU-s / prompt | GPU-s / success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | 0.250 [0.17, 0.33] | 0.286 [0.07, 0.57] | 0.263 [0.17, 0.36] | 0.286 [0.07, 0.50] | 0.083 [0.00, 0.25] | 5.3 | 21.2 |
| best_of_4 | 0.425 [0.34, 0.51] | 0.643 [0.36, 0.86] | 0.412 [0.30, 0.51] | 0.500 [0.21, 0.79] | 0.167 [0.00, 0.42] | 20.7 | 48.7 |
| routed_best_of_k | 0.600 [0.52, 0.68] | 0.786 [0.57, 1.00] | 0.512 [0.40, 0.62] | 0.643 [0.36, 0.86] | 0.917 [0.75, 1.00] | 14.0 | 23.3 |
| agent_rule | 0.550 [0.46, 0.64] | 0.714 [0.50, 0.93] | 0.500 [0.39, 0.61] | 0.571 [0.36, 0.79] | 0.667 [0.42, 0.92] | 11.2 | 20.4 |

## Agent minus baseline (same prompts)

| Comparison | all | counting | genai | spatial | text |
| --- | --- | --- | --- | --- | --- |
| agent_rule - best_of_4 | 0.125 [0.03, 0.22] | 0.071 [-0.21, 0.36] | 0.087 [-0.01, 0.20] | 0.071 [-0.21, 0.36] | 0.500 [0.25, 0.75] |
| agent_rule - routed_best_of_k | -0.050 [-0.14, 0.05] | -0.071 [-0.36, 0.21] | -0.013 [-0.12, 0.10] | -0.071 [-0.36, 0.21] | -0.250 [-0.50, -0.08] |
| agent_rule - single | 0.300 [0.22, 0.38] | 0.429 [0.14, 0.64] | 0.237 [0.15, 0.34] | 0.286 [0.07, 0.50] | 0.583 [0.33, 0.83] |

Prompts the agent's final image still fails, by the diagnosis of the first image: low_alignment 34, misspelled_text 3, passed own checks 12, wrong_count 3, wrong_relation 2.
