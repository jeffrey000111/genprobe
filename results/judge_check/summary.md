# Blind human check of the judge's pass / fail calls (module C, one annotator)

Made 2026-10-23T07:22:58.841+00:00 at commit `b334a6e` by `python -m genprobe.judge_check analyze`; protocol in `docs/judge_check_protocol.md` (D-033). Judge pass = judge score == 5. Labelled: 163 of 163 main-part items, 50 test prompts. 95 % CIs from resampling prompts.

- Attention checks: 10 of 10 right (pass mark 9 of 10): **passed**.
- Hidden repeats: same answer on 0.800 of 15 images; Cohen's κ 0.602.
- Answers on the main images: no 49, unsure 16, yes 73.

## Judge vs person (images answered yes or no)

| Measure | Value [95 % CI] | Images |
| --- | --- | --- |
| Same call (accuracy) | 0.762 [0.67, 0.85] | 122 |
| Cohen's κ | 0.532 [0.36, 0.71] | 122 |
| Judge passes the person also passes | 0.893 [0.81, 0.96] | 56 |
| Judge fails the person also fails | 0.652 [0.51, 0.79] | 66 |
| Person's passes the judge also passes | 0.685 [0.55, 0.82] | 73 |

Counts: judge fail / person no 43, judge fail / person yes 23, judge pass / person no 6, judge pass / person yes 50.

By category (exploratory): genai 0.731 (κ 0.49, 78 images); counting 0.867 (κ 0.70, 15 images); spatial 0.714 (κ 0.43, 14 images); text 0.867 (κ 0.74, 15 images).

## Pass rates on the same images

Person: yes = pass (no and unsure = not). Judge: score 5.

| Group | Person | Judge | Unsure | Judge passes the person does not pass |
| --- | --- | --- | --- | --- |
| best_of_4 | 0.440 [0.30, 0.58] | 0.320 [0.20, 0.46] | 8 | 0.062 [0.00, 0.19] (of 16) |
| routed_best_of_k | 0.580 [0.44, 0.72] | 0.580 [0.44, 0.72] | 4 | 0.241 [0.10, 0.41] (of 29) |
| agent_rule | 0.560 [0.42, 0.70] | 0.440 [0.30, 0.58] | 6 | 0.227 [0.05, 0.41] (of 22) |

| Paired difference (same prompts) | Person | Judge |
| --- | --- | --- |
| agent_rule - routed_best_of_k | -0.020 [-0.14, 0.10] | -0.140 [-0.28, 0.00] |
| agent_rule - best_of_4 | 0.120 [-0.04, 0.28] | 0.120 [-0.02, 0.28] |
| routed_best_of_k - best_of_4 | 0.140 [-0.04, 0.32] | 0.260 [0.12, 0.40] |
