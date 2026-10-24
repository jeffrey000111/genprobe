# DPO checkpoint chosen on the dev prompts (D-037)

Made 2026-10-24T11:50:09.121+00:00 at commit `aae6d90` by `python -m genprobe.dpo select`. Judge pass rate (score 5) on seed 0 of 120 dev prompts, grades `grade_dpo_dev_full_gemma4_31b`; highest wins, ties go to fewer steps.

| Arm | Pass rate [95 % CI] |
| --- | --- |
| dpo_base | 0.150 [0.09, 0.22] |
| dpo_s250 | 0.133 [0.07, 0.20] |
| dpo_s500 | 0.150 [0.09, 0.22] |
| dpo_s1000 | 0.150 [0.09, 0.22] |
| dpo_s2000 | 0.183 [0.12, 0.25] |

Selected: **dpo_s2000**.
