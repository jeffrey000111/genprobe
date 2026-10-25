# Multi-shot videos from a brief (test, D-038)

Made 2026-10-25T19:09:13.681+00:00 at commit `5c4ccbc` by `python -m genprobe.story analyze`. 24 brief(s), 78 shots; Wan2.2 TI2V-5B, 49 frames per shot. Judge: Gemma 4 31B on the first, middle and last frame of a shot (shot) or the middle frame of every shot (story, consistency); pass = score 5. 95 % CIs from resampling briefs. Agent thresholds: keyframe VQAScore 0.9, video VQAScore 0.6.

| Arm | Shot pass rate | Shot expected score | VQAScore (video) | Story pass | Story score | Consistency pass | Consistency score | GPU-s per brief |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A text to video | 0.462 [0.34, 0.58] | 4.36 [4.19, 4.52] | 0.854 [0.79, 0.91] | 0.542 [0.33, 0.75] | 4.42 [4.12, 4.67] | 0.250 [0.08, 0.42] | 3.12 [2.67, 3.58] | 212 |
| B keyframe → image to video | 0.500 [0.38, 0.62] | 4.45 [4.31, 4.60] | 0.936 [0.90, 0.96] | 0.708 [0.54, 0.88] | 4.71 [4.54, 4.88] | 0.292 [0.12, 0.50] | 3.62 [3.17, 4.04] | 215 |
| C agent (checks + redo) | 0.526 [0.42, 0.63] | 4.47 [4.34, 4.59] | 0.939 [0.90, 0.96] | 0.667 [0.46, 0.83] | 4.67 [4.46, 4.83] | 0.250 [0.08, 0.42] | 3.46 [3.00, 3.92] | 242 |
| D re-ranking (2 videos per shot) | 0.513 [0.41, 0.63] | 4.48 [4.36, 4.61] | 0.942 [0.91, 0.97] | 0.625 [0.42, 0.83] | 4.62 [4.42, 4.83] | 0.375 [0.21, 0.58] | 3.62 [3.12, 4.08] | 434 |

| Test | Difference [CI] | CI level | Verdict |
| --- | --- | --- | --- |
| H7: shot pass rate, B - A | 0.038 [-0.15, 0.21] | 0.975 | inconclusive |
| H8: shot pass rate, C - B | 0.026 [-0.04, 0.09] | 0.975 | inconclusive |
| C - D (exploratory): shot pass rate, C - D | 0.013 [-0.06, 0.09] | 0.95 | exploratory |
| C - A (exploratory): shot pass rate, C - A | 0.064 [-0.09, 0.20] | 0.95 | exploratory |

Exploratory, added after the test results were seen: paired differences per brief.

| Difference | Value [95 % CI] |
| --- | --- |
| story score, B - A | 0.29 [0.04, 0.54] |
| consist score, B - A | 0.50 [0.00, 1.04] |
| story score, C - B | -0.04 [-0.12, 0.00] |
| consist score, C - B | -0.17 [-0.38, 0.00] |
| story score, C - A | 0.25 [0.04, 0.46] |
| consist score, C - A | 0.33 [-0.17, 0.83] |
| story score, C - D | 0.04 [-0.08, 0.17] |
| consist score, C - D | -0.17 [-0.38, 0.00] |
| story score, D - B | -0.08 [-0.25, 0.08] |
| consist score, D - B | 0.00 [-0.17, 0.17] |

Agent: keyframe redone on 0.205 of shots, video redone on 0.051; 1.33 keyframes and 1.09 videos per shot; 0.026 of shots ended below tau_v.

GPU-seconds: generation plus the checks' VQAScore runs, measured per item; planning (the same for every arm) and model loading are not counted.
