# Multi-shot videos from a brief (smoke, D-038)

Made 2026-10-25T05:47:42.362+00:00 at commit `d0c19b3` by `python -m genprobe.story analyze`. 1 brief(s), 4 shots; Wan2.2 TI2V-5B, 49 frames per shot. Judge: Gemma 4 31B on the first, middle and last frame of a shot (shot) or the middle frame of every shot (story, consistency); pass = score 5. 95 % CIs from resampling briefs. Agent thresholds: keyframe VQAScore 0.9, video VQAScore 0.9.

**Smoke run: a pipeline check only; no number here is a result.**

| Arm | Shot pass rate | Shot expected score | VQAScore (video) | Story pass | Story score | Consistency pass | Consistency score | GPU-s per brief |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A text to video | 0.250 [0.25, 0.25] | 4.05 [4.05, 4.05] | 0.909 [0.91, 0.91] | 0.000 [0.00, 0.00] | 4.00 [4.00, 4.00] | 0.000 [0.00, 0.00] | 3.00 [3.00, 3.00] | 263 |
| B keyframe → image to video | 0.500 [0.50, 0.50] | 4.50 [4.50, 4.50] | 0.814 [0.81, 0.81] | 1.000 [1.00, 1.00] | 5.00 [5.00, 5.00] | 0.000 [0.00, 0.00] | 4.00 [4.00, 4.00] | 277 |
| C agent (checks + redo) | 0.750 [0.75, 0.75] | 4.75 [4.75, 4.75] | 0.967 [0.97, 0.97] | 1.000 [1.00, 1.00] | 5.00 [5.00, 5.00] | 1.000 [1.00, 1.00] | 5.00 [5.00, 5.00] | 283 |
| D re-ranking (2 videos per shot) | 0.500 [0.50, 0.50] | 4.48 [4.48, 4.48] | 0.852 [0.85, 0.85] | 1.000 [1.00, 1.00] | 5.00 [5.00, 5.00] | 0.000 [0.00, 0.00] | 3.00 [3.00, 3.00] | 550 |

| Test | Difference [CI] | CI level | Verdict |
| --- | --- | --- | --- |
| H7: shot pass rate, B - A | 0.250 [0.25, 0.25] | 0.975 | supported |
| H8: shot pass rate, C - B | 0.250 [0.25, 0.25] | 0.975 | supported |
| C - D (exploratory): shot pass rate, C - D | 0.250 [0.25, 0.25] | 0.95 | exploratory |
| C - A (exploratory): shot pass rate, C - A | 0.500 [0.50, 0.50] | 0.95 | exploratory |

Agent: keyframe redone on 0.250 of shots, video redone on 0.000; 1.25 keyframes and 1.00 videos per shot; 0.000 of shots ended below tau_v.

GPU-seconds: generation plus the checks' VQAScore runs, measured per item; planning (the same for every arm) and model loading are not counted.
