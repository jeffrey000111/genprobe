# The agent's video threshold, chosen on dev (D-038)

Made 2026-10-25T09:39:12.609+00:00 at commit `91b5b9d` by `python -m genprobe.story tune`. Arm B's 25 dev shots; the judge (Gemma 4 31B, 3 frames, pass = score 5) passes 0.520 of them. A shot passes the agent's check when its VQAScore on the whole video is at least tau_v.

| tau_v | Balanced accuracy against the judge | Accuracy | Shots passing the check |
| --- | --- | --- | --- |
| 0.5 | 0.625 | 0.640 | 0.880 |
| 0.6 | 0.667 | 0.680 | 0.840 |
| 0.7 | 0.667 | 0.680 | 0.840 |
| 0.8 | 0.667 | 0.680 | 0.840 |
| 0.9 | 0.628 | 0.640 | 0.800 |

Chosen: **tau_v = 0.6** (highest balanced accuracy, ties to the lower).
