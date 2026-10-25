# DPO training record (D-037)

Made 2026-10-25T02:20:06.614+00:00 at commit `a6664de` by `python -m genprobe.dpo training` from `dpo_94406.out` and the pairs' metadata.

Pairs: 821 from 1200 training prompts (4800 candidate images); 379 prompts dropped because their highest and lowest VQAScore differ by less than 0.1; mean gap of the kept pairs 0.575.

Training: 2000 steps; the values the progress bar shows for single batches (6001 logged), averaged per window of 250 steps.

| Steps | Logged values | Implicit accuracy | DPO loss |
| --- | --- | --- | --- |
| 1–250 | 750 | 0.612 | 0.679 |
| 251–500 | 750 | 0.676 | 0.655 |
| 501–750 | 750 | 0.711 | 0.630 |
| 751–1000 | 750 | 0.722 | 0.613 |
| 1001–1250 | 750 | 0.744 | 0.591 |
| 1251–1500 | 750 | 0.749 | 0.573 |
| 1501–1750 | 750 | 0.762 | 0.556 |
| 1751–2000 | 750 | 0.780 | 0.537 |

Both are measured on the training pairs. The loss is ln 2 = 0.693 while the LoRA and the reference model agree; implicit accuracy is near 0.5 while the LoRA has learned nothing about the pairs.
