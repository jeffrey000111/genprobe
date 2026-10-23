# Agent run `devrule1` (rule policy, dev split)

Made 2026-10-22T23:42:10.997+00:00 at commit `69cc57d` by `python -m genprobe.agent.run report`. 120 prompts. **Pass rates here use the agent's own checkers** (detector, OCR, VQAScore), which the agent optimizes; the comparison that counts is the independent judge's (D-030).

| Group | Category | Prompts | Passes own checks |
| --- | --- | --- | --- |
| agent_rule | all | 120 | 0.667 |
| agent_rule | counting | 14 | 0.786 |
| agent_rule | genai | 80 | 0.550 |
| agent_rule | spatial | 14 | 1.000 |
| agent_rule | text | 12 | 0.917 |
| best_of_4 | all | 120 | 0.533 |
| best_of_4 | counting | 14 | 0.643 |
| best_of_4 | genai | 80 | 0.463 |
| best_of_4 | spatial | 14 | 0.857 |
| best_of_4 | text | 12 | 0.500 |
| routed_best_of_k | all | 120 | 0.725 |
| routed_best_of_k | counting | 14 | 0.786 |
| routed_best_of_k | genai | 80 | 0.637 |
| routed_best_of_k | spatial | 14 | 0.929 |
| routed_best_of_k | text | 12 | 1.000 |
| single | all | 120 | 0.325 |
| single | counting | 14 | 0.357 |
| single | genai | 80 | 0.312 |
| single | spatial | 14 | 0.500 |
| single | text | 12 | 0.167 |

Agent: 2.62 attempts per prompt on average (the first is the base image); spent 11.2 of 20.7 GPU-seconds on average; 0 prompts over budget.
Stops: {'passed': 80, 'no affordable action left': 40}. Actions taken: {'raise_cfg': 55, 'switch_model': 73, 'reseed': 50, 'rewrite': 17}. The final image came from: {'switch_model': 42, 'reseed': 19, 'raise_cfg': 7, 'rewrite': 1} (prompts where it is not the base image).

Routed best-of-k used k = {'genai:4': 80, 'counting:3': 14, 'spatial:3': 14, 'text:4': 12} (category:k: prompts).
Costs (median seconds per image): {"gen": {"sdxl": 3.0505, "sd35m": 4.6018, "flux_schnell": 1.1286}, "score": {"vqascore": 0.8519, "detect": 4.4642, "ocr": 2.3489}}.
Settings: {"tau_genai": 0.9, "delta": 0.02, "cfg_high": 7.5, "route": {"counting": "sd35m", "spatial": "sd35m", "text": "flux_schnell", "genai": "flux_schnell"}, "rules": {"wrong_count": ["rewrite", "reseed", "switch_model", "reseed"], "missing_object": ["rewrite", "switch_model", "reseed"], "wrong_relation": ["rewrite", "switch_model", "reseed"], "misspelled_text": ["switch_model", "reseed", "rewrite"], "missing_text": ["switch_model", "rewrite", "reseed"], "low_alignment": ["raise_cfg", "switch_model", "reseed"]}, "max_attempts": 8}.
