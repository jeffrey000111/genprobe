# Agent run `devsdxl2` (rule_sdxl policy, dev split)

Made 2026-10-23T00:51:24.221+00:00 at commit `f2718a2` by `python -m genprobe.agent.run report`. 120 prompts. **Pass rates here use the agent's own checkers** (detector, OCR, VQAScore), which the agent optimizes; the comparison that counts is the independent judge's (D-030).

| Group | Category | Prompts | Passes own checks |
| --- | --- | --- | --- |
| agent_rule_sdxl | all | 120 | 0.425 |
| agent_rule_sdxl | counting | 14 | 0.286 |
| agent_rule_sdxl | genai | 80 | 0.450 |
| agent_rule_sdxl | spatial | 14 | 0.429 |
| agent_rule_sdxl | text | 12 | 0.417 |
| best_of_4 | all | 120 | 0.433 |
| best_of_4 | counting | 14 | 0.286 |
| best_of_4 | genai | 80 | 0.463 |
| best_of_4 | spatial | 14 | 0.357 |
| best_of_4 | text | 12 | 0.500 |
| routed_best_of_k | all | 120 | 0.675 |
| routed_best_of_k | counting | 14 | 0.500 |
| routed_best_of_k | genai | 80 | 0.637 |
| routed_best_of_k | spatial | 14 | 0.786 |
| routed_best_of_k | text | 12 | 1.000 |
| single | all | 120 | 0.242 |
| single | counting | 14 | 0.071 |
| single | genai | 80 | 0.312 |
| single | spatial | 14 | 0.071 |
| single | text | 12 | 0.167 |

Agent: 2.78 attempts per prompt on average (the first is the base image); spent 13.8 of 20.7 GPU-seconds on average; 0 prompts over budget.
Stops: {'passed': 51, 'no affordable action left': 69}. Actions taken: {'raise_cfg': 70, 'reseed': 119, 'rewrite': 25}. The final image came from: {'reseed': 40, 'raise_cfg': 17, 'rewrite': 8} (prompts where it is not the base image).

Routed best-of-k used k = {'genai:4': 80, 'counting:3': 14, 'spatial:3': 14, 'text:4': 12} (category:k: prompts).
Costs (median seconds per image): {"gen": {"sdxl": 3.0505, "sd35m": 4.6018, "flux_schnell": 1.1286}, "score": {"vqascore": 0.8519, "detect": 4.4642, "ocr": 2.3489}}.
Settings: {"ruleset": "r2", "tau_genai": 0.9, "tau_template_vqa": 0.9, "delta": 0.02, "cfg_high": 7.5, "route": {"counting": "sd35m", "spatial": "sd35m", "text": "flux_schnell", "genai": "flux_schnell"}, "rules": {"wrong_count": ["switch_model", "rewrite", "reseed"], "missing_object": ["switch_model", "rewrite", "reseed"], "wrong_relation": ["switch_model", "rewrite", "reseed"], "misspelled_text": ["switch_model", "reseed", "rewrite"], "missing_text": ["switch_model", "rewrite", "reseed"], "low_alignment": ["switch_model", "raise_cfg", "reseed"]}, "max_attempts": 12}.
