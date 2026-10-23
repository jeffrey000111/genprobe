# Agent run `testrule` (rule policy, test split)

Made 2026-10-23T03:36:28.331+00:00 at commit `01855d0` by `python -m genprobe.agent.run report`. 480 prompts. **Pass rates here use the agent's own checkers** (detector, OCR, VQAScore), which the agent optimizes; the comparison that counts is the independent judge's (D-030).

| Group | Category | Prompts | Passes own checks |
| --- | --- | --- | --- |
| agent_rule | all | 480 | 0.729 |
| agent_rule | counting | 56 | 0.696 |
| agent_rule | genai | 320 | 0.697 |
| agent_rule | spatial | 56 | 0.804 |
| agent_rule | text | 48 | 0.896 |
| best_of_4 | all | 480 | 0.477 |
| best_of_4 | counting | 56 | 0.357 |
| best_of_4 | genai | 320 | 0.497 |
| best_of_4 | spatial | 56 | 0.536 |
| best_of_4 | text | 48 | 0.417 |
| routed_best_of_k | all | 480 | 0.700 |
| routed_best_of_k | counting | 56 | 0.661 |
| routed_best_of_k | genai | 320 | 0.656 |
| routed_best_of_k | spatial | 56 | 0.804 |
| routed_best_of_k | text | 48 | 0.917 |
| single | all | 480 | 0.265 |
| single | counting | 56 | 0.143 |
| single | genai | 320 | 0.322 |
| single | spatial | 56 | 0.214 |
| single | text | 48 | 0.083 |

Agent: 2.78 attempts per prompt on average (the first is the base image); spent 11.5 of 21.7 GPU-seconds on average; 0 prompts over budget.
Stops: {'passed': 350, 'no affordable action left': 130}. Actions taken: {'switch_model': 357, 'raise_cfg': 75, 'reseed': 369, 'rewrite': 53}. The final image came from: {'switch_model': 166, 'reseed': 102, 'raise_cfg': 14, 'rewrite': 21} (prompts where it is not the base image).

Routed best-of-k used k = {'genai:4': 320, 'counting:3': 56, 'spatial:3': 56, 'text:4': 48} (category:k: prompts).
Costs (median seconds per image): {"gen": {"sdxl": 3.0504, "sd35m": 4.6027, "flux_schnell": 1.128}, "score": {"vqascore": 0.8518, "detect": 5.5647, "ocr": 2.3489}}.
Settings: {"ruleset": "r2", "tau_genai": 0.9, "tau_template_vqa": 0.9, "delta": 0.02, "cfg_high": 7.5, "route": {"counting": "sd35m", "spatial": "sd35m", "text": "flux_schnell", "genai": "flux_schnell"}, "rules": {"wrong_count": ["switch_model", "rewrite", "reseed"], "missing_object": ["switch_model", "rewrite", "reseed"], "wrong_relation": ["switch_model", "rewrite", "reseed"], "misspelled_text": ["switch_model", "reseed", "rewrite"], "missing_text": ["switch_model", "rewrite", "reseed"], "low_alignment": ["switch_model", "raise_cfg", "reseed"]}, "max_attempts": 12}.
