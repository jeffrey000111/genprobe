# Blind human check: DPO vs base SD 1.5 (one annotator, D-037)

Made 2026-10-25T01:47:37.697+00:00 at commit `e9a97fd` by `python -m genprobe.dpo_check analyze`; protocol in `docs/dpo_check_protocol.md`. Labelled 138 of 138 items. 95 % CIs from resampling prompts.

- Attention checks (part 1): 0 of 6 right (pass mark 5 of 6): **NOT passed**.

## Which image matches the text better?

Answers: base 2, neither 45, new 4, same 9. Hidden repeats: same answer on 0.833 of 6 (κ 0.571).

Share of decided pairs where the DPO image was picked: **0.667** [0.33, 1.00] over 6 pairs (0.5 = no difference).

By stratum: counting 1.00 (n=2); spatial 0.50 (n=2); text 0.00 (n=1); genai 1.00 (n=1).

## Which image looks better?

Answers: base 2, new 3, same 55. Hidden repeats: same answer on 1.000 of 6 (κ 1.000).

Share of decided pairs where the DPO image was picked: **0.600** [0.20, 1.00] over 5 pairs (0.5 = no difference).

By stratum: counting 1.00 (n=2); spatial 0.50 (n=2); text 0.00 (n=1); genai – (n=0).
