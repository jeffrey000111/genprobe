# Blind human check, round 2: DPO vs base SD 1.5 (one annotator, D-037)

Made 2026-10-25T02:15:13.372+00:00 at commit `1d80c26` by `python -m genprobe.dpo_check analyze`; protocol in `docs/dpo_check_protocol.md`. Labelled 138 of 138 items. 95 % CIs from resampling prompts.

- Attention checks (part 1): 5 of 6 right (pass mark 5 of 6): **passed**.

## Which image matches the text better?

Answers: base 3, neither 43, new 4, same 10. Hidden repeats: same answer on 1.000 of 6 (κ 1.000).

Share of decided pairs where the DPO image was picked: **0.571** [0.14, 0.86] over 7 pairs (0.5 = no difference).

By stratum: counting 0.50 (n=2); spatial 1.00 (n=1); text – (n=0); genai 0.50 (n=4).

## Which image looks better?

Answers: base 3, new 5, same 52. Hidden repeats: same answer on 1.000 of 6 (κ 1.000).

Share of decided pairs where the DPO image was picked: **0.625** [0.25, 0.88] over 8 pairs (0.5 = no difference).

By stratum: counting 0.50 (n=2); spatial 1.00 (n=1); text 1.00 (n=1); genai 0.50 (n=4).
