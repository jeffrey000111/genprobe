"""Leaderboard, paired comparisons and figures (genprobe/figures.py) on a tiny synthetic set."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from genprobe import figures
from genprobe.samples import Prompt


class TestHelpers(unittest.TestCase):
    def test_paired(self):
        a = {"p1": [0.9, 0.7], "p2": [0.5], "p3": [0.2]}
        b = {"p1": [0.5], "p2": [0.5], "p4": [0.0]}
        r = figures.paired(a, b)
        self.assertEqual(r["n_prompts"], 2)                  # only prompts both models have
        self.assertAlmostEqual(r["mean_diff"], (0.8 - 0.5 + 0.0) / 2)
        self.assertEqual((r["win_rate"], r["tie_rate"]), (0.5, 0.5))
        self.assertTrue(math.isnan(figures.paired({}, b)["mean_diff"]))

    def test_pareto_front(self):
        pts = {"cheap": (0.5, 0.4), "best": (1.1, 0.8), "dominated": (2.0, 0.6), "slow_top": (5.0, 0.9)}
        self.assertEqual(figures.pareto_front(pts), ["cheap", "best", "slow_top"])

    def test_groups_and_text_bins(self):
        g = Prompt("gb1", "x", source="genai_bench", tags=["basic", "counting", "spatial_relation"])
        t = Prompt("tc1", "x", source="template_count", tags=["counting"])
        self.assertEqual(figures.prompt_groups(g), ["GenAI · counting", "GenAI · spatial relation"])
        self.assertEqual(figures.prompt_groups(t), ["template · counting"])
        self.assertEqual(figures.text_bin(Prompt("t", "x", spec={"text": "PLEASE WAIT HERE"})), "3–4 words")
        self.assertEqual(figures.text_bin(Prompt("t", "x", spec={"text": "A B C D E F G H"})), "5–8 words")
        self.assertIsNone(figures.text_bin(t))

    def test_model_order_is_fixed(self):
        self.assertEqual(figures.ordered_models({"flux_schnell", "zzz", "sd15"}), ["sd15", "flux_schnell", "zzz"])


class TestRun(unittest.TestCase):
    def test_writes_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prompts = [
                {"prompt_id": "gb0001", "prompt": "a", "source": "genai_bench", "tags": ["basic", "scene"],
                 "split": "test"},
                {"prompt_id": "tt0001", "prompt": "b", "source": "template_text", "tags": ["text"],
                 "spec": {"text": "SALE"}, "split": "test"},
                {"prompt_id": "tt0002", "prompt": "c", "source": "template_text", "tags": ["text"],
                 "spec": {"text": "OPEN EVERY DAY UNTIL NINE"}, "split": "test"},
                {"prompt_id": "tt0003", "prompt": "d", "source": "template_text", "tags": ["text"],
                 "spec": {"text": "HOT"}, "split": "dev"},
            ]
            (d / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
            cols = ["sample_id", "model", "prompt_id", "seed", "category", "prompt", "status", "gen_seconds",
                    "warmup", "vqascore", "ocr", "clip_truncated", "human_correct", "human_note"]
            with open(d / "per_sample.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for model, v, t in (("sd15", 0.3, 0.5), ("flux_schnell", 0.8, 1.2)):
                    for p in prompts:
                        for s in (0, 1):
                            w.writerow({"sample_id": f"{model}__{p['prompt_id']}__s{s}", "model": model,
                                        "prompt_id": p["prompt_id"], "seed": s, "category": p["tags"][0],
                                        "prompt": p["prompt"], "status": "ok", "gen_seconds": t,
                                        "warmup": s == 0 and p["prompt_id"] == "gb0001", "vqascore": v,
                                        "ocr": (1.0 if model == "flux_schnell" else 0.5) if p["tags"] == ["text"] else "",
                                        "clip_truncated": False, "human_correct": "", "human_note": ""})
            res = figures.run(d, str(d / "p.jsonl"), split="test")
            for f in ("leaderboard.csv", "pairwise.csv", "figures.md", "fig_category_heatmap.png",
                      "fig_frontier.png", "fig_text_length.png", "text_length.csv"):
                self.assertTrue((d / f).is_file(), f)
            board = {(r["model"], r["metric"]): r for r in res["leaderboard"]}
            self.assertEqual(board[("sd15", "vqascore")]["n_prompts"], 3)        # dev prompt left out
            self.assertAlmostEqual(board[("flux_schnell", "vqascore")]["mean"], 0.8)
            self.assertEqual(board[("sd15", "ocr")]["n_prompts"], 2)
            pair = [r for r in res["pairwise"] if r["metric"] == "vqascore"][0]
            self.assertEqual((pair["model_a"], pair["model_b"]), ("sd15", "flux_schnell"))
            self.assertAlmostEqual(pair["mean_diff"], -0.5)
            self.assertTrue(pair["ci_excludes_0"])
            self.assertIn("Highest vqascore: FLUX.1 schnell", res["titles"]["fig_frontier.png"])
            self.assertIn("Exact text: 50% for both 1–2 words and 5–8 words", res["titles"]["fig_text_length.png"])


if __name__ == "__main__":
    unittest.main()
