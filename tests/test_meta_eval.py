"""Metric meta-evaluation (genprobe/meta_eval.py): pairs, agreement, ceiling, alpha, and an
end-to-end run on a tiny fake GenAI-Bench with fake metric scores."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from genprobe import genai_bench, meta_eval as me


class TestPieces(unittest.TestCase):
    def test_pairs_skip_equal_means(self):
        pairs, ties, total = me.make_pairs({"p": {"a": 3.0, "b": 3.0, "c": 4.5}})
        self.assertEqual((ties, total), (1, 3))
        self.assertEqual(pairs["p"], [("a", "c", -1), ("b", "c", -1)])

    def test_agree_counts_ties_half(self):
        self.assertEqual(me.agree(0.9, 0.1, 1), 1.0)
        self.assertEqual(me.agree(0.9, 0.1, -1), 0.0)
        self.assertEqual(me.agree(0.5, 0.5, 1), 0.5)

    def test_agreement_and_difference(self):
        means = {f"p{i}": {"a": 1.0, "b": 2.0, "c": 3.0} for i in range(20)}
        pairs, _, _ = me.make_pairs(means)
        pids = sorted(pairs)
        good = {f"{x}": v for x, v in (("a", 0.1), ("b", 0.2), ("c", 0.3))}
        flat = {"a": 0.5, "b": 0.5, "c": 0.5}
        bad = {"a": 0.3, "b": 0.2, "c": 0.1}
        self.assertEqual(me.agreement(pairs, pids, good, n_boot=200)["agreement"], 1.0)
        self.assertEqual(me.agreement(pairs, pids, flat, n_boot=200)["agreement"], 0.5)
        self.assertEqual(me.agreement(pairs, pids, bad, n_boot=200)["agreement"], 0.0)
        d = me.difference(pairs, pids, good, flat, n_boot=200, level=0.975)
        self.assertEqual((d["difference"], d["pairs"], d["n_prompts"], d["level"]), (0.5, 60, 20, 0.975))
        self.assertEqual(me.verdict(d["ci"]), "supported")
        self.assertEqual(me.verdict(me.difference(pairs, pids, bad, good, n_boot=200)["ci"]), "contradicted")
        self.assertEqual(me.verdict([-0.1, 0.2]), "inconclusive")

    def test_difference_uses_pairs_both_scored(self):
        pairs, _, _ = me.make_pairs({"p": {"a": 1.0, "b": 2.0, "c": 3.0}})
        partial = {"a": 0.1, "b": 0.2}                          # c not scored
        full = {"a": 0.3, "b": 0.2, "c": 0.1}
        d = me.difference(pairs, ["p"], partial, full, n_boot=50)
        self.assertEqual((d["pairs"], d["difference"]), (1, 1.0))

    def test_ceiling(self):
        raw = {f"p{i}": {"a": [1.0, 1.0, 1.0], "b": [3.0, 3.0, 3.0], "c": [5.0, 5.0, 5.0]} for i in range(5)}
        means = {p: {k: sum(v) / 3 for k, v in imgs.items()} for p, imgs in raw.items()}
        pairs, _, _ = me.make_pairs(means)
        c = me.ceiling(pairs, sorted(pairs), raw, n_boot=100)
        self.assertEqual((c["agreement"], c["n_prompts"], c["by_position"]), (1.0, 5, [1.0, 1.0, 1.0]))
        # rater 0 always reversed, the other two agree with each other
        raw = {"p": {"a": [5.0, 1.0, 1.0], "b": [1.0, 5.0, 5.0]}}
        pairs, _, _ = me.make_pairs({"p": {k: sum(v) / 3 for k, v in raw["p"].items()}})
        c = me.ceiling(pairs, ["p"], raw, n_boot=10)
        # position 0 says a > b, the other two say b > a; positions 1 and 2: the other two tie (6 vs 6), skipped
        self.assertEqual((c["agreement"], c["n_prompts"], c["pairs"]), (0.0, 1, 1))
        self.assertTrue(math.isnan(c["by_position"][1]))
        raw = {"p": {"a": [2.0, 1.0, 1.0], "b": [1.0, 4.0, 5.0]}}
        pairs, _, _ = me.make_pairs({"p": {k: sum(v) / 3 for k, v in raw["p"].items()}})
        c = me.ceiling(pairs, ["p"], raw, n_boot=10)
        self.assertEqual(c["by_position"], [0.0, 1.0, 1.0])
        self.assertAlmostEqual(c["agreement"], 2 / 3)

    def test_ceiling_skips_images_without_three_ratings(self):
        raw = {"p": {"a": [1.0, 2.0], "b": [4.0, 5.0, 5.0]}}
        pairs, _, _ = me.make_pairs({"p": {"a": 1.5, "b": 14 / 3}})
        self.assertEqual(me.ceiling(pairs, ["p"], raw, n_boot=10)["n_prompts"], 0)

    def test_krippendorff_matches_published_example(self):
        # Krippendorff (2011), "Computing Krippendorff's alpha-reliability", 4 coders x 12 units with
        # missing values; ordinal alpha = 0.815 (nominal 0.743, interval 0.849).
        A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
        B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
        C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
        D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
        units = [[float(x) for x in col if x is not None] for col in zip(A, B, C, D)]
        self.assertAlmostEqual(me.krippendorff_alpha_ordinal(units), 0.815, places=3)
        self.assertEqual(me.krippendorff_alpha_ordinal([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]), 1.0)

    def test_kendall_tau_b(self):
        self.assertEqual(me.kendall_tau_b([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertEqual(me.kendall_tau_b([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertAlmostEqual(me.kendall_tau_b([1, 2, 2, 3], [1, 2, 3, 3]), 0.8)
        self.assertTrue(math.isnan(me.kendall_tau_b([1, 1], [2, 3])))


class TestRun(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            folder = d / "gb"
            folder.mkdir()
            n = 30
            rating = {m: k + 1 for k, m in enumerate(genai_bench.MODELS[:5])}   # 1..5 by model
            rating[genai_bench.MODELS[5]] = 3                                   # ties with the 3rd model
            data = {f"{i:05d}": {"id": f"{i:05d}", "prompt": f"p {i}",
                                 "models": {m: [r, r, r] for m, r in rating.items()}} for i in range(n)}
            data["00000"]["models"]["DALLE_3"] = [1, 41, 1]                     # out of range: dropped
            (folder / "genai_image.json").write_text(json.dumps(data))
            prompts, recs = [], []
            for i in range(n):
                tags = ["basic", "counting"] if i < 10 else ["advanced", "negation"]
                prompts.append({"prompt_id": f"gb{i:04d}", "prompt": f"p {i}", "source": "genai_bench",
                                "tags": tags, "split": "test"})
                for m in genai_bench.MODELS:
                    recs.append({"sample_id": f"{m}__gb{i:04d}__s0", "model": m, "prompt_id": f"gb{i:04d}",
                                 "prompt": f"p {i}", "seed": 0, "path": "x.jpeg", "status": "ok"})
            (d / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
            (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

            def write_scores(metric, fn):
                (d / "scores" / metric).mkdir(parents=True)
                with open(d / "scores" / metric / "x.jsonl", "w") as f:
                    for r in recs:
                        f.write(json.dumps({"sample_id": r["sample_id"], "metric": metric, "metric_version": "v1",
                                            "value": fn(r), "status": "ok", "created_at": "2026-10-19T00:00:00"})
                                + "\n")

            write_scores("vqascore", lambda r: float(rating[r["model"]]))            # orders like people
            write_scores("clipscore", lambda r: 0.25)                              # all equal: 1/2
            write_scores("pickscore", lambda r: -float(rating[r["model"]]))        # reversed
            write_scores("hpsv2", lambda r: 0.25)
            res = me.run(["clipscore", "vqascore", "pickscore", "hpsv2"], str(d / "p.jsonl"), str(d / "m.jsonl"),
                         str(d), str(folder), out=d / "out", n_boot=200)
            self.assertEqual((res["prompts"], res["images"], res["pairs_total"]), (n, 6 * n, 15 * n))
            self.assertEqual(res["pairs_tied_excluded"], n)                         # model 3 vs model 6
            self.assertEqual(len(res["dropped_ratings"]), 1)
            h = res["hypotheses"]
            self.assertEqual((h["H1"]["difference"], h["H1"]["n_prompts"], h["H1"]["verdict"]), (0.5, 10, "supported"))
            self.assertEqual((h["H2a"]["difference"], h["H2a"]["level"], h["H2a"]["verdict"]), (1.0, 0.975, "supported"))
            self.assertEqual(h["H2b"]["difference"], 0.5)
            rows = {(r["metric"], r["tag"]): r for r in res["rows"]}
            self.assertEqual(rows[("vqascore", "negation")]["agreement"], 1.0)
            self.assertEqual(rows[("pickscore", "all")]["agreement"], 0.0)
            self.assertEqual(rows[("vqascore", "all")]["kendall_tau_mean"], 1.0)
            self.assertEqual(rows[(me.HUMAN, "all")]["agreement"], 1.0)
            self.assertEqual([r["tag"] for r in res["rows"] if r["metric"] == "vqascore"],
                             ["all", "basic", "advanced", "counting", "negation"])
            self.assertEqual(res["coverage"]["hpsv2"], {"scored": 180, "images": 180, "errors": 0})
            for f in ("hypotheses.md", "agreement_by_tag.csv", "summary.json", "fig_meta_eval_heatmap.png"):
                self.assertTrue((d / "out" / f).is_file(), f)
            md = (d / "out" / "hypotheses.md").read_text()
            # one image unscored: nothing is computed unless asked
            lines = (d / "scores" / "hpsv2" / "x.jsonl").read_text().splitlines()
            (d / "scores" / "hpsv2" / "x.jsonl").write_text("\n".join(lines[1:]) + "\n")
            with self.assertRaises(me.IncompleteScores):
                me.run(["vqascore", "hpsv2"], str(d / "p.jsonl"), str(d / "m.jsonl"), str(d), str(folder),
                       out=d / "out2", n_boot=50)
            self.assertFalse((d / "out2").exists())
            res = me.run(["vqascore", "hpsv2"], str(d / "p.jsonl"), str(d / "m.jsonl"), str(d), str(folder),
                         out=d / "out2", n_boot=50, allow_incomplete=True)
            self.assertEqual(res["coverage"]["hpsv2"]["scored"], 179)
            self.assertIn("| H1 | vqascore - clipscore, prompts tagged 'counting' |", md)
            self.assertIn("97.5 %", md)


if __name__ == "__main__":
    unittest.main()
