"""VLM judge (genprobe/judge.py, genprobe/judge_eval.py) without a GPU: what is asked, how answers
are read, and the analysis, with a fake engine in place of vLLM."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from genprobe import genai_bench, judge, judge_eval


def make_bench_files(d: Path, n_dev: int = 4, n_test: int = 30):
    folder = d / "gb"
    folder.mkdir()
    rating = {m: k + 1 for k, m in enumerate(genai_bench.MODELS[:5])}
    rating[genai_bench.MODELS[5]] = 3
    n = n_dev + n_test
    data = {f"{i:05d}": {"id": f"{i:05d}", "prompt": f"p {i}", "models": {m: [r, r, r] for m, r in rating.items()}}
            for i in range(n)}
    (folder / "genai_image.json").write_text(json.dumps(data))
    prompts, recs = [], []
    for i in range(n):
        pid = f"gb{i:04d}"
        prompts.append({"prompt_id": pid, "prompt": f'a "quoted" {{brace}} prompt {i}', "source": "genai_bench",
                        "tags": ["basic", "counting"], "split": "dev" if i < n_dev else "test"})
        for m in genai_bench.MODELS:
            recs.append({"sample_id": f"{m}__{pid}__s0", "model": m, "prompt_id": pid, "prompt": f"p {i}", "seed": 0,
                         "path": f"/imgs/{m}/{i}.jpeg", "status": "ok"})
    (d / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return judge.load_bench(str(d / "p.jsonl"), str(d / "m.jsonl"), folder), rating


class FakeEngine:
    """Absolute: score = the model's true rating (so it orders like people); pairwise: always
    "1" (pure position bias); no logprobs except a fixed digit distribution."""
    version = "fake"

    def __init__(self, rating):
        self.rating = rating

    def generate(self, reqs, log=print):
        out = []
        for r in reqs:
            n = judge.N_SAMPLES if r.mode == "sample" else 1
            if r.kind == "abs":
                s = 1 if r.meta["variant"] != "real" else self.rating[r.meta["model"]]
                text = json.dumps({"issues": [], "score": s})
                toks = ['{"', 'issues', '":', ' []', ',', ' "', 'score', '":', ' ', str(s), '}']
                tops = [{t: 0.0} for t in toks]
                tops[9] = {str(s): math.log(0.8), str(min(s + 1, 5)) if s < 5 else "4": math.log(0.2), '"': -5.0}
                raws = [judge.Raw(text, "stop", toks, tops if r.mode == "greedy" else None) for _ in range(n)]
            else:
                raws = [judge.Raw(json.dumps({"reason": "x", "better": "1"}), "stop") for _ in range(n)]
            out.append(raws)
        return out


class TestPieces(unittest.TestCase):
    def test_prompt_text_keeps_quotes_and_braces(self):
        r = judge.Request("k", "abs", "greedy", "gb0000", 'a "red" {cube}', ["x"])
        self.assertIn('Prompt: "a \\"red\\" {cube}"', r.text())
        self.assertIn('{"seen": what the image shows, "issues": [', r.text())
        r = judge.Request("k", "pair", "greedy", "gb0000", "two cats", ["x", "y"])
        self.assertIn('"better": "1", "2" or "same"}', r.text())

    def test_score_digit_distribution(self):
        toks = ['{"', 'issues', '":', ' ["two cats"],', ' "', 'score', '":', ' ', '3', '}']
        tops = [{t: 0.0} for t in toks]
        tops[8] = {"3": math.log(0.6), "4": math.log(0.3), "x": math.log(0.1)}
        d = judge.score_digit_distribution(toks, tops)
        self.assertAlmostEqual(d["3"], 0.6 / 0.9)
        self.assertAlmostEqual(sum(int(k) * v for k, v in d.items()), (3 * 0.6 + 4 * 0.3) / 0.9)
        self.assertIsNone(judge.score_digit_distribution(['{"', 'issues', '": []}'], [{}, {}, {}]))

    def test_rubric_is_frozen(self):
        # changing a prompt, schema or image size after the dev rounds changes this hash; the test
        # round then refuses to run (D-028 revision 3)
        self.assertEqual(judge.template_id(), judge.FROZEN_TEMPLATE)

    def test_parse(self):
        self.assertEqual(judge.parse_abs('{"issues": ["no dog"], "score": 2}'), (2, ["no dog"], True))
        self.assertEqual(judge.parse_abs('{"issues": ["no dog", "cut off'), (None, [], False))
        self.assertEqual(judge.parse_abs('{"issues": [], "score": 4, "x'), (4, [], False))
        self.assertEqual(judge.parse_pair('{"reason": "r", "better": "2"}'), ("second", "r", True))
        self.assertEqual(judge.parse_pair('{"reason": "r", "better": "same"}')[0], "same")
        v, why, ok = judge.parse_pair('{"issues_1": ["no cat"], "issues_2": [], "better": "2"}')
        self.assertEqual((v, ok), ("second", True))
        self.assertEqual(json.loads(why), {"issues_1": ["no cat"], "issues_2": []})
        self.assertEqual(judge.parse_pair('{"reason": "cut'), (None, "", False))


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.bench, self.rating = make_bench_files(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dev_round(self):
        reqs = judge.plan(self.bench, "dev")
        kinds = {}
        for r in reqs:
            k = (r.kind, r.mode, r.meta.get("variant"))
            kinds[k] = kinds.get(k, 0) + 1
        # 4 dev prompts x 6 images; 14 of 15 pairs per prompt differ (models 3 and 6 tie)
        self.assertEqual(kinds[("abs", "greedy", "real")], 24)
        self.assertEqual(kinds[("abs", "greedy", "grey")], 24)
        self.assertEqual(kinds[("abs", "greedy", "mismatch")], 24)
        self.assertEqual(kinds[("abs", "sample", "real")], 24)
        self.assertEqual(kinds[("pair", "greedy", None)], 2 * 4 * 14)
        self.assertEqual(kinds[("pair", "sample", None)], 4 * 14)
        self.assertTrue(all(r.prompt_id < "gb0004" for r in reqs))
        mism = [r for r in reqs if r.meta.get("variant") == "mismatch"]
        for r in mism:
            other = self.bench.images[r.meta["image_from"]]
            self.assertEqual(other["model"], r.meta["model"])
            self.assertNotEqual(other["prompt_id"], r.prompt_id)
        grey = [r for r in reqs if r.meta.get("variant") == "grey"]
        self.assertTrue(all(r.images[0].startswith("grey:/imgs/") for r in grey))
        pair = [r for r in reqs if r.kind == "pair" and r.mode == "greedy" and r.meta["order"] == 2][0]
        self.assertEqual(pair.images, [self.bench.images[pair.meta["first"]]["path"],
                                       self.bench.images[pair.meta["second"]]["path"]])
        self.assertEqual(pair.meta["pair_id"], f"{pair.meta['second']}|{pair.meta['first']}")

    def test_test_round_uses_h3_sample(self):
        h3 = judge.h3_pairs(self.bench, n=100)
        self.assertEqual(h3, judge.h3_pairs(self.bench, n=100))           # deterministic
        self.assertEqual(len(h3), 100)
        self.assertTrue(all(pid >= "gb0004" for pid, *_ in h3))           # no dev prompts
        reqs = judge.plan(self.bench, "test")
        n_pairs = len({r.meta["pair_id"] for r in reqs if r.kind == "pair"})
        self.assertEqual(n_pairs, min(judge.N_H3_PAIRS, 30 * 14))
        self.assertEqual(sum(r.meta.get("variant") == "grey" for r in reqs), min(judge.N_TEST_BLIND, 30 * 6))

    def test_run_and_analysis(self):
        meta = judge.run("dev", "t1", str(self.d / "judge"), engine=FakeEngine(self.rating), bench=self.bench,
                         log=lambda *a: None)
        self.assertEqual(meta["records"], 24 * 3 + 24 * 3 + 2 * 56 + 56 * 3)
        res = judge_eval.run("t1", "dev", data=str(self.d), out_root=str(self.d / "res"), bench=self.bench, n_boot=200)
        self.assertEqual(res["parsing"]["abs"]["json_ok"], 1.0)
        a = res["absolute_vs_people"]
        self.assertEqual(a["judge score (1-5)"]["agreement"], 1.0)            # orders like people
        self.assertEqual(a["judge expected score"]["agreement"], 1.0)
        self.assertNotIn("vqascore (reference)", a)                             # no scores in the fake data dir
        self.assertNotIn("judge_minus_vqascore", res)
        self.assertEqual(res["position_bias"]["value"], 1.0)                    # always "1"
        self.assertEqual(res["position_bias"]["verdict"], "supported")
        self.assertEqual(res["order_agreement"]["value"], 0.0)                  # never the same image twice
        self.assertEqual(res["pairwise_vs_people"]["value"], 0.5)
        self.assertEqual(res["consistency"]["abs"]["samples_all_equal"], 1.0)
        self.assertEqual(res["blind_test"]["grey"]["mean_score_grey"], 1.0)
        self.assertEqual(res["score_distribution"], {"1": 4, "2": 4, "3": 8, "4": 4, "5": 4})
        for f in ("summary.md", "summary.json", "judge_bias.csv"):
            self.assertTrue((self.d / "res" / "t1" / f).is_file())
        self.assertIn("dev round: descriptive only", (self.d / "res" / "t1" / "summary.md").read_text())
        self.assertEqual(res["examples"]["counts"]["absolute"], 0)              # fake judge orders like people
        self.assertEqual(res["examples"]["counts"]["pairwise"], 0)              # it never picks consistently
        self.assertEqual(res["examples"]["counts"]["grey_scored_3_plus"], 0)    # fake grey images score 1


if __name__ == "__main__":
    unittest.main()
