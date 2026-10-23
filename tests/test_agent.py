"""Generation-debugging agent (genprobe/agent/): rules, budget and a full simulated run with fake
images and simulated checker scores (no GPU)."""

import json
import tempfile
import unittest
from pathlib import Path

from genprobe import judge
from genprobe.agent import compare, core, gen, run
from genprobe.agent.core import Attempt, PromptState

COSTS = {"gen": {"sdxl": 3.0, "sd35m": 4.6, "flux_schnell": 1.1}, "score": {"vqascore": 0.5, "detect": 1.0, "ocr": 0.2}}


def state(category, checks, spec=None, prompt="p", budget=None):
    a0 = Attempt(k=0, model="sdxl", gen_prompt=prompt, cfg=None, seed=0, sample_id="sdxl__x__s0", action="base",
                 reason="", status="ok", checks=checks, gen_seconds=3.0, score_seconds=core.score_cost(category, COSTS))
    return PromptState(prompt_id="x", prompt=prompt, category=category, spec=spec, tags=[category],
                       budget=budget if budget is not None else core.budget_for(category, COSTS), attempts=[a0])


class TestCore(unittest.TestCase):
    def test_pass_and_better(self):
        cnt = lambda pred, v=0.95: {"detect": float(pred == 3),  # noqa: E731
                                    "detect_extra": {"count_pred": pred, "count_target": 3}, "vqascore": v}
        self.assertTrue(core.passed("counting", cnt(3)))
        self.assertFalse(core.passed("counting", cnt(3, v=0.5)))            # r2: checker AND VQAScore >= 0.9
        self.assertTrue(core.better(cnt(3), cnt(4), "counting"))
        self.assertTrue(core.better(cnt(4), cnt(6), "counting"))            # nearer miss
        self.assertFalse(core.better(cnt(6), cnt(4), "counting"))
        self.assertFalse(core.better({"vqascore": 0.51}, {"vqascore": 0.50}, "genai"))   # within delta: noise
        self.assertTrue(core.better({"vqascore": 0.95}, {"vqascore": 0.50}, "genai"))
        self.assertTrue(core.passed("text", {"ocr": 0.86, "vqascore": 0.95}))
        self.assertFalse(core.passed("text", {"ocr": 0.84, "vqascore": 0.95}))
        self.assertEqual(core.diagnose("text", {"ocr": 0.9, "vqascore": 0.3}), "low_alignment")

    def test_diagnose(self):
        self.assertEqual(core.diagnose("counting", {"detect_extra": {"count_pred": 0, "count_target": 2}}), "missing_object")
        self.assertEqual(core.diagnose("counting", {"detect_extra": {"count_pred": 5, "count_target": 2}}), "wrong_count")
        self.assertEqual(core.diagnose("spatial", {"detect": 0.0, "detect_extra": {"missing": ["b"]}}), "missing_object")
        self.assertEqual(core.diagnose("spatial", {"detect": 0.0, "detect_extra": {"missing": []}}), "wrong_relation")
        self.assertEqual(core.diagnose("text", {"ocr": 0.1}), "missing_text")
        self.assertEqual(core.diagnose("genai", {"vqascore": 0.2}), "low_alignment")

    def test_rewrites_keep_the_requirement(self):
        self.assertEqual(core.rewrite_rule("counting", "a photo of three cups", {"objects": [{"name": "cup", "count": 3}]}),
                         "a photo of exactly three cups, three in total, each one fully visible and clearly separated")
        self.assertIn("exactly one apple", core.rewrite_rule("counting", "", {"objects": [{"name": "apple", "count": 1}]}))
        r = core.rewrite_rule("spatial", "", {"relation": {"type": "right_of", "a": "dog", "b": "umbrella"}})
        self.assertEqual(r, "a dog on the right side of the image and an umbrella on the left side, both clearly visible")
        t = core.rewrite_rule("text", 'a sign that says "OPEN"', {"text": "OPEN"})
        self.assertTrue(t.startswith('a sign that says "OPEN"') and '"OPEN"' in t[20:])
        self.assertIsNone(core.rewrite_rule("genai", "two cats", None))

    def test_rule_order_and_budget(self):
        miss = {"detect": 0.0, "detect_extra": {"count_pred": 4, "count_target": 3}, "vqascore": 0.4}
        st = state("counting", miss, spec={"objects": [{"name": "cup", "count": 3}]})
        action, params, reason = core.next_rule_action(st, COSTS)
        self.assertEqual((action, params["model"]), ("switch_model", "sd35m"))   # r2: routed model first
        self.assertTrue(reason.startswith("wrong_count"))
        action, params, _ = core.next_rule_action(st, COSTS)
        self.assertEqual(action, "rewrite")
        self.assertIn("exactly three cups", params["gen_prompt"])
        # 5.5 GPU-seconds left: the rule list's reseed (3.0 + 1.5) fits
        st.attempts[0].gen_seconds = st.budget - 1.5 - 5.5
        self.assertEqual(core.next_rule_action(st, COSTS)[0], "reseed")
        # list used up but budget left: more new seeds until the budget runs out
        st.attempts[0].gen_seconds = st.budget - 1.5 - 5.0
        action, params, reason = core.next_rule_action(st, COSTS)
        self.assertEqual(action, "reseed")
        self.assertIn("rule list used up", reason)
        st.attempts.append(Attempt(k=1, model="sdxl", gen_prompt="p", cfg=None, seed=1001, sample_id="s1",
                                   action="reseed", reason="", status="ok", checks=miss, gen_seconds=3.0,
                                   score_seconds=1.5))
        self.assertIsNone(core.next_rule_action(st, COSTS))                 # 0.5 left: nothing fits

    def test_genai_actions(self):
        st = state("genai", {"vqascore": 0.3})
        self.assertEqual(core.next_rule_action(st, COSTS)[:2][0], "switch_model")
        self.assertEqual(core.next_rule_action(st, COSTS)[0], "raise_cfg")   # best is still the SDXL image
        st2 = state("genai", {"vqascore": 0.3})
        st2.policy = "rule_sdxl"                                            # ablation: never switches
        self.assertEqual(core.next_rule_action(st2, COSTS)[0], "raise_cfg")

    def test_rules_are_frozen(self):
        self.assertEqual(core.RULESET, core.FROZEN_RULESET)       # changing the rules needs a new decision

    def test_budget_is_best_of_four(self):
        self.assertAlmostEqual(core.budget_for("counting", COSTS), 4 * (3.0 + 1.5))
        self.assertAlmostEqual(core.budget_for("genai", COSTS), 4 * (3.0 + 0.5))


class TestRun(unittest.TestCase):
    """init -> (plan -> fake generation -> simulated checks -> ingest) x rounds -> report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = Path(self.tmp.name)
        self.root = d / "agent"
        prompts = [
            {"prompt_id": "tc0001", "prompt": "a photo of three cups", "tags": ["counting"], "split": "dev",
             "spec": {"objects": [{"name": "cup", "count": 3}]}},
            {"prompt_id": "ts0001", "prompt": "a dog to the left of a cat", "tags": ["spatial"], "split": "dev",
             "spec": {"relation": {"type": "left_of", "a": "dog", "b": "cat"}}},
            {"prompt_id": "tt0001", "prompt": 'a sign that says "OPEN"', "tags": ["text"], "split": "dev",
             "spec": {"text": "OPEN"}},
            {"prompt_id": "gb0001", "prompt": "two cats on a sofa", "tags": ["basic", "counting"], "split": "dev"},
            {"prompt_id": "gb0002", "prompt": "a red cube", "tags": ["basic", "attribute"], "split": "dev"},
            {"prompt_id": "gb0003", "prompt": "a blue ball", "tags": ["basic"], "split": "test"},
        ]
        (d / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
        (d / "manifests").mkdir()
        (d / "scores").mkdir()
        gen_recs, self.scores = [], {m: [] for m in ("vqascore", "detect", "ocr")}
        t = {"sdxl": 3.0, "sd35m": 4.6, "flux_schnell": 1.1}
        for p in prompts:
            for m in t:
                for s in range(4):
                    sid = f"{m}__{p['prompt_id']}__s{s}"
                    gen_recs.append({"sample_id": sid, "model": m, "prompt_id": p["prompt_id"], "prompt": p["prompt"],
                                     "seed": s, "path": f"/v1/{sid}.png", "status": "ok", "gen_seconds": t[m],
                                     "warmup": False})
                    self.add_checks(sid, p["prompt_id"], s, base=True, model=m)
        for m in t:
            (d / "manifests" / f"gen_{m}_shard000of001.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in gen_recs if r["model"] == m))
        self.flush()

    def tearDown(self):
        self.tmp.cleanup()

    def add_checks(self, sid, pid, seed, base=False, gen_prompt="", model="sdxl"):
        """Simulated checkers: base images all fail except gb0002 and seed 2 of tc0001; a rewritten
        counting prompt passes; flux passes text; anything else fails."""
        ok = (pid == "gb0002") or (base and pid == "tc0001" and seed == 2) or ("exactly" in gen_prompt) \
            or (pid.startswith("tt0001") and model == "flux_schnell")
        rec = lambda v, extra=None: {"sample_id": sid, "metric_version": "v", "value": v, "status": "ok",  # noqa: E731
                                     "created_at": f"2026-10-22T00:00:{len(self.scores['vqascore']):02d}", "seconds": 0.5,
                                     "extra": extra or {}}
        self.scores["vqascore"].append(rec(0.95 if ok else 0.30 + 0.01 * (seed % 4)))
        if pid.startswith("tc"):
            self.scores["detect"].append(rec(float(ok), {"check": "count", "count_pred": 3 if ok else 5, "count_target": 3}))
        if pid.startswith("ts"):
            self.scores["detect"].append(rec(0.0, {"check": "left_of", "missing": []}))
        if pid.startswith("tt"):
            self.scores["ocr"].append(rec(1.0 if ok else 0.5, {"ocr_norm": "OPEN" if ok else "OPFN"}))

    def flush(self):
        for m, recs in self.scores.items():
            (self.d / "scores" / m).mkdir(exist_ok=True)
            (self.d / "scores" / m / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    def test_full_run(self):
        d, root = self.d, self.root
        info = run.init("t1", "dev", prompts_path=str(d / "prompts.jsonl"), data=str(d), root=root)
        self.assertEqual((info["prompts"], info["passed_at_start"]), (5, 1))       # gb0002 passes at once
        self.assertEqual(info["costs"]["gen"], {"sdxl": 3.0, "sd35m": 4.6, "flux_schnell": 1.1})
        for k in range(1, 9):
            n = run.plan("t1", k, root=root)
            if n == 0:
                break
            with self.assertRaises(SystemExit):                                    # no double planning
                run.plan("t1", k, root=root)
            gen.generate_round("t1", k, "fake", root=root, log=lambda *a: None)
            for line in (root / "t1" / f"manifests").glob(f"gen_r{k}_*.jsonl"):
                for r in map(json.loads, line.read_text().splitlines()):
                    self.assertTrue(Path(r["path"]).is_file())
                    self.assertNotEqual(r["prompt"], "") and self.assertTrue(r["prompt"] in
                        {"a photo of three cups", "a dog to the left of a cat", 'a sign that says "OPEN"',
                         "two cats on a sofa"})                                     # checks see the original
                    self.add_checks(r["sample_id"], r["prompt_id"], r["seed"], gen_prompt=r["gen_prompt"],
                                    model=r["model"])
            self.flush()
            run.ingest("t1", k, data=str(d), root=root)
        states = {s.prompt_id: s for s in run.load_states("t1", root)}
        tc = states["tc0001"]
        self.assertEqual([a.action for a in tc.attempts], ["base", "switch_model", "rewrite"])  # rewrite fixed it
        self.assertEqual((tc.best, tc.stop_reason), (2, "passed"))
        tt = states["tt0001"]
        self.assertEqual(tt.attempts[1].action, "switch_model")
        self.assertEqual((tt.attempts[1].model, tt.stop_reason), ("flux_schnell", "passed"))
        ts = states["ts0001"]                                                       # never fixed: runs out
        self.assertEqual(ts.stop_reason, "no affordable action left")
        self.assertLessEqual(ts.spent(), ts.budget + 1e-6)
        self.assertLess(ts.budget - ts.spent(), 4.0)                                # the budget was used up
        self.assertEqual(ts.best, 0)                                                # nothing clearly better
        res = run.report("t1", data=str(d), root=root, results=d / "results")
        final = [json.loads(x) for x in (root / "t1" / "final.jsonl").read_text().splitlines()]
        b4 = {f["prompt_id"]: f["sample_id"] for f in final if f["group"] == "best_of_4"}
        self.assertEqual(b4["tc0001"], "sdxl__tc0001__s2")                           # its passing seed
        self.assertEqual(res["agent"]["over_budget"], 0)
        pr = res["internal_pass_rate"]
        self.assertEqual(pr["agent_rule|all"]["pass_rate"], 3 / 5)
        self.assertEqual(pr["single|all"]["pass_rate"], 1 / 5)
        self.assertEqual(pr["best_of_4|all"]["pass_rate"], 2 / 5)
        rb = {f["prompt_id"]: f for f in final if f["group"] == "routed_best_of_k"}
        self.assertEqual(rb["tc0001"]["model"], "sd35m")
        self.assertEqual(rb["tc0001"]["k"], 2)                                      # 16 // (4.6 + 1.0)
        self.assertEqual(rb["gb0001"]["k"], 4)                                      # flux is cheap
        self.assertTrue(rb["gb0001"]["sample_id"].startswith("flux_schnell__gb0001__s"))
        self.assertTrue(all(f["gpu_seconds"] <= states[f["prompt_id"]].budget + 1e-6 for f in final))
        self.assertEqual(rb["tt0001"]["sample_id"].split("__")[0], "flux_schnell")
        audit = [json.loads(x) for x in (root / "t1" / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(sum(a["step"] == 0 for a in audit), 5)
        self.assertTrue(all("budget_left" in a and "decision_reason" in a for a in audit))
        self.assertTrue((d / "results" / "t1" / "summary.md").is_file())

        # final grading by a (fake) independent judge: passes the agent's final images for tc0001
        # and tt0001, and gb0002's base image
        good = {f["sample_id"] for f in final if f["group"] == "agent_rule" and f["prompt_id"] in ("tc0001", "tt0001")}
        good.add("sdxl__gb0002__s0")

        class FakeJudge:
            version = "fake-judge"

            def generate(self, reqs, log=print):
                return [[judge.Raw(json.dumps({"seen": "x", "issues": [], "score": 5 if r.meta["sample_id"] in good
                                               else 2}), "stop")] for r in reqs]

        meta = judge.grade(str(root / "t1" / "final.jsonl"), "g1", str(d / "judge"), engine=FakeJudge(),
                           log=lambda *a: None)
        self.assertEqual(meta["records"], len({f["sample_id"] for f in final}))      # each image once
        cmp_ = compare.compare("t1", "g1", data=str(d), root=root, results=d / "results", n_boot=200)
        rate = lambda g: cmp_["pass_rate"][g]["all"]["value"]  # noqa: E731
        self.assertAlmostEqual(rate("agent_rule"), 3 / 5)
        self.assertAlmostEqual(rate("single"), 1 / 5)
        self.assertAlmostEqual(cmp_["paired_difference"]["agent_rule - single|all"]["value"], 2 / 5)
        self.assertIn("agent_rule - routed_best_of_k|all", cmp_["paired_difference"])
        self.assertEqual(cmp_["agent_unfixed_by_first_diagnosis"]["agent_rule"],
                         {"wrong_relation": 1, "low_alignment": 1})
        self.assertTrue((d / "results" / "t1" / "judged.md").is_file())

        # ablation run that never switches model, compared together with the full agent
        run.init("t2", "dev", policy="rule_sdxl", prompts_path=str(d / "prompts.jsonl"), data=str(d), root=root)
        for k in range(1, 13):
            if run.plan("t2", k, root=root) == 0:
                break
            gen.generate_round("t2", k, "fake", root=root, log=lambda *a: None)
            for line in (root / "t2" / "manifests").glob(f"gen_r{k}_*.jsonl"):
                for r in map(json.loads, line.read_text().splitlines()):
                    self.assertEqual(r["model"], "sdxl")                           # never switched
                    self.add_checks(r["sample_id"], r["prompt_id"], r["seed"], gen_prompt=r["gen_prompt"],
                                    model=r["model"])
            self.flush()
            run.ingest("t2", k, data=str(d), root=root)
        run.report("t2", data=str(d), root=root, results=d / "results")
        judge.grade(str(root / "t2" / "final.jsonl"), "g2", str(d / "judge"), engine=FakeJudge(), log=lambda *a: None)
        both = compare.compare(["t1", "t2"], ["g1", "g2"], data=str(d), root=root, results=d / "results", n_boot=200)
        self.assertEqual(both["groups"], ["single", "best_of_4", "routed_best_of_k", "agent_rule@t1",
                                          "agent_rule_sdxl@t2"])
        self.assertIn("agent_rule@t1 - agent_rule_sdxl@t2|all", both["paired_difference"])
        self.assertTrue((d / "results" / "t1+t2" / "judged.md").is_file())


if __name__ == "__main__":
    unittest.main()
