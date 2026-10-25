"""Multi-shot video pipeline (genprobe/story.py, D-038): briefs, the plan parser, the agent's search
over keyframes and videos, what each round makes, the judge's requests, tau_v and the analysis.
No model is loaded."""

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from genprobe import story as S
from genprobe.samples import make_sample_id

REPO = Path(__file__).resolve().parents[1]


def jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def score(sid, v, sec=1.0):
    return {"sample_id": sid, "metric_version": "v", "value": v, "status": "ok", "created_at": "2026-10-24", "seconds": sec}


class TestStoryBasics(unittest.TestCase):
    def test_briefs_file(self):
        rows = [json.loads(x) for x in (REPO / S.BRIEFS).read_text().splitlines() if x.strip()]
        ids = [r["prompt_id"] for r in rows]
        self.assertEqual(len(rows), 32)
        dev = {r["prompt_id"] for r in rows if r["split"] == "dev"}
        self.assertEqual(dev, set(random.Random(0).sample(ids, 8)))     # the split D-038 describes
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            self.assertEqual([b["prompt_id"] for b in S.briefs("smoke")], [sorted(dev)[0]])
            self.assertEqual(len(S.briefs("test")), 24)
        finally:
            os.chdir(cwd)

    def test_parse_plan(self):
        ok = json.dumps({"character": "an orange cat", "shots": [{"description": d} for d in "abc"]})
        self.assertEqual(S.parse_plan(ok)["shots"], ["a", "b", "c"])
        self.assertIsNone(S.parse_plan(json.dumps({"character": "x", "shots": [{"description": "a"}]})))
        self.assertIsNone(S.parse_plan("not json"))

    def test_templates(self):
        shot, story, consist = S._judge_text()
        self.assertIn('"a red kite"', shot.format(prompt='"a red kite"'))
        for t in (story, consist):
            text = t.replace("{n}", "3").format(prompt='"x"')
            self.assertIn("3 shots", text)
            self.assertIn('{"seen"', text)
        self.assertEqual(len(S.story_template_id()), 8)
        self.assertEqual(S.story_template_id(), S.story_template_id())

    def test_states(self):
        k = lambda s: make_sample_id(S.KF_MODEL, "t1s1", s)  # noqa: E731
        self.assertEqual(S.kf_state("t1s1", {k(0): 0.5}), {"chosen": None, "tried": [0], "next": 1})
        self.assertEqual(S.kf_state("t1s1", {k(0): 0.5, k(1): 0.95})["chosen"], 1)
        self.assertEqual(S.kf_state("t1s1", {k(0): 0.5, k(1): 0.7, k(2): 0.6})["chosen"], 1)   # none passes: highest
        v = lambda w, kk=0: make_sample_id(S.i2v(kk), "t1s1", w)  # noqa: E731
        st = S.c_video_state("t1s1", 0, {v(0): 0.4, v(1): 0.8}, 0.7)
        self.assertEqual((st["final"], st["next"]), (v(1), None))
        st = S.c_video_state("t1s1", 0, {v(0): 0.4}, 0.7)
        self.assertEqual((st["final"], st["next"]), (None, v(1)))
        st = S.c_video_state("t1s1", 0, {v(0): 0.4, v(1): 0.5, v(2): 0.45}, 0.7)
        self.assertEqual(st["final"], v(1))
        d = S.arm_state("D", "t1s1", {}, {v(0): 0.5, v(1): 0.5}, None)
        self.assertEqual(d["final"], v(0))                                                        # ties: seed 0


class TestStoryPipeline(unittest.TestCase):
    """A fake run: 2 test briefs (3 and 4 shots), 1 dev brief."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        jl(S.BRIEFS, [{"prompt_id": "d1", "prompt": "dev brief", "split": "dev"},
                      {"prompt_id": "t1", "prompt": "test brief one", "split": "test"},
                      {"prompt_id": "t2", "prompt": "test brief two", "split": "test"}])
        shots = []
        for b, n in (("d1", 3), ("t1", 3), ("t2", 4)):
            shots += [{"prompt_id": f"{b}s{i}", "prompt": f"shot {i} of {b}", "brief_id": b, "shot": i,
                       "split": "dev" if b == "d1" else "test", "character": f"hero of {b}"} for i in range(1, n + 1)]
        jl(S.SHOTS, shots)
        self.shots = [r["prompt_id"] for r in shots]
        self.root = Path("data/story")

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def kf(self, values):        # {(shot, seed): vqascore}
        jl(self.root / "kf/manifests/gen_fluxkf_shard000of001.jsonl",
           [{"sample_id": make_sample_id(S.KF_MODEL, s, k), "model": S.KF_MODEL, "prompt_id": s, "seed": k,
             "path": f"kf/{s}_{k}.png", "status": "ok", "gen_seconds": 1.0} for (s, k) in values])
        jl(self.root / "kf/scores/vqascore/x.jsonl",
           [score(make_sample_id(S.KF_MODEL, s, k), v, 0.5) for (s, k), v in values.items()])

    def vids(self, values, made_only=()):   # {sample_id: vqascore on video}; made_only: made, not scored
        rows = []
        for sid in list(values) + list(made_only):
            model, shot, w = sid.split("__")
            rows.append({"sample_id": sid, "model": model, "prompt_id": shot, "seed": int(w[1:]), "path": f"v/{sid}.mp4",
                         "frames_png": [f"f/{sid}/{i}.png" for i in range(3)], "status": "ok", "gen_seconds": 60.0})
        jl(self.root / "video/manifests/gen_x.jsonl", rows)
        jl(self.root / "video/scores/vqascore_video/x.jsonl", [score(s, v, 2.0) for s, v in values.items()])

    def test_rounds(self):
        test_shots = [s for s in self.shots if not s.startswith("d")]
        kfv = {(s, 0): 0.95 for s in test_shots}
        kfv[("t1s2", 0)] = 0.5
        self.kf(kfv)
        with self.assertRaises(SystemExit):          # t1s2 still needs keyframe seed 1
            S.video_todo("test", 0)
        kfv[("t1s2", 1)] = 0.92
        self.kf(kfv)
        (self.root / "tau_v.json").write_text(json.dumps({"tau_v": 0.7}))
        r0 = {t["sample_id"] for t in S.video_todo("test", 0)}
        self.assertIn(make_sample_id(S.i2v(1), "t1s2", 0), r0)          # the agent's keyframe 1
        self.assertIn(make_sample_id(S.i2v(0), "t1s2", 1), r0)          # D's second video
        self.assertEqual(len(r0), 7 * 3 + 1)                            # A, B (= C), D's second; + C's k1
        v = {s: 0.9 for s in r0}
        v[make_sample_id(S.i2v(0), "t1s1", 0)] = 0.3                    # the agent's first try fails...
        v[make_sample_id(S.i2v(0), "t1s1", 1)] = 0.4                    # ... and D's second video is its second
        v[make_sample_id(S.i2v(1), "t1s2", 0)] = 0.2
        self.vids(v)
        r1 = [t["sample_id"] for t in S.video_todo("test", 1)]
        self.assertEqual(r1, sorted([make_sample_id(S.i2v(0), "t1s1", 2), make_sample_id(S.i2v(1), "t1s2", 1)]))
        self.vids(v, made_only=r1)
        with self.assertRaises(SystemExit):          # made but not scored
            S.video_todo("test", 2)

    def full_test_run(self):
        test_shots = [s for s in self.shots if not s.startswith("d")]
        self.kf({(s, 0): 0.95 for s in test_shots})
        (self.root / "tau_v.json").write_text(json.dumps({"tau_v": 0.7}))
        v = {}
        for s in test_shots:
            v[make_sample_id(S.T2V, s, 0)] = 0.5
            v[make_sample_id(S.i2v(0), s, 0)] = 0.8
            v[make_sample_id(S.i2v(0), s, 1)] = 0.6
        v[make_sample_id(S.i2v(0), "t2s4", 0)] = 0.3                    # the agent redoes this one
        v[make_sample_id(S.i2v(0), "t2s4", 2)] = 0.75
        self.vids(v)
        return test_shots

    def grade_all(self, name, shot_score):
        reqs = S.judge_requests(name)
        recs = []
        for r in reqs:
            sc = shot_score(r.meta["sample_id"]) if r.meta["variant"] == "shot" else 4
            recs.append({"key": r.key, "score": sc, "expected_score": float(sc), "variant": r.meta["variant"]})
        jl(S.JUDGE_DIR / f"story_{name}_r1.jsonl", recs)
        return reqs

    def test_judge_and_analyze(self):
        test_shots = self.full_test_run()
        # A always fails, B and D pass except on t2s4, the agent's third try passes there
        def shot_score(sid):
            model, shot, w = sid.split("__")
            if model == S.T2V:
                return 3
            return 2 if (shot == "t2s4" and w == "s0") else 5
        reqs = self.grade_all("test", shot_score)
        kinds = [r.meta["variant"] for r in reqs]
        # shots: A 7, B 7; C and D share B's except on t2s4 (C: seed 2, D: seed 1); stories dedupe the same way
        self.assertEqual(kinds.count("shot"), 7 + 7 + 2)
        self.assertTrue(all(len(r.images) == (3 if r.meta["variant"] == "shot" else len(r.meta["shots"])) for r in reqs))
        res = S.analyze("test", n_boot=200)
        self.assertAlmostEqual(res["by_arm"]["A"]["shot_pass"]["value"], 0.0)
        self.assertAlmostEqual(res["by_arm"]["B"]["shot_pass"]["value"], 6 / 7)
        self.assertAlmostEqual(res["by_arm"]["C"]["shot_pass"]["value"], 1.0)
        self.assertAlmostEqual(res["tests"]["H7"]["value"], 6 / 7)
        self.assertAlmostEqual(res["tests"]["H8"]["value"], 1 / 7)
        self.assertEqual(res["tests"]["H8"]["level"], S.LEVEL)
        # costs per brief: t1 has 3 shots, t2 has 4
        a_cost = res["by_arm"]["A"]["gpu_seconds_per_brief"]["value"]
        self.assertAlmostEqual(a_cost, (3 * 60 + 4 * 60) / 2)
        b_cost = res["by_arm"]["B"]["gpu_seconds_per_brief"]["value"]
        self.assertAlmostEqual(b_cost, (3 * 61 + 4 * 61) / 2)                       # keyframe 1 s + video 60 s
        c_cost = res["by_arm"]["C"]["gpu_seconds_per_brief"]["value"]
        # C: every shot keyframe 1 + check 0.5 + video 60 + check 2; t2s4 two more videos with checks
        self.assertAlmostEqual(c_cost, (7 * 63.5 + 2 * 62) / 2)
        self.assertAlmostEqual(res["agent"]["video_redone"], 1 / 7)
        self.assertTrue(Path("results/story/summary.md").exists())
        self.assertEqual(len(Path("results/story/shots.jsonl").read_text().splitlines()), 7)

    def test_tune(self):
        dev = [s for s in self.shots if s.startswith("d")]
        self.kf({(s, 0): 0.95 for s in dev})
        vals = dict(zip(dev, (0.55, 0.65, 0.85)))
        self.vids({make_sample_id(S.i2v(0), s, 0): x for s, x in vals.items()})
        self.grade_all("dev", lambda sid: 5 if sid.split("__")[1] == "d1s3" else 3)
        res = S.tune(results=Path("results/story"))
        self.assertEqual(res["tau_v"], 0.7)            # 0.7 and 0.8 both separate perfectly: the lower wins
        self.assertEqual(json.loads((self.root / "tau_v.json").read_text())["tau_v"], 0.7)


if __name__ == "__main__":
    unittest.main()
