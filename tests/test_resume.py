"""Test 2 (guide §4.6): kill a run halfway, restart, and only the remainder is generated."""

import tempfile
import unittest
import warnings
from pathlib import Path

from PIL import Image

from genprobe.backends import FakeBackend
from genprobe.generate import manifest_dir, run_generation
from genprobe.samples import make_sample_id
from genprobe.utils.manifest import done_ids, latest_by_id, read_many
from tests.helpers import KillAfter, SimulatedKill, make_cfg, make_prompts, quiet


def records(out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_many(manifest_dir(out).glob("gen_*.jsonl"))


class TestResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cfg = make_cfg()                  # batch_size=3, seeds [0, 1]
        self.prompts = make_prompts(10)        # -> 20 samples
        self.all_ids = {make_sample_id("toy", p.prompt_id, s) for p in self.prompts for s in (0, 1)}

    def tearDown(self):
        self.tmp.cleanup()

    def run_gen(self, backend, **kw):
        return run_generation(cfg=kw.pop("cfg", self.cfg), prompts=self.prompts, out=self.out,
                              backend=backend, backend_kind="fake", log=quiet, **kw)

    def test_interrupt_then_resume(self):
        first = KillAfter(self.cfg, n=7)
        with self.assertRaises(SimulatedKill):
            self.run_gen(first)
        self.assertEqual(len(first.calls), 6)  # two full batches of 3 got through

        # Simulate the debris a SIGKILL leaves: a half-written image and a half-written line.
        img_dir = self.out / "images" / "toy"
        (img_dir / "p009_s1.png.tmp").write_bytes(b"\x89PNG partial")
        manifest = next(manifest_dir(self.out).glob("gen_toy_*.jsonl"))
        with open(manifest, "a") as f:
            f.write('{"sample_id": "toy__p009__s1", "sta')

        second = FakeBackend(self.cfg)
        with self.assertWarns(UserWarning):  # the broken manifest line is detected and skipped
            counts = self.run_gen(second)

        self.assertEqual(set(first.calls) & set(second.calls), set(), "regenerated a finished sample")
        self.assertEqual(set(first.calls) | set(second.calls), self.all_ids)
        self.assertEqual(counts["skipped"], 6)
        self.assertEqual(counts["ok"], 14)
        self.assertEqual(list(img_dir.glob("*.tmp")), [], "leftover .tmp not cleaned")

        latest = latest_by_id(records(self.out))
        self.assertEqual(set(latest), self.all_ids)
        self.assertTrue(all(r["status"] == "ok" for r in latest.values()))
        for r in latest.values():
            with Image.open(r["path"]) as im:
                im.verify()  # raises on a truncated file

        # Third run: nothing left to do.
        third = FakeBackend(self.cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts = self.run_gen(third)
        self.assertEqual(third.calls, [])
        self.assertEqual(counts["skipped"], 20)

    def test_sharded_runs_cover_everything_once(self):
        backends = [FakeBackend(self.cfg) for _ in range(3)]
        for k, b in enumerate(backends):
            self.run_gen(b, shard=k, num_shards=3)
        calls = [c for b in backends for c in b.calls]
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(set(calls), self.all_ids)
        # Re-running with a different shard count must not redo anything.
        again = FakeBackend(self.cfg)
        self.run_gen(again, shard=0, num_shards=1)
        self.assertEqual(again.calls, [])

    def test_errors_are_retried_blanks_are_not(self):
        bad = make_sample_id("toy", "p003", 0)
        blank = make_sample_id("toy", "p004", 1)
        counts = self.run_gen(FakeBackend(self.cfg, fail_ids={bad}, blank_ids={blank}))
        self.assertEqual((counts["ok"], counts["blank"], counts["error"]), (18, 1, 1))

        retry = FakeBackend(self.cfg)
        counts = self.run_gen(retry)
        self.assertEqual(retry.calls, [bad], "only the errored sample should be retried")
        latest = latest_by_id(records(self.out))
        self.assertEqual(latest[bad]["status"], "ok")
        self.assertEqual(latest[blank]["status"], "blank")

    def test_one_bad_prompt_does_not_sink_its_batch(self):
        bad = make_sample_id("toy", "p000", 0)  # first sample -> first batch
        counts = self.run_gen(FakeBackend(self.cfg, fail_ids={bad}))
        self.assertEqual(counts["error"], 1)
        self.assertEqual(counts["ok"], 19)

    def test_config_change_regenerates(self):
        self.run_gen(FakeBackend(self.cfg))
        changed = make_cfg(steps=2)
        b = FakeBackend(changed)
        self.run_gen(b, cfg=changed)
        self.assertEqual(set(b.calls), self.all_ids)
        recs = records(self.out)
        self.assertEqual(len(done_ids(recs, match={"config_hash": changed.config_hash()})), 20)

    def test_deleted_image_is_regenerated(self):
        self.run_gen(FakeBackend(self.cfg))
        victim = self.out / "images" / "toy" / "p002_s0.png"
        victim.unlink()
        b = FakeBackend(self.cfg)
        self.run_gen(b)
        self.assertEqual(b.calls, [make_sample_id("toy", "p002", 0)])
        self.assertTrue(victim.exists())

    def test_manifest_records_have_traceability_fields(self):
        self.run_gen(FakeBackend(self.cfg))
        r = records(self.out)[0]
        for k in ("sample_id", "model", "model_repo", "prompt_id", "prompt", "seed", "steps", "cfg",
                  "height", "width", "scheduler", "path", "gen_seconds", "peak_mem_gb", "gpu",
                  "git_commit", "config_hash", "run_id", "status", "created_at", "warmup"):
            self.assertIn(k, r)
        warm = [x for x in records(self.out) if x["warmup"]]
        self.assertEqual(len(warm), self.cfg.batch_size, "only the first batch is flagged warmup")
        self.assertEqual(len(list((self.out / "runs").glob("gen_toy_*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
