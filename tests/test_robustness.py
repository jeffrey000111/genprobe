"""Transient-filesystem retries and the generation-coverage check in summaries."""

import errno
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from genprobe.backends import FakeBackend
from genprobe.generate import run_generation
from genprobe.summarize import generation_coverage, summarize
from genprobe.utils.io import atomic_save_image, retry_transient
from genprobe.utils.manifest import latest_by_id, read_many
from tests.helpers import make_cfg, make_prompts, quiet


class TestRetry(unittest.TestCase):
    def test_stale_handle_is_retried(self):
        calls, pauses = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise OSError(errno.ESTALE, "Stale file handle")
            return "done"
        self.assertEqual(retry_transient(flaky, sleep=pauses.append), "done")
        self.assertEqual((len(calls), pauses), (3, [2.0, 4.0]))

    def test_other_errors_and_persistent_failures_raise(self):
        def full():
            raise OSError(errno.ENOSPC, "No space left on device")
        with self.assertRaises(OSError):
            retry_transient(full, sleep=lambda s: None)

        n = []

        def always_stale():
            n.append(1)
            raise OSError(errno.ESTALE, "Stale file handle")
        with self.assertRaises(OSError):
            retry_transient(always_stale, attempts=3, sleep=lambda s: None)
        self.assertEqual(len(n), 3)

    def test_atomic_save_writes_png(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "x.png"
            atomic_save_image(Image.new("RGB", (4, 4)), p)
            self.assertTrue(p.is_file())
            self.assertFalse(p.with_name("x.png.tmp").exists())


class TestCoverage(unittest.TestCase):
    def test_missing_shard_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prompts = make_prompts(4)
            ppath = out / "prompts.jsonl"
            ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
            full = make_cfg(name="full", seeds=[0, 1], batch_size=2)
            run_generation(cfg=full, prompts=prompts, out=out, backend=FakeBackend(full), backend_kind="fake",
                           log=quiet)
            part = make_cfg(name="part", seeds=[0, 1], batch_size=2)
            run_generation(cfg=part, prompts=prompts, out=out, backend=FakeBackend(part), backend_kind="fake",
                           shard=0, num_shards=2, log=quiet)          # shard 1 of 2 never ran
            manifests = str(out / "manifests" / "gen_*.jsonl")
            gen = latest_by_id(read_many(sorted(Path(out, "manifests").glob("gen_*.jsonl"))))
            cov = generation_coverage(gen, 4, ["full", "part"])
            self.assertEqual(cov["full"], {"expected": 8, "ok": 8, "blank": 0, "error": 0, "no_record": 0})
            self.assertEqual((cov["part"]["ok"], cov["part"]["no_record"]), (4, 4))
            res = summarize(prompts_path=str(ppath), manifests=manifests, data=str(out), metrics=[],
                            out=str(out / "res"), full_run_images=2400)
            self.assertEqual(res["generation_coverage"]["part"]["no_record"], 4)
            md = (out / "res" / "summary.md").read_text()
            self.assertIn("**Incomplete:** part is missing 4 images", md)


if __name__ == "__main__":
    unittest.main()
