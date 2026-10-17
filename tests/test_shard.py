"""Test 1 (guide §4.6): same input -> same shards every time; shards are disjoint and complete."""

import random
import unittest

from genprobe.samples import build_samples, make_sample_id, parse_sample_id
from genprobe.utils.shard import chunks, select_shard
from tests.helpers import make_prompts


class TestShard(unittest.TestCase):
    def setUp(self):
        self.samples = build_samples(make_prompts(37), "toy", [0, 1, 2, 3], "out")  # 148, not divisible
        self.key = lambda s: s.sample_id

    def test_deterministic_regardless_of_input_order(self):
        for n in (1, 2, 7, 20):
            for k in range(n):
                ref = [s.sample_id for s in select_shard(self.samples, k, n, self.key)]
                shuffled = self.samples[:]
                random.Random(k * 31 + n).shuffle(shuffled)
                got = [s.sample_id for s in select_shard(shuffled, k, n, self.key)]
                self.assertEqual(ref, got)

    def test_disjoint_and_complete(self):
        for n in (1, 3, 20, 200):  # 200 > number of samples: some shards are empty
            seen = []
            for k in range(n):
                seen.extend(s.sample_id for s in select_shard(self.samples, k, n, self.key))
            self.assertEqual(len(seen), len(set(seen)), "a sample landed in two shards")
            self.assertEqual(set(seen), {s.sample_id for s in self.samples}, "a sample was dropped")

    def test_balanced(self):
        sizes = [len(select_shard(self.samples, k, 20, self.key)) for k in range(20)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            select_shard(self.samples, 5, 5, self.key)
        with self.assertRaises(ValueError):
            select_shard(self.samples, -1, 5, self.key)
        with self.assertRaises(ValueError):
            select_shard(self.samples, 0, 0, self.key)
        with self.assertRaises(ValueError):
            select_shard(self.samples + self.samples[:1], 0, 2, self.key)  # duplicate id

    def test_chunks(self):
        self.assertEqual([len(c) for c in chunks(list(range(7)), 3)], [3, 3, 1])

    def test_sample_id_roundtrip_and_validation(self):
        sid = make_sample_id("sdxl", "gb0042", 1)
        self.assertEqual(sid, "sdxl__gb0042__s1")
        self.assertEqual(parse_sample_id(sid), ("sdxl", "gb0042", 1))
        for bad in [("sd__xl", "p", 0), ("sdxl", "p__1", 0), ("sdxl", "p 1", 0), ("sdxl", "p", -1)]:
            with self.assertRaises(ValueError):
                make_sample_id(*bad)


if __name__ == "__main__":
    unittest.main()
