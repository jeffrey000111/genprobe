"""The repo is public: tracked text files must not carry account paths (user names live in them)."""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# /home/<account>/..., /fs/atipa/home/<account>/... (HPC3 compute nodes), /Users/<name>/...
ACCOUNT_PATH = re.compile(r"(?:/fs/atipa)?/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/")


class TestNoAccountPaths(unittest.TestCase):
    def test_results_and_docs(self):
        try:
            files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        bad = []
        for f in files:
            if not f.endswith((".md", ".json", ".csv", ".txt", ".yaml", ".jsonl")):
                continue
            text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            for m in ACCOUNT_PATH.finditer(text):
                bad.append(f"{f}: {m.group(0)}")
        self.assertEqual(bad, [], "account paths in files that would be pushed to GitHub")


if __name__ == "__main__":
    unittest.main()
