"""git_commit marks a checkout dirty only for changes that affect what a run computes."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from genprobe.utils.env import git_commit


@unittest.skipUnless(shutil.which("git"), "git not installed")
class TestGitCommit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        for rel in ("genprobe/x.py", "results/table.csv", "docs/hpc_check.md"):
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text("v1\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def test_clean(self):
        self.assertNotIn("dirty", git_commit(str(self.repo)))

    def test_job_outputs_do_not_count(self):
        (self.repo / "results/table.csv").write_text("v2\n")
        (self.repo / "docs/hpc_check.md").write_text("v2\n")
        self.assertNotIn("dirty", git_commit(str(self.repo)))

    def test_code_changes_count_from_any_subdirectory(self):
        (self.repo / "genprobe/x.py").write_text("v2\n")
        self.assertTrue(git_commit(str(self.repo)).endswith("-dirty"))
        self.assertTrue(git_commit(str(self.repo / "results")).endswith("-dirty"))

    def test_outside_a_repo(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(git_commit(d), "nogit")


if __name__ == "__main__":
    unittest.main()
