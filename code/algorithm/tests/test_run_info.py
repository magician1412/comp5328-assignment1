"""run_info must work without git: missing binary or code outside a repository."""

from datetime import datetime
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_info

NO_GIT = {"commit": None, "commit_short": None, "dirty": None}


class GitStateTests(unittest.TestCase):
    def test_missing_git_binary(self):
        with patch.object(run_info.subprocess, "run",
                          side_effect=FileNotFoundError("git")) as run:
            self.assertEqual(run_info.git_state(), NO_GIT)
        run.assert_called()

    def test_not_a_repository(self):
        error = subprocess.CalledProcessError(128, ["git"], stderr="not a git repository")
        with patch.object(run_info.subprocess, "run", side_effect=error):
            self.assertEqual(run_info.git_state(), NO_GIT)

    def test_git_timeout(self):
        error = subprocess.TimeoutExpired(["git"], 30)
        with patch.object(run_info.subprocess, "run", side_effect=error):
            self.assertEqual(run_info.git_state(), NO_GIT)

    def test_run_folder_uses_nogit_label(self):
        with tempfile.TemporaryDirectory() as out:
            started = datetime(2026, 10, 5, 12, 0, 0)
            path = run_info.make_run_dir(out, started, None, "orl")
            self.assertEqual(path.name, "20261005-120000_nogit_orl")
            again = run_info.make_run_dir(out, started, None, "orl")
            self.assertEqual(again.name, "20261005-120000_nogit_orl_2")


if __name__ == "__main__":
    unittest.main()
