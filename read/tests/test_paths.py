"""Where run folders and results live: the environment first, then configs/paths.json."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pave_read import paths


class PathTests(unittest.TestCase):
    def test_environment_moves_each_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PAVE_RUNS_ROOT": f"{tmp}/runs", "PAVE_RESULTS_ROOT": f"{tmp}/results"}
            with patch.dict(os.environ, env):
                got = (paths.runs_root(), paths.results_root())
            self.assertEqual(got, tuple(Path(v).resolve() for v in env.values()))


if __name__ == "__main__":
    unittest.main()
