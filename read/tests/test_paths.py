"""Where run folders, results and the dashboard live: the environment first, then configs/paths.json."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pave_read import paths


class PathTests(unittest.TestCase):
    def test_environment_moves_each_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"NET2CELL_RUNS_ROOT": f"{tmp}/runs", "NET2CELL_RESULTS_ROOT": f"{tmp}/results", "NET2CELL_DASHBOARD_ROOT": f"{tmp}/dash",
                   "NET2CELL_DASHBOARD_TEMPLATE": f"{tmp}/page.html"}
            with patch.dict(os.environ, env):
                got = (paths.runs_root(), paths.results_root(), paths.dashboard_root(), paths.dashboard_template())
            self.assertEqual(got, tuple(Path(v).resolve() for v in env.values()))

    def test_the_dashboard_template_and_map_library_exist(self):
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("NET2CELL_DASHBOARD")}, clear=True):
            self.assertTrue(paths.dashboard_template().exists())
            self.assertTrue((paths.dashboard_root() / "vendor" / "leaflet.js").exists())


if __name__ == "__main__":
    unittest.main()
