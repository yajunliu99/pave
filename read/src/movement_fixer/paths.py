"""Where run folders and exported results live.

Run folders (every intermediate image, cache and log) are large and are kept out of the synced project
folder; compact results are exported back into it. The locations come from the environment
(NET2CELL_RUNS_ROOT, NET2CELL_RESULTS_ROOT), then configs/paths.json, then the project's experiments/
and results/ folders.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _setting(env, key, default):
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    config = ROOT / "configs" / "paths.json"
    if config.exists():
        value = json.loads(config.read_text()).get(key)
        if value:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else ROOT / path).resolve()
    return (ROOT / default).resolve()


def runs_root():
    return _setting("NET2CELL_RUNS_ROOT", "runs_root", "experiments")


def results_root():
    return _setting("NET2CELL_RESULTS_ROOT", "results_root", "results")
