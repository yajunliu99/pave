"""Where run folders and results live.

Run folders (every intermediate image, cache and log) are large and can be kept out of a synced project folder. The
locations come from the environment (PAVE_RUNS_ROOT, PAVE_RESULTS_ROOT), then configs/paths.json, then the project's
experiments/ and results/ folders.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _setting(env, key, default):
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    config = ROOT / "configs" / "paths.json"
    if config.exists():
        value = json.loads(config.read_text()).get(key)
        if value:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else ROOT / path).resolve()
    return (ROOT / default).resolve() if default else None


def runs_root():
    return _setting("PAVE_RUNS_ROOT", "runs_root", "experiments")


def results_root():
    return _setting("PAVE_RESULTS_ROOT", "results_root", "results")

