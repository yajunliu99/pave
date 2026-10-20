"""Where run folders, exported results and the dashboard live.

Run folders (every intermediate image, cache and log) are large and are kept out of the synced project
folder; compact results are exported back into it. The dashboard folder holds the dashboard's map library and the
built page, and by default its template. The locations come from the environment (NET2CELL_RUNS_ROOT,
NET2CELL_RESULTS_ROOT, NET2CELL_DASHBOARD_ROOT, NET2CELL_DASHBOARD_TEMPLATE), then configs/paths.json, then the
project's experiments/, results/ and dashboard/ folders.
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
    return (ROOT / default).resolve() if default else None


def runs_root():
    return _setting("NET2CELL_RUNS_ROOT", "runs_root", "experiments")


def results_root():
    return _setting("NET2CELL_RESULTS_ROOT", "results_root", "results")


def dashboard_root():
    return _setting("NET2CELL_DASHBOARD_ROOT", "dashboard_root", "dashboard")


def dashboard_template():
    """The page the dashboard is built from: intersection_explorer_template.html in the dashboard folder unless set."""
    return _setting("NET2CELL_DASHBOARD_TEMPLATE", "dashboard_template", None) or dashboard_root() / "intersection_explorer_template.html"
