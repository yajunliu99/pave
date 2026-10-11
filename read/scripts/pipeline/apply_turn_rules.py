"""Apply the traffic-law movement defaults (hybrid/turn_rules.py) to a finished classification run.

New runs apply them inside the classification stage. This brings an older run up to date without repeating
it: it reads the run's lanes and movements, adds the defaults (applying them twice changes nothing), writes
movements.json and turn_rule_notes.json, and marks the run complete again so later stages rebuild. No model
calls are made.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.hybrid.common import read_json, write_json
from movement_fixer.hybrid.turn_rules import apply_kerb_turn_default


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="classification run folder (hybrid_pipeline_<name>)")
    a = p.parse_args()
    run = a.run.resolve()
    config, lanes, movements = (read_json(run / f) for f in ("inference_config.json", "lanes.json", "movements.json"))
    ruled, notes = apply_kerb_turn_default(movements, lanes, config["sections"], config.get("driving_side", "right"))
    write_json(run / "movements.json", ruled)
    write_json(run / "turn_rule_notes.json", notes)
    status = read_json(run / "status.json") if (run / "status.json").exists() else {}
    status = {k: v for k, v in status.items() if k not in ("error_type", "message")}
    write_json(run / "status.json", {**status, "state": "complete", "turn_rules_applied_utc": datetime.now(timezone.utc).isoformat()})
    (run / "error.json").unlink(missing_ok=True)  # left by a cache-only replay that could not reach the movement step
    print({"run": run.name, "applied": [n["approach"] for n in notes if n["applied"]]})


if __name__ == "__main__":
    main()
