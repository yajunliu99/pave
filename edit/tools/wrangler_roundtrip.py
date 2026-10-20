"""Apply a pipeline run's project card with Network Wrangler and compare the result with the
pipeline's own after-network.

  uv run --python 3.13 --with network-wrangler==1.0b4 python tools/wrangler_roundtrip.py out/sr347_i10_ddi/mock

Reads   <run>/wrangler/base/{links.json,nodes.geojson}, <run>/wrangler/project_card.yml,
        <run>/after_network/, <run>/run_state.json
Writes  <run>/wrangler/roundtrip.json and <run>/wrangler/after_from_wrangler/ (GMNS tables)
Exit 0 when nodes, links and movements match.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
warnings.filterwarnings("ignore")

from pave_edit.gmns import Network  # noqa: E402
from pave_edit.wrangler import SitePlacement, to_gmns  # noqa: E402

TOL_M = 0.01


def roundtrip(run: Path, card_path: Path | None = None) -> dict:
    """Apply the run's project card to its exported base with network_wrangler; compare with after_network."""
    run = Path(run).resolve()
    from network_wrangler import load_roadway_from_dir
    from network_wrangler.scenario import create_scenario
    from projectcard import read_card
    logging.getLogger("WranglerLogger").setLevel(logging.WARNING)

    state = json.loads((run / "run_state.json").read_text())
    place = SitePlacement.for_scenario(state["scenario"])
    w = run / "wrangler"
    base = load_roadway_from_dir(w / "base", file_format="geojson")
    card = read_card(card_path or (w / "project_card.yml"), validate=True)
    scenario = create_scenario(base_scenario={"road_net": base}, project_card_list=[card])
    scenario.apply_all_projects()
    rn = scenario.road_net
    got = to_gmns(rn.links_df.drop(columns="geometry").to_dict("records"),
                  rn.nodes_df.drop(columns="geometry").to_dict("records"), place, "after_from_wrangler")
    want = Network.load(run / "after_network")
    want.rebuild_movements()
    got.write(w / "after_from_wrangler")

    report = {"project": card.project, "applied_projects": list(scenario.applied_projects),
              "changes_in_card": len(card.changes) if getattr(card, "changes", None) else 1,
              "nodes": _compare(want.nodes, got.nodes, _node_diff),
              "links": _compare(want.links, got.links, _link_diff),
              "movements": _movements(want, got)}
    report["match"] = all(not report[k]["mismatches"] for k in ("nodes", "links", "movements"))
    (w / "roundtrip.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    report = roundtrip(Path(sys.argv[1]))
    print(f"project card: {report['project']} ({report['changes_in_card']} changes) applied by network_wrangler")
    for k in ("nodes", "links", "movements"):
        r = report[k]
        print(f"  {k:<9} pipeline {r['pipeline']:>3}  wrangler {r['wrangler']:>3}  mismatches {len(r['mismatches'])}")
        for m in r["mismatches"][:10]:
            print(f"     {m}")
    for n in report["links"].get("notes", []):
        print(f"  note: {n}")
    print("round trip:", "MATCH" if report["match"] else "DIFFERS")
    return 0 if report["match"] else 1


def _compare(want: dict, got: dict, diff) -> dict:
    mism, notes = [], []
    for i in sorted(want.keys() | got.keys()):
        if i not in got:
            mism.append(f"{i} missing after Network Wrangler")
        elif i not in want:
            mism.append(f"{i} only in the Network Wrangler result")
        else:
            d, n = diff(want[i], got[i])
            mism += [f"{i}: {x}" for x in d]
            notes += [f"{i}: {x}" for x in n]
    return {"pipeline": len(want), "wrangler": len(got), "mismatches": mism, "notes": notes}


def _node_diff(a, b):
    d = []
    if math.dist((a["x_coord"], a["y_coord"]), (b["x_coord"], b["y_coord"])) > TOL_M:
        d.append(f"moved {math.dist((a['x_coord'], a['y_coord']), (b['x_coord'], b['y_coord'])):.3f} m")
    d += [f"{k} {a.get(k)!r} != {b.get(k)!r}" for k in ("node_type", "ctrl_type", "control_group")
          if str(a.get(k, "")) != str(b.get(k, ""))]
    return d, []


def _link_diff(a, b):
    d = [f"{k} {a.get(k)!r} != {b.get(k)!r}" for k in ("from_node_id", "to_node_id", "facility_type",
                                                          "control_group", "control_status", "lanes")
         if str(a.get(k, "")) != str(b.get(k, ""))]
    notes = []
    if len(a["geometry"]) > 2:
        notes.append(f"curved geometry ({len(a['geometry'])} points) becomes a straight A-B line in the card")
    elif abs(a["length"] - b["length"]) * 1000 > TOL_M:
        d.append(f"length {a['length']} km != {b['length']} km")
    return d, notes


def _movements(want, got):
    k = lambda m: (m["node_id"], m["ib_link_id"], m["ob_link_id"], m["ctrl_type"])  # noqa: E731
    a, b = {k(m) for m in want.movements}, {k(m) for m in got.movements}
    return {"pipeline": len(a), "wrangler": len(b),
            "mismatches": [f"only in pipeline {x}" for x in sorted(a - b)] + [f"only in Network Wrangler {x}" for x in sorted(b - a)]}


if __name__ == "__main__":
    sys.exit(main())
