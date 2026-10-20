"""What Network Wrangler adds: combining project cards on one base network.

  uv run --no-project --python 3.13 --with "network-wrangler==1.0b4" --with matplotlib \
      python tools/wrangler_scenarios.py

Builds the DDI and SPUI project cards with the pipeline (from the blind Mapbox proposal), adds two
example widening cards, then asks Network Wrangler to apply five combinations to the existing
network. Writes out/sr347_i10_ddi/wrangler_scenarios/scenarios.json, which
tools/build_dashboard.py shows on the dashboard.

The widening cards are made up for this demonstration; they are not ADOT projects.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
warnings.filterwarnings("ignore")

import yaml  # noqa: E402

from pave_edit.gmns import Network  # noqa: E402
from pave_edit.pipeline import run  # noqa: E402
from pave_edit.vlm import MockProposer  # noqa: E402
from pave_edit.wrangler import SitePlacement, to_gmns, write_card  # noqa: E402

SCEN = ROOT / "scenarios" / "sr347_i10_ddi"
PROPOSAL = SCEN / "proposals" / "blind_mapbox_nonames.json"
OUT = ROOT / "out" / "sr347_i10_ddi" / "wrangler_scenarios"
DDI, SPUI = "SR 347 / I-10 diverging diamond", "SR 347 / I-10 single point interchange"
WIDEN, WIDEN_CHECKED = "Queen Creek Rd bridge: 3 lanes each way (example)", "Queen Creek Rd bridge: add one lane each way (example)"
BRIDGE = [205, 206]
EXAMPLE_NOTE = "Made-up example for this demonstration, not an ADOT project. It assumes the DDI layout of the bridge."


def build_cards() -> tuple[dict, dict]:
    cards, runs = {}, {}
    for target, name in (("ddi", DDI), ("spui", SPUI)):
        out = OUT / target
        with contextlib.redirect_stdout(io.StringIO()):
            code = run(SCEN, out, MockProposer(PROPOSAL, name="session"), hide_names=True, target=target)
        if code != 0:
            raise SystemExit(f"pipeline failed for {target}")
        card = yaml.safe_load((out / "wrangler" / "project_card.yml").read_text())
        card["project"] = name
        if target == "spui":
            card["dependencies"] = {"conflicts": [DDI]}
        cards[name], runs[name] = card, out
    cards[WIDEN] = {"project": WIDEN, "tags": ["example"], "notes": EXAMPLE_NOTE,
                    "dependencies": {"prerequisites": [DDI]},
                    "roadway_property_change": {"facility": {"links": {"model_link_id": BRIDGE, "ignore_missing": False}},
                                                "property_changes": {"lanes": {"set": 3}, "lanes_known": {"set": True}}}}
    cards[WIDEN_CHECKED] = {"project": WIDEN_CHECKED, "tags": ["example"], "notes": EXAMPLE_NOTE,
                            "dependencies": {"prerequisites": [DDI]},
                            "roadway_property_change": {"facility": {"links": {"model_link_id": BRIDGE, "ignore_missing": False}},
                                                        "property_changes": {"lanes": {"existing": 2, "change": 1,
                                                                                       "existing_value_conflict": "error"}}}}
    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        write_card(card, OUT / "cards" / f"{_slug(name)}.yml")
    return cards, runs


def describe(card: dict) -> dict:
    deps = card.get("dependencies", {})
    if "changes" in card:
        kinds = [next(iter(c)) for c in card["changes"]]
        change = f"{len(kinds)} network changes ({', '.join(sorted(set(k.replace('_', ' ') for k in kinds)))})"
    else:
        pc = card["roadway_property_change"]["property_changes"]["lanes"]
        change = ("set lanes to 3 on links 205 and 206" if "set" in pc else
                  f"add {pc['change']} lane on links 205 and 206, expecting {pc['existing']} lanes today")
    return {"project": card["project"], "change": change, "requires": deps.get("prerequisites", []),
            "conflicts_with": deps.get("conflicts", []), "example": "example" in card.get("tags", [])}


def apply(card_names: list[str], cards: dict, base_dir: Path) -> dict:
    from network_wrangler import load_roadway_from_dir
    from network_wrangler.scenario import create_scenario
    from projectcard import read_card
    logging.getLogger("WranglerLogger").setLevel(logging.CRITICAL)
    res = {"submitted": card_names, "queued": None, "applied": [], "outcome": "applied", "error_type": None, "error": None}
    sc = None
    try:
        base = load_roadway_from_dir(base_dir, file_format="geojson")
        pcs = [read_card(OUT / "cards" / f"{_slug(n)}.yml", validate=True) for n in card_names]
        sc = create_scenario(base_scenario={"road_net": base}, project_card_list=pcs)
        res["queued"] = [cards_by_lower(cards)[p] for p in sc.queued_projects]
        sc.apply_all_projects()
    except Exception as e:  # noqa: BLE001 - the refusal itself is the result we report
        res["outcome"] = "stopped" if sc is not None and sc.applied_projects else "refused"
        res["error_type"], res["error"] = type(e).__name__, str(e).strip().splitlines()[0][:300]
    if sc is not None:
        res["applied"] = [cards_by_lower(cards)[p] for p in sc.applied_projects]
        res["_net"] = sc.road_net
    return res


def cards_by_lower(cards: dict) -> dict:
    return {k.lower(): k for k in cards}


def compare(net, place, want: Network) -> list[str]:
    got = to_gmns(net.links_df.drop(columns="geometry").to_dict("records"),
                  net.nodes_df.drop(columns="geometry").to_dict("records"), place)
    want.rebuild_movements()
    diffs = []
    if set(got.nodes) != set(want.nodes):
        diffs.append(f"nodes differ: {sorted(set(got.nodes) ^ set(want.nodes))}")
    for i in sorted(set(got.links) | set(want.links)):
        a, b = want.links.get(i), got.links.get(i)
        if a is None or b is None:
            diffs.append(f"link {i} only on one side")
            continue
        for k in ("from_node_id", "to_node_id", "facility_type", "control_status", "lanes"):
            if str(a.get(k, "")) != str(b.get(k, "")):
                diffs.append(f"link {i} {k}: {a.get(k)!r} -> {b.get(k)!r}")
    ma = {(m["node_id"], m["ib_link_id"], m["ob_link_id"]) for m in want.movements}
    mb = {(m["node_id"], m["ib_link_id"], m["ob_link_id"]) for m in got.movements}
    if ma != mb:
        diffs.append(f"{len(ma ^ mb)} turning movements differ")
    return diffs, (len(got.nodes), len(got.links), len(got.movements))


def main() -> int:
    cards, runs = build_cards()
    state = json.loads((runs[DDI] / "run_state.json").read_text())
    place = SitePlacement.for_scenario(state["scenario"])
    base_dir = runs[DDI] / "wrangler" / "base"

    singles = []
    for name in (DDI, SPUI):
        r = apply([name], cards, base_dir)
        diffs, size = compare(r.pop("_net"), place, Network.load(runs[name] / "after_network"))
        singles.append({"project": name, "changes": len(cards[name]["changes"]), "nodes": size[0], "links": size[1],
                        "movements": size[2], "identical": not diffs and r["outcome"] == "applied", "differences": diffs})

    plan = [
        ("order", "Applies projects in a safe order",
         "The widening card says it requires the DDI. It was submitted first; Network Wrangler applied the DDI first anyway.",
         [WIDEN, DDI]),
        ("prerequisite", "Refuses a project whose prerequisite is missing",
         "The widening card was submitted without the DDI it depends on.",
         [WIDEN]),
        ("conflict", "Refuses projects declared incompatible",
         "The single point card states that it cannot be combined with the DDI; both were submitted.",
         [DDI, SPUI]),
        ("existing", "Stops when a card's assumption about today's network is wrong",
         "The second widening card adds one lane and expects 2 lanes today. The SR 347 network has no lane counts "
         "(stored as 0 with lanes_known = false), so the expectation fails and nothing is changed on the bridge.",
         [DDI, WIDEN_CHECKED]),
    ]
    scenarios = []
    for key, title, why, names in plan:
        r = apply(names, cards, base_dir)
        net = r.pop("_net", None)
        result = None
        if r["outcome"] == "applied" and net is not None:
            diffs, size = compare(net, place, Network.load(runs[DDI] / "after_network"))
            lanes = net.links_df.set_index("model_link_id").loc[BRIDGE, "lanes"].tolist()
            other = [d for d in diffs if not any(d.startswith(f"link {i} lanes") for i in BRIDGE)]
            result = {"nodes": size[0], "links": size[1], "movements": size[2], "bridge_lanes": lanes,
                      "differences_from_ddi": diffs, "other_differences": other}
        scenarios.append({"id": key, "title": title, "why": why, "cards": [describe(cards[n]) for n in names],
                          **r, "result": result})

    report = {"tool": "network_wrangler 1.0b4, projectcard 0.3.3", "singles": singles, "scenarios": scenarios}
    (OUT / "scenarios.json").write_text(json.dumps(report, indent=2))
    for s in singles:
        print(f"{s['project']:<42} {s['changes']:>2} changes -> {s['nodes']} nodes / {s['links']} links / "
              f"{s['movements']} movements, identical to pipeline: {s['identical']}")
    for s in scenarios:
        print(f"[{s['id']}] {s['outcome'].upper():<8} submitted {s['submitted']}")
        print(f"          queued {s['queued']} applied {s['applied']}")
        if s["error"]:
            print(f"          {s['error_type']}: {s['error']}")
        if s["result"]:
            print(f"          bridge lanes {s['result']['bridge_lanes']}; vs DDI: {s['result']['differences_from_ddi']}")
    print(f"wrote {OUT / 'scenarios.json'}")
    return 0


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


if __name__ == "__main__":
    sys.exit(main())
