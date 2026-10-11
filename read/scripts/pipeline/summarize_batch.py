"""Summarise a batch of site runs: one CSV row per site (the sites themselves are viewed in dashboard/intersection_explorer.html).

Reads only run outputs (no model or map requests). Lane strings list each approach's supported motor lanes
from left to right: upper case is the lane use read from markings (L, T, R, combinations such as TR for a
shared lane); lower case is a turn the movement step paired the lane with when no marking was read; ? means
neither. "+rail" marks a lane shared with rail. All are model predictions, not reference lane counts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.lane_check import reference_lanes, reference_name
from movement_fixer.paths import results_root, runs_root

LETTER = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}


def utdf_lanes(path):
    """UTDF approaches on each incoming macronet link: {link: {UTDF name: "L T TR"}} (lane 1 is leftmost)."""
    if not path or not Path(path).exists():
        return {}
    return {link: {name: " ".join(lanes) for name, lanes in named.items()} for link, named in reference_lanes(path).items()}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def lane_string(section, movements):
    lanes = sorted((r for r in section["regions"] if r.get("existence") == "supported" and r.get("motor_lane_index")),
                   key=lambda r: r["motor_lane_index"])
    paired = {}
    for m in movements:
        if m["in_section_id"] == section["section_id"] and m["status"] == "candidate":
            for pair in m["lane_pairs"]:
                paired.setdefault(pair["in_region_id"], set()).add(m["turn"])
    out = []
    for r in lanes:
        use = r.get("lane_use_prediction", {})
        turns = use.get("allowed_turns") or []
        mark = "".join(LETTER[t] for t in ("left", "through", "right", "u_turn") if t in turns)
        mark = mark or "".join(LETTER[t].lower() for t in ("left", "through", "right", "u_turn") if t in paired.get(r["region_id"], ())) or "?"
        out.append(mark + ("+rail" if "rail" in r["surface_type"] else ""))
    bikes = sum(r.get("surface_type") == "bicycle_strip" and r.get("existence") == "supported" for r in section["regions"])
    return " ".join(out) + (f" (+{bikes} bike)" if bikes else "")


def site_row(site, prefix, utdf):
    name = f"{prefix}_{site['node_id']}"
    exp = runs_root()
    status = (read(exp / f"hybrid_pipeline_{name}/status.json") or {}).get("state")
    row = {"utdf_intid": site["utdf_intid"], "utdf_name": site["utdf_name"], "node_id": site["node_id"], "arms": len(site["arms"]),
           "classification": status or "not started"}
    lanes = read(exp / f"hybrid_pipeline_{name}/lanes.json") or []
    movements = read(exp / f"hybrid_pipeline_{name}/movements.json") or []
    row["approach_lanes"] = "; ".join(f"{s['section_id'][3:]}: {lane_string(s, movements)}" for s in lanes if s["section_id"].startswith("in_"))
    config_legs = (read(ROOT / "configs/pilots" / f"site_{name}.json") or {}).get("legs", {})
    utdf_rows, matched, compared = [], 0, 0
    for s in lanes:
        if not s["section_id"].startswith("in_"):
            continue
        named = utdf.get(str(config_legs.get(s["section_id"][3:], {}).get("in_link_id")), {})
        reference = named.get(reference_name(named, s["section_id"][3:]))
        if reference is None:
            continue
        utdf_rows.append(f"{s['section_id'][3:]}: {reference}")
        compared += 1
        matched += s["count"]["model_motor_count"] == len(reference.split())
    row["utdf_lanes"] = "; ".join(utdf_rows)
    row["lane_count_agreement"] = f"{matched}/{compared}" if compared else ""
    row["exit_lanes"] = "; ".join(f"{s['section_id'][4:]}: {s['count']['model_motor_count']}" for s in lanes if s["section_id"].startswith("out_"))
    row["movement_candidates"] = sum(m["status"] == "candidate" for m in movements)
    row["movement_unresolved"] = sum(m["status"] == "unresolved" for m in movements)
    config = read(ROOT / "configs/pilots" / f"site_{name}.json") or {}
    row["untraced_sections"] = " ".join(u["id"] for u in config.get("untraced_sections", []))
    imagery = read(exp / f"imagery_{name}/status.json") or {}
    row["missing_street_view_bands"] = " ".join(f"{m['section_id']}/{m['position']}" for m in imagery.get("missing", []))
    calls = 0
    for path in (f"geometry_loop_{name}/geometry/summary.json", f"hybrid_pipeline_{name}/summary.json"):
        calls += (read(exp / path) or {}).get("vlm_calls_this_run", 0)
    calls += (read(exp / f"downstream_audit_{name}/summary.json") or {}).get("api_calls", 0)
    calls += (read(exp / f"control_generic_{name}/audit/summary.json") or {}).get("api_calls", 0)
    row["model_calls_recorded"] = calls
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--prefix", default="tempe")
    p.add_argument("--csv", type=Path, help="default <results root>/<list>_summary.csv")
    p.add_argument("--utdf-movements", type=Path, default=ROOT.parent / "net2cell_utdf/build/movements/movement_utdf.csv")
    a = p.parse_args()
    batch = json.loads(a.sites.read_text())
    utdf = utdf_lanes(a.utdf_movements)
    rows = [site_row(s, a.prefix, utdf) for s in batch["sites"]]
    out_csv = a.csv or results_root() / f"{a.sites.stem}_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    done = sum(r["classification"] == "complete" for r in rows)
    agree = [tuple(map(int, r["lane_count_agreement"].split("/"))) for r in rows if r["lane_count_agreement"] and r["classification"] == "complete"]
    print({"sites": len(rows), "complete": done, "approaches_agreeing_with_utdf": [sum(x for x, _ in agree), sum(y for _, y in agree)],
           "csv": str(out_csv)})


if __name__ == "__main__":
    main()
