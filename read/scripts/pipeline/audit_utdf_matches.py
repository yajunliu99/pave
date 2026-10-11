"""Check how UTDF intersections were matched to network nodes (net2cell_utdf movement_utdf.csv), and flag doubtful ones.

For each UTDF intersection: its original position (lon0/lat0 in dashboard_data.json), the matched node, how many
other nodes that node connects to, and the nearest node that is a junction (three or more neighbours) and touches
roads named like the UTDF approaches. A match is flagged when the matched node is not a junction although UTDF
gives a named cross street, when it lies far from the UTDF position while a matching junction lies much closer,
or when two UTDF intersections share a node. Decisions taken on the flags are kept in
configs/utdf_node_overrides.json; this script only reports. Reads net2cell_utdf, writes nothing there.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.paths import results_root

SUFFIXES = (" avenue", " road", " street", " boulevard", " drive", " way", " parkway", " lane")


def street(name):
    name = (name or "").lower().strip()
    for s in SUFFIXES:
        name = name.replace(s, "")
    return name.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--utdf-root", type=Path, default=ROOT.parent / "net2cell_utdf")
    p.add_argument("--sites", type=Path, help="restrict to a batch list")
    p.add_argument("--far-m", type=float, default=40.0)
    p.add_argument("--output", type=Path, default=results_root() / "reviews/utdf_node_audit.json")
    a = p.parse_args()
    build = a.utdf_root / "build"
    ints = {i["id"]: i for i in json.loads((build / "dashboard/dashboard_data.json").read_text())["ints"]}
    nodes = {int(r["node_id"]): r for r in csv.DictReader(open(build / "network/macronet/node.csv", newline="", encoding="utf-8-sig"))}
    neighbours, names = defaultdict(set), defaultdict(set)
    for l in csv.DictReader(open(build / "network/macronet/link.csv", newline="", encoding="utf-8-sig")):
        f, t = int(l["from_node_id"]), int(l["to_node_id"])
        neighbours[f].add(t)
        neighbours[t].add(f)
        names[f].add(street(l.get("name")))
        names[t].add(street(l.get("name")))
    matched = {}
    for r in csv.DictReader(open(build / "movements/movement_utdf.csv", newline="", encoding="utf-8-sig")):
        if r.get("utdf_intid") and r.get("node_id"):
            matched[int(float(r["utdf_intid"]))] = int(float(r["node_id"]))
    wanted = {s["utdf_intid"] for s in json.loads(a.sites.read_text())["sites"]} if a.sites else set(matched)
    claims = defaultdict(list)
    for uid, node in matched.items():
        claims[node].append(uid)

    def metres(i, node):
        r = nodes[node]
        return math.hypot((i["lon0"] - float(r["x_coord"])) * 111320 * math.cos(math.radians(i["lat0"])), (i["lat0"] - float(r["y_coord"])) * 110540)

    rows = []
    for uid in sorted(wanted):
        i, node = ints[uid], matched.get(uid)
        if node is None or node not in nodes:
            rows.append({"utdf_intid": uid, "name": i["nm"], "flags": ["unmatched"]})
            continue
        streets = {street(x.get("name")) for x in i["ap"].values() if x.get("name")}
        cross = len(streets) >= 2
        junctions = sorted((metres(i, n), n) for n in nodes if len(neighbours[n]) >= 3
                           and len({s for s in streets if any(s and s in m for m in names[n])}) >= min(2, len(streets)))
        best = junctions[0] if junctions else None
        flags = []
        if len(neighbours[node]) < 3 and cross:
            flags.append("matched_node_is_not_a_junction")
        if len(neighbours[node]) < 3 and not cross:
            flags.append("mid_block_or_single_street_node")
        if metres(i, node) > a.far_m:
            flags.append("far_from_utdf_position")
        if best and best[1] != node and best[0] + 15 < metres(i, node):
            flags.append("closer_matching_junction")
        if len(claims[node]) > 1:
            flags.append("node_shared_with_" + ",".join(str(u) for u in claims[node] if u != uid))
        rows.append({"utdf_intid": uid, "name": i["nm"], "utdf_node_type": i.get("nt"), "control": i.get("ct"),
                     "approaches": {k: x.get("name") for k, x in i["ap"].items()}, "matched_node": node,
                     "matched_node_neighbours": len(neighbours[node]), "distance_m": round(metres(i, node), 1),
                     "nearest_matching_junction": {"node": best[1], "distance_m": round(best[0], 1)} if best else None,
                     "flags": flags})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"rows": rows, "far_m": a.far_m,
                                    "note": "Positions are UTDF lon0/lat0 as converted by net2cell_utdf; flags are prompts for a look, not verdicts."},
                                   indent=1, ensure_ascii=False))
    flagged = [r for r in rows if r["flags"]]
    for r in flagged:
        print(r["utdf_intid"], r["name"], r.get("matched_node"), r["flags"], r.get("nearest_matching_junction"))
    print({"checked": len(rows), "flagged": len(flagged), "output": str(a.output)})


if __name__ == "__main__":
    main()
