"""Check chain on an edited network. Every failure names the rule and the objects involved.

Common checks C1-C6 run for every template; the template's spec adds the movement rules (C3)
and a family of template-specific checks (DDI-*, SPUI-*, RBT-*).
"""
from __future__ import annotations

import math
from collections import deque
from itertools import combinations

from .gmns import Network
from .proposal import Terminal

LEG_RADIUS_M = 150.0   # boundary nodes closer than this belong to one leg (U-turn pairs are skipped)


def _result(name, passed, detail, fact=None):
    return {"check": name, "passed": bool(passed), "detail": detail, "fact": fact}


# ---------- graph helpers ----------

def shortest_path(net: Network, s: int, t: int, allowed_links=None):
    """BFS on directed links; returns (link_ids, node_ids) or None."""
    q, seen = deque([(s, [], [s])]), {s}
    while q:
        n, ls, ns = q.popleft()
        if n == t:
            return ls, ns
        for l in net.links.values():
            if l["from_node_id"] != n or (allowed_links is not None and l["link_id"] not in allowed_links):
                continue
            if l["to_node_id"] not in seen:
                seen.add(l["to_node_id"])
                q.append((l["to_node_id"], ls + [l["link_id"]], ns + [l["to_node_id"]]))
    return None


def external_od_pairs(net: Network):
    bnd = [n for n in net.nodes.values() if n["node_type"] == "boundary"]
    origins = [n["node_id"] for n in bnd if net.out_links(n["node_id"])]
    dests = [n["node_id"] for n in bnd if net.in_links(n["node_id"])]
    return [(o, d) for o in origins for d in dests if o != d and math.dist(net.xy(o), net.xy(d)) > LEG_RADIUS_M]


def route_table(net: Network):
    rows = []
    for o, d in external_od_pairs(net):
        p = shortest_path(net, o, d)
        rows.append(dict(origin_node=o, destination_node=d,
                         link_sequence=";".join(map(str, p[0])) if p else "",
                         node_sequence=";".join(map(str, p[1])) if p else "",
                         status="reachable" if p else "UNREACHABLE"))
    return rows


def group_links(net: Network, group: str) -> set[int]:
    return {l["link_id"] for l in net.links.values() if l.get("control_group") == group}


def proper_crossing(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def links_cross(a: dict, b: dict) -> bool:
    return any(proper_crossing(p, q, s, u) for p, q in zip(a["geometry"], a["geometry"][1:])
               for s, u in zip(b["geometry"], b["geometry"][1:]))


# ---------- the chain ----------

def run_checks(before: Network, after: Network, terminals: list[Terminal], spec: dict, expected_facts=()) -> list[dict]:
    r: list[dict] = []
    groups = set(spec["groups"])

    # C1 table integrity
    bad = []
    for l in after.links.values():
        missing = [end for end in ("from_node_id", "to_node_id") if l[end] not in after.nodes]
        bad += [f"link {l['link_id']} {end} {l[end]} missing" for end in missing]
        if l["length"] <= 0:
            bad.append(f"link {l['link_id']} has length {l['length']}")
        if not missing and (tuple(l["geometry"][0]) != after.xy(l["from_node_id"])
                            or tuple(l["geometry"][-1]) != after.xy(l["to_node_id"])):
            bad.append(f"link {l['link_id']} geometry ends are not on its nodes")
    for m in after.movements:
        if after.links[m["ib_link_id"]]["to_node_id"] != m["node_id"] or after.links[m["ob_link_id"]]["from_node_id"] != m["node_id"]:
            bad.append(f"movement {m['mvmt_id']} is not adjacent at node {m['node_id']}")
    r.append(_result("C1 table integrity", not bad, bad or "IDs are unique; every link starts and ends at an existing node and its drawn line ends on those nodes; every turning movement joins two links that meet"))

    # C2 external OD reachability, compared with the base
    pairs_b = {p for p in external_od_pairs(before) if shortest_path(before, *p)}
    pairs_a = {p for p in external_od_pairs(after) if shortest_path(after, *p)}
    lost, gained = sorted(pairs_b - pairs_a), sorted(pairs_a - pairs_b)
    r.append(_result("C2 external OD reachability", not lost,
                     f"{len(pairs_a)} of {len(external_od_pairs(after))} trips between the approach roads are possible"
                     + (f"; LOST {lost}" if lost else "") + (f"; new {gained}" if gained else ""),
                     "all_external_od_preserved"))

    # C3 movement set of the template: required movements present, forbidden ones absent
    msgs, n_req, n_forb = [], 0, 0
    for g, gs in spec["groups"].items():
        allowed = group_links(after, g)
        for a, b, kind in gs["required"]:
            n_req += 1
            if not shortest_path(after, a, b, allowed):
                msgs.append(f"{g}: missing {kind} {a}->{b}")
        for a, b, kind in gs["forbidden"]:
            n_forb += 1
            if shortest_path(after, a, b, allowed):
                msgs.append(f"{g}: forbidden {kind} {a}->{b} present")
    r.append(_result("C3 movement set", not msgs,
                     msgs or f"All {n_req} required turning movements are present and all {n_forb} prohibited ones are absent",
                     spec.get("movement_fact")))

    # C4 grade separation: arterial never touches freeway nodes; same-level crossings need a node
    msgs = []
    fw_nodes = {e for l in after.links.values() if l["facility_type"] == "freeway" for e in (l["from_node_id"], l["to_node_id"])}
    for l in after.links.values():
        if l["facility_type"] in ("arterial", "intersection_connector") and {l["from_node_id"], l["to_node_id"]} & fw_nodes:
            msgs.append(f"{l['facility_type']} link {l['link_id']} touches a freeway node")
    for a, b in combinations(after.links.values(), 2):
        if {a["from_node_id"], a["to_node_id"]} & {b["from_node_id"], b["to_node_id"]} or a["layer"] != b["layer"]:
            continue
        if not links_cross(a, b):
            continue
        same_group = (a["facility_type"] == b["facility_type"] == "intersection_connector"
                      and a.get("control_group") and a.get("control_group") == b.get("control_group"))
        if not same_group:
            msgs.append(f"links {a['link_id']} and {b['link_id']} cross at the same level without a shared node")
    r.append(_result("C4 grade separation", not msgs, msgs or "The arterial never shares a node with the freeway; links cross at the same level only inside one intersection"))

    # C5 freeway mainline untouched
    msgs = [f"freeway link {i} changed" for i, l in before.links.items()
            if l["facility_type"] == "freeway" and (i not in after.links or _link_sig(after.links[i]) != _link_sig(l))]
    r.append(_result("C5 freeway unchanged", not msgs, msgs or "Every freeway link is identical to the existing network", "freeway_links_unchanged"))

    # C6 no collateral change outside the bound terminals, their host links and the links between them
    ports = {n for t in terminals for n in t.port_nodes()}
    old_groups = {t.group for t in terminals}
    allowed_links = set(spec.get("host_links", [])) | {
        l["link_id"] for l in before.links.values()
        if l.get("control_group") in old_groups | groups or {l["from_node_id"], l["to_node_id"]} <= ports}
    msgs = []
    for i, n in before.nodes.items():
        if i in ports:
            continue
        if i not in after.nodes:
            msgs.append(f"node {i} deleted outside the edit")
        elif _node_sig(n) != _node_sig(after.nodes[i]):
            msgs.append(f"node {i} changed outside the edit")
    for i, l in before.links.items():
        if i in allowed_links:
            continue
        if i not in after.links:
            msgs.append(f"link {i} deleted outside the edit")
        elif _link_sig(l) != _link_sig(after.links[i]):
            msgs.append(f"link {i} changed outside the edit")
    for i, l in after.links.items():
        if i not in before.links and l.get("control_group") not in groups:
            msgs.append(f"new link {i} outside the edited control groups")
    for i, n in after.nodes.items():
        if i not in before.nodes and n.get("control_group") not in groups:
            msgs.append(f"new node {i} outside the edited control groups")
    r.append(_result("C6 no collateral change", not msgs, msgs or "Only the ramp terminals, the links leading into them and the links between them changed",
                     "no_collateral_changes"))

    for fam in spec.get("families", []):
        r += FAMILIES[fam](before, after, terminals, spec)
    for c in r:
        c["asserted_by_proposal"] = c["fact"] in set(expected_facts) if c["fact"] else None
    return r


# ---------- template families ----------

def _ddi_checks(before, after, terminals, spec):
    r = []
    msgs = []
    for t in terminals:
        allowed = group_links(after, t.group)
        for a, b in [("approach_in", "bridge_out"), ("bridge_in", "departure_out")]:
            p = shortest_path(after, t.ports[a], t.ports[b], allowed)
            xs = [n for n in (p[1] if p else []) if after.nodes[n]["node_type"] == "crossover"]
            if len(xs) != 1:
                msgs.append(f"{t.name}: through {t.ports[a]}->{t.ports[b]} passes {len(xs)} crossover nodes")
    r.append(_result("DDI-1 crossover traversal", not msgs, msgs or "Each arterial through movement passes exactly one crossover at each ramp terminal",
                     "through_movements_cross_over_at_each_terminal"))
    msgs = _left_side_msgs(after, terminals)
    r.append(_result("DDI-2 left-side running", not msgs, msgs or "Between the crossovers, each direction has opposing traffic on its right (left-side driving)",
                     "left_side_running_between_terminals"))
    msgs = []
    xs = [n for n in after.nodes.values() if n["node_type"] == "crossover"]
    for n in xs:
        k_in, k_out = len(after.in_links(n["node_id"])), len(after.out_links(n["node_id"]))
        if (k_in, k_out) != (1, 1):
            msgs.append(f"crossover {n['node_id']} has {k_in} in / {k_out} out links (would allow turning between directions)")
    for a, b in combinations(xs, 2):
        if after.find_link(a["node_id"], b["node_id"]) or after.find_link(b["node_id"], a["node_id"]):
            msgs.append(f"crossover nodes {a['node_id']} and {b['node_id']} are joined by a link")
    pairs = sorted({tuple(sorted((a["node_id"], b["node_id"]))) for a, b in combinations(xs, 2)
                    if (a["x_coord"], a["y_coord"]) == (b["x_coord"], b["y_coord"])})
    r.append(_result("DDI-3 crossover nodes not merged", not msgs and xs,
                     msgs or f"{len(xs)} crossover nodes; each pair at the same spot {pairs} stays separate, so no turn is possible between directions"))
    return r


def _left_side_msgs(net, terminals):
    """Opposing bridge link must lie on each bridge link's right (left-side running)."""
    if len(terminals) != 2:
        return ["needs two terminals"]
    a = terminals[0]
    if a.links["bridge_out"] not in net.links or a.links["bridge_in"] not in net.links:
        return ["bridge links are gone"]
    l1, l2 = net.links[a.links["bridge_out"]], net.links[a.links["bridge_in"]]
    msgs = []
    for l, other in [(l1, l2), (l2, l1)]:
        (x1, y1), (x2, y2) = l["geometry"][0], l["geometry"][-1]
        mid = [(p + q) / 2 for p, q in zip(other["geometry"][0], other["geometry"][-1])]
        if (x2 - x1) * (mid[1] - y1) - (y2 - y1) * (mid[0] - x1) >= 0:
            msgs.append(f"opposing link {other['link_id']} is on the left of {l['link_id']} (right-hand running)")
    return msgs


def _spui_checks(before, after, terminals, spec):
    (g, gs), = spec["groups"].items()
    allowed = group_links(after, g)
    ports = {n for t in terminals for n in t.port_nodes() if n in after.nodes}
    other = sorted({l.get("control_group") for l in after.links.values()
                    if l["facility_type"] == "intersection_connector"
                    and {l["from_node_id"], l["to_node_id"]} & ports and l.get("control_group") != g})
    r = [_result("SPUI-1 single signal group", not other,
                 [f"connectors in other groups {other}"] if other else "One signal controls every ramp and arterial movement",
                 "single_signal_controls_all_ramp_movements")]
    long_paths = [f"{a}->{b} uses {len(p[0])} links" for a, b, _ in gs["required"]
                  for p in [shortest_path(after, a, b, allowed)] if p and len(p[0]) != 1]
    r.append(_result("SPUI-2 one conflict point per movement", not long_paths,
                     long_paths or f"Each of the {len(gs['required'])} movements crosses the single intersection once"))
    return r


def _roundabout_checks(before, after, terminals, spec):
    r, ccw, ctl, oneway = [], [], [], []
    for t in terminals:
        ring = [l for l in after.links.values() if l.get("control_group") == t.group and l.get("control_status") == "circulating"]
        pts = [p for l in ring for p in l["geometry"][:-1]]
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        turn = sum((math.atan2(l["geometry"][-1][1] - cy, l["geometry"][-1][0] - cx)
                    - math.atan2(l["geometry"][0][1] - cy, l["geometry"][0][0] - cx)) % (2 * math.pi) for l in ring)
        if abs(turn - 2 * math.pi) > 1e-6:
            ccw.append(f"{t.name}: ring turns {math.degrees(turn):.0f} deg counter-clockwise, expected 360")
        ring_nodes = {l["from_node_id"] for l in ring}
        for n in ring_nodes:
            ins = [l for l in ring if l["to_node_id"] == n]
            outs = [l for l in ring if l["from_node_id"] == n]
            if len(ins) != 1 or len(outs) != 1:
                oneway.append(f"{t.name}: ring node {n} has {len(ins)} in / {len(outs)} out circulating links")
        for l in after.links.values():
            if l.get("control_group") == t.group and str(l.get("control_status", "")).startswith("signal"):
                ctl.append(f"{t.name}: link {l['link_id']} is signal-controlled")
        entries = [l for l in after.links.values() if l.get("control_group") == t.group and l.get("control_status") == "yield_entry"]
        for l in entries:
            if after.nodes[l["to_node_id"]].get("ctrl_type") != "yield":
                ctl.append(f"{t.name}: entry {l['link_id']} does not end at a yield node")
    r.append(_result("RBT-1 counter-clockwise circulation", not ccw, ccw or "every ring circulates counter-clockwise once (right-hand traffic)",
                     "counterclockwise_circulation"))
    r.append(_result("RBT-2 one-way ring", not oneway, oneway or "each ring node has one circulating link in and one out"))
    r.append(_result("RBT-3 yield-controlled entries", not ctl, ctl or "entries yield to circulating traffic; no signals on the rings",
                     "yield_controlled_entries"))
    return r


FAMILIES = {"ddi": _ddi_checks, "spui": _spui_checks, "roundabout": _roundabout_checks}


def _node_sig(n):
    return (n["x_coord"], n["y_coord"], n["node_type"], n.get("control_group", ""))


def _link_sig(l):
    return (l["from_node_id"], l["to_node_id"], l["facility_type"], tuple(map(tuple, l["geometry"])), l.get("lanes", ""))
