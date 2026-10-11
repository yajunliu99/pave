"""Closed catalogs, the proposer prompt, and the gates a proposal must pass before any edit.

The VLM is a proposer over objects that already have IDs. It names a facility class, binds
template roles to existing link IDs, and lists operations from a closed catalog. Everything
it names is checked against the base tables here; nothing it says is applied unchecked.
"""
from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass, field

from .gmns import Network

FACILITY_CLASSES = ["diamond", "tight_diamond", "ddi", "spui", "roundabout_diamond", "parclo_a", "parclo_b",
                    "parclo_ab", "cloverleaf", "directional", "trumpet", "other_or_unclear"]

# op name -> implemented yet?
OPERATIONS = {"replace_facility": True, "add_lane": False, "add_ramp": False,
              "add_flyover": False, "realign_ramp": False, "add_managed_lanes": False,
              "meter_ramps": False, "set_allowed_uses": False}
IMPLEMENTED_CONVERSIONS = {("diamond", "ddi"), ("diamond", "spui"), ("diamond", "roundabout_diamond")}

# role -> (facility_type, port side, definition shown to the model)
TERMINAL_ROLES = {
    "approach_in": ("arterial", "in", "arterial link arriving at this terminal from its outer leg (the side away from the freeway)"),
    "departure_out": ("arterial", "out", "arterial link leaving this terminal toward its outer leg"),
    "bridge_in": ("arterial", "in", "arterial link arriving at this terminal from the other terminal, across the freeway"),
    "bridge_out": ("arterial", "out", "arterial link leaving this terminal toward the other terminal, across the freeway"),
    "offramp_in": ("ramp", "in", "off-ramp link that ends at this terminal"),
    "onramp_out": ("ramp", "out", "on-ramp link that starts at this terminal"),
}
# Connectors a diamond terminal must have (in-role, out-role, movement), and ones it must not.
DIAMOND_CONNECTORS = [("approach_in", "bridge_out", "through"), ("bridge_in", "departure_out", "through"),
                      ("approach_in", "onramp_out", "right"), ("bridge_in", "onramp_out", "left"),
                      ("offramp_in", "departure_out", "right"), ("offramp_in", "bridge_out", "left")]
FORBIDDEN_PAIRS = [("approach_in", "departure_out"), ("bridge_in", "bridge_out"), ("offramp_in", "onramp_out")]

EXPECTED_FACTS = {
    "all_external_od_preserved": "every external origin-destination pair reachable before the edit stays reachable",
    "freeway_links_unchanged": "no freeway mainline link changes",
    "six_terminal_movements_per_terminal": "each terminal keeps its 6 movements and gains no U-turn or ramp-to-ramp movement",
    "through_movements_cross_over_at_each_terminal": "both arterial through movements pass a crossover node at each terminal",
    "left_side_running_between_terminals": "between the terminals, arterial traffic runs on the left side",
    "no_collateral_changes": "nothing outside the bound terminals and the links between them changes",
    "ten_spui_movements": "(SPUI) the single point carries the 10 through, left and right movements and no U-turn or ramp-to-ramp movement",
    "single_signal_controls_all_ramp_movements": "(SPUI) one signal controls every ramp and cross-street movement",
    "all_entries_reach_all_exits": "(roundabout terminals) every entry reaches every exit, U-turns included",
    "counterclockwise_circulation": "(roundabout terminals) each ring circulates counter-clockwise",
    "yield_controlled_entries": "(roundabout terminals) entries yield; no signals on the rings",
}

GRID_N = 6
GRID_ROWS = string.ascii_uppercase[:GRID_N]


# ---------- grid scaffolding (rows A.. top to bottom, columns 1.. left to right) ----------

def cell_center(cell: str, extent) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = extent
    row, col = GRID_ROWS.index(cell[0].upper()), int(cell[1:]) - 1
    w, h = (xmax - xmin) / GRID_N, (ymax - ymin) / GRID_N
    return (xmin + (col + 0.5) * w, ymax - (row + 0.5) * h)


def is_cell(s) -> bool:
    return (isinstance(s, str) and len(s) >= 2 and s[0].upper() in GRID_ROWS
            and s[1:].isdigit() and 1 <= int(s[1:]) <= GRID_N)


# ---------- resolved terminals ----------

@dataclass
class Terminal:
    name: str
    links: dict[str, int]                                   # role -> host link id
    ports: dict[str, int] = field(default_factory=dict)     # role -> port node id
    connectors: dict[tuple[str, str], int] = field(default_factory=dict)  # (in, out) -> link id
    group: str = ""                                         # control_group of its connectors in the base

    def port_nodes(self) -> list[int]:
        return list(self.ports.values())


def resolve_ports(net: Network, name: str, links: dict[str, int]) -> Terminal:
    t = Terminal(name=name, links=dict(links))
    for role, lid in links.items():
        l = net.links[lid]
        t.ports[role] = l["to_node_id"] if TERMINAL_ROLES[role][1] == "in" else l["from_node_id"]
    for a, b, _ in DIAMOND_CONNECTORS:
        c = net.find_link(t.ports[a], t.ports[b])
        if c:
            t.connectors[(a, b)] = c["link_id"]
    groups = [net.links[c].get("control_group") for c in t.connectors.values() if net.links[c].get("control_group")]
    t.group = max(set(groups), key=groups.count) if groups else name
    return t


def table_signature(net: Network) -> str:
    """Facility class read from the tables alone (no image): the deterministic cross-check."""
    if any(n["node_type"] == "crossover" for n in net.nodes.values()):
        return "ddi"
    groups = {}
    for l in net.links.values():
        if l["facility_type"] != "ramp":
            continue
        end = net.nodes[l["to_node_id"]] if net.nodes[l["from_node_id"]]["node_type"] == "merge_diverge" \
            else net.nodes[l["from_node_id"]]
        kind = "off" if net.nodes[l["from_node_id"]]["node_type"] == "merge_diverge" else "on"
        g = groups.setdefault(end.get("control_group") or "?", {"off": 0, "on": 0})
        g[kind] += 1
    counts = sorted((g["off"], g["on"]) for k, g in groups.items() if k != "?")
    if counts == [(1, 1), (1, 1)]:
        return "diamond"
    if counts == [(2, 2)]:
        return "spui"
    return "unknown"


# ---------- prompt ----------

SYSTEM_PROMPT = """You are the proposer stage of a traffic-network scenario-editing pipeline.
You never draw, digitize or estimate coordinates. You answer only with IDs that exist in the
tables you are given and with labels from the closed catalogs. Topology, turning movements and
geometry are produced afterwards by deterministic templates and checked by code; anything you
name that is not in the tables is rejected. Return exactly one JSON object and nothing else."""


def build_prompt(scenario: dict, net: Network, extent, *, hide_names: bool = False) -> str:
    node_rows = "\n".join(
        f"{n['node_id']},{'' if hide_names else n['name']},{n['node_type']},{n['x_coord']:g},{n['y_coord']:g}"
        for n in net.nodes.values())
    link_rows = "\n".join(
        f"{l['link_id']},{'' if hide_names else l['name']},{l['from_node_id']},{l['to_node_id']},{l['facility_type']}"
        for l in net.links.values())
    roles = "\n".join(f"  - {r}: {d[2]} (facility_type must be {d[0]})" for r, d in TERMINAL_ROLES.items())
    facts = "\n".join(f"  - {k}: {v}" for k, v in EXPECTED_FACTS.items())
    ops = ", ".join(OPERATIONS)
    classes = ", ".join(FACILITY_CLASSES)
    panel_a = "satellite image of the site BEFORE the project (Mapbox imagery, no network drawn on it)"
    return f"""SCENARIO
{scenario['sentence']}

IMAGE
The image has two panels.
Panel A (left): {panel_a}.
Panel B (right): the GMNS base network rendered in a schematic local coordinate system.
  {scenario.get('local_cs_note', '')}
  Link IDs are printed in coloured boxes (grey freeway, blue arterial, green ramp); intersection
  connectors are orange and unlabelled. Node IDs are printed next to the white circles.
  A {GRID_N}x{GRID_N} grid is overlaid on panel B: rows {GRID_ROWS[0]}-{GRID_ROWS[-1]} top to bottom, columns 1-{GRID_N} left to right
  (cell "C2" = third row, second column).

NODE TABLE (node_id,name,node_type,x,y)
{node_rows}

LINK TABLE (link_id,name,from_node_id,to_node_id,facility_type); all links are directed
{link_rows}

CLOSED CATALOGS
facility_class: {classes}
operations: {ops}
terminal roles (one host link per role, per ramp terminal):
{roles}
expected_facts vocabulary:
{facts}

TASK
1. Read the facility class of the existing interchange from panel A and the tables.
2. Identify each ramp terminal (the at-grade junction where ramps meet the arterial). Name it by
   its compass side along the arterial (for example "SW", "NE") and give the grid cell that contains it.
3. For each terminal bind every role to exactly one link_id from the LINK TABLE. For every link you
   name, copy its from_node_id and to_node_id from the table; a mismatch rejects the proposal.
4. Write the operation list that the scenario sentence asks for, using only catalog operations.
5. List the expected_facts (from the vocabulary) that must hold after the edit.
6. List everything the image and text do not establish (lane counts, signal timing, turn pockets,
   demand, ...) in not_established. Unknown is not zero.

OUTPUT: one JSON object with the keys in exactly this order
{{
  "evidence": ["<short observations from the image and tables, written before any decision>"],
  "facility_class": "<facility_class from the catalog>",
  "arterial_axis_nodes": [<boundary node id at one end of the arterial>, <boundary node id at the other end>],
  "terminals": [
    {{"name": "<side>", "anchor_cell": "<grid cell>",
      "bindings": {{"<role>": {{"link_id": <int>, "from_node_id": <int>, "to_node_id": <int>}}, ...all 6 roles...}}}}
  ],
  "operations": [{{"op": "<operation>", "from_class": "<facility_class>", "to_class": "<facility_class>", "terminals": ["<side>", ...]}}],
  "expected_facts": ["<fact>", ...],
  "not_established": ["<free text>", ...]
}}"""


def build_repair_prompt(query: str, previous: dict | str, defects: list[dict]) -> str:
    prev = previous if isinstance(previous, str) else json.dumps(previous, indent=1)
    lines = "\n".join(f"- [{d['gate']}] {d['where']}: {d['message']}" for d in defects)
    return (f"{query}\n\nYOUR PREVIOUS ANSWER\n{prev}\n\nTHE CHECKS REJECTED IT FOR THESE REASONS\n{lines}\n\n"
            "Return a corrected JSON object with the same keys in the same order. Fix only what the "
            "defects name; re-read the LINK TABLE for every ID you use.")


# ---------- gates ----------

def _defect(gate, where, message):
    return {"gate": gate, "where": where, "message": message}


def validate(proposal, net: Network, extent) -> tuple[list[dict], list[Terminal]]:
    """Run P1-P8 in order. Later gates only run when earlier ones leave something to check."""
    d: list[dict] = []
    # P1 schema
    if not isinstance(proposal, dict):
        return [_defect("P1 schema", "root", "answer is not a JSON object")], []
    for key, typ in [("evidence", list), ("facility_class", str), ("terminals", list),
                     ("operations", list), ("expected_facts", list), ("not_established", list)]:
        if not isinstance(proposal.get(key), typ):
            d.append(_defect("P1 schema", key, f"missing or not a {typ.__name__}"))
    if d:
        return d, []
    if proposal["facility_class"] not in FACILITY_CLASSES:
        d.append(_defect("P1 schema", "facility_class", f"'{proposal['facility_class']}' is not in the catalog"))
    for i, op in enumerate(proposal["operations"]):
        if not isinstance(op, dict) or op.get("op") not in OPERATIONS:
            d.append(_defect("P1 schema", f"operations[{i}]", f"operation {op!r} is not in the catalog"))
    for f in proposal["expected_facts"]:
        if f not in EXPECTED_FACTS:
            d.append(_defect("P1 schema", "expected_facts", f"'{f}' is not in the vocabulary"))
    names = []
    for i, t in enumerate(proposal["terminals"]):
        where = f"terminals[{i}]"
        if not isinstance(t, dict) or not isinstance(t.get("bindings"), dict):
            d.append(_defect("P1 schema", where, "terminal must be an object with 'bindings'"))
            continue
        names.append(t.get("name"))
        missing = set(TERMINAL_ROLES) - set(t["bindings"])
        extra = set(t["bindings"]) - set(TERMINAL_ROLES)
        if missing:
            d.append(_defect("P1 schema", where, f"roles not bound: {sorted(missing)}"))
        if extra:
            d.append(_defect("P1 schema", where, f"unknown roles: {sorted(extra)}"))
        for role, b in t["bindings"].items():
            if not (isinstance(b, dict) and all(isinstance(b.get(k), int) for k in ("link_id", "from_node_id", "to_node_id"))):
                d.append(_defect("P1 schema", f"{where}.{role}", "binding needs integer link_id, from_node_id, to_node_id"))
    if len(set(names)) != len(names):
        d.append(_defect("P1 schema", "terminals", "terminal names must be distinct"))
    if d:
        return d, []

    # P2 ID conformance, P3 echo-neighbours
    for t in proposal["terminals"]:
        for role, b in t["bindings"].items():
            where = f"{t['name']}.{role}"
            l = net.links.get(b["link_id"])
            if l is None:
                d.append(_defect("P2 id exists", where, f"link {b['link_id']} is not in the link table"))
                continue
            if (l["from_node_id"], l["to_node_id"]) != (b["from_node_id"], b["to_node_id"]):
                d.append(_defect("P3 echo neighbours", where,
                                 f"link {b['link_id']} runs {l['from_node_id']}->{l['to_node_id']}, "
                                 f"answer echoed {b['from_node_id']}->{b['to_node_id']}"))
    for n in proposal.get("arterial_axis_nodes") or []:
        if n not in net.nodes:
            d.append(_defect("P2 id exists", "arterial_axis_nodes", f"node {n} is not in the node table"))
    if d:
        return d, []

    # P4 role fit against the diamond terminal template
    terminals = []
    for t in proposal["terminals"]:
        term = resolve_ports(net, t["name"], {r: b["link_id"] for r, b in t["bindings"].items()})
        terminals.append(term)
        for role, lid in term.links.items():
            want = TERMINAL_ROLES[role][0]
            got = net.links[lid]["facility_type"]
            if got != want:
                d.append(_defect("P4 role fit", f"{term.name}.{role}", f"link {lid} is {got}, role needs {want}"))
        if len(set(term.ports.values())) != len(term.ports):
            d.append(_defect("P4 role fit", term.name, f"roles resolve to repeated port nodes {term.ports}"))
        for a, b, kind in DIAMOND_CONNECTORS:
            if (a, b) not in term.connectors:
                d.append(_defect("P4 role fit", term.name,
                                 f"no {kind} connector {a}({term.ports[a]}) -> {b}({term.ports[b]}); "
                                 "these bindings do not describe a diamond terminal"))
        ports = set(term.port_nodes())
        known = set(term.connectors.values())
        for l in net.links.values():
            if l["from_node_id"] in ports and l["to_node_id"] in ports and l["link_id"] not in known:
                d.append(_defect("P4 role fit", term.name, f"unexpected connector {l['link_id']} between port nodes"))

    # P5 the bridge links continue from one terminal to the other
    if len(terminals) == 2 and not d:
        a, b = terminals
        if a.links["bridge_out"] != b.links["bridge_in"] or b.links["bridge_out"] != a.links["bridge_in"]:
            d.append(_defect("P5 bridge continuity", f"{a.name}/{b.name}",
                             f"{a.name}.bridge_out={a.links['bridge_out']} must equal {b.name}.bridge_in={b.links['bridge_in']}, "
                             f"and {b.name}.bridge_out={b.links['bridge_out']} must equal {a.name}.bridge_in={a.links['bridge_in']}"))

    # P6 anchor cell snaps to the terminal it names
    if terminals and not d:
        centroids = {t.name: _centroid(net, t.port_nodes()) for t in terminals}
        for t in proposal["terminals"]:
            cell = t.get("anchor_cell")
            if not is_cell(cell):
                d.append(_defect("P6 anchor", t["name"], f"anchor_cell {cell!r} is not a grid cell"))
                continue
            c = cell_center(cell, extent)
            nearest = min(centroids, key=lambda k: math.dist(c, centroids[k]))
            if nearest != t["name"]:
                d.append(_defect("P6 anchor", t["name"], f"cell {cell} lies nearest terminal {nearest}"))

    # P7 facility class against the table signature
    sig = table_signature(net)
    if sig != "unknown" and proposal["facility_class"] != sig:
        d.append(_defect("P7 class cross-check", "facility_class",
                         f"answer says {proposal['facility_class']}, the tables read as {sig}"))
    expected_terminals = {"diamond": 2, "tight_diamond": 2, "spui": 1}.get(proposal["facility_class"])
    if expected_terminals and len(proposal["terminals"]) != expected_terminals:
        d.append(_defect("P7 class cross-check", "terminals",
                         f"{proposal['facility_class']} has {expected_terminals} ramp terminals, answer gives {len(proposal['terminals'])}"))

    # P8 operations are executable here and consistent with the class
    if not proposal["operations"]:
        d.append(_defect("P8 operations", "operations", "empty operation list"))
    for i, op in enumerate(proposal["operations"]):
        where = f"operations[{i}]"
        if not OPERATIONS[op["op"]]:
            d.append(_defect("P8 operations", where, f"'{op['op']}' is in the catalog but not implemented yet"))
            continue
        if op.get("from_class") != proposal["facility_class"]:
            d.append(_defect("P8 operations", where, f"from_class {op.get('from_class')!r} differs from facility_class"))
        if (op.get("from_class"), op.get("to_class")) not in IMPLEMENTED_CONVERSIONS:
            d.append(_defect("P8 operations", where, f"no template for {op.get('from_class')} -> {op.get('to_class')}"))
        unknown = set(op.get("terminals") or []) - {t.name for t in terminals}
        if unknown or not op.get("terminals"):
            d.append(_defect("P8 operations", where, f"terminals {op.get('terminals')} must name bound terminals"))
    return d, terminals


def _centroid(net: Network, node_ids):
    pts = [net.xy(n) for n in node_ids]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def arterial_bearing(net: Network, axis_nodes) -> float | None:
    """Bearing of the arterial in the network's own coordinate system, from two named nodes."""
    if not axis_nodes or len(axis_nodes) != 2:
        return None
    (x1, y1), (x2, y2) = net.xy(axis_nodes[0]), net.xy(axis_nodes[1])
    return round(math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360, 1)
