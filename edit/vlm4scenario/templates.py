"""Parametric facility templates. They own topology and geometry; the proposer only binds roles.

Every template takes the base network and the bound terminals of a diamond and returns
(after_network, log, spec). `spec` tells the check chain what the new facility must do:
  groups    control group -> required and forbidden port-to-port movements
  families  template-specific check families to run (see checks.FAMILIES)
  host_links  bound links whose port end may move or that may be removed

diamond -> ddi         swap the bridge-side ports, split both through connectors at their
                       crossing with two co-located but distinct crossover nodes
diamond -> spui        one signalised junction over the freeway; bridge ports and links removed
diamond -> roundabout  each ramp terminal becomes a one-way counter-clockwise ring with
                       yield-controlled entries
"""
from __future__ import annotations

import math

from .gmns import Network, polyline_length_km
from .proposal import DIAMOND_CONNECTORS, FORBIDDEN_PAIRS, Terminal


class TemplateError(RuntimeError):
    pass


class IdRegistry:
    """Hands out new IDs that collide with nothing in the base network.

    New nodes start at the next block of ten above the largest node ID and new links at the
    next block of a hundred. The second half of a split link gets link_id + 100 when free, so a
    split link and its continuation read together.
    """

    def __init__(self, net: Network):
        self.used_nodes = set(net.nodes)
        self.used_links = set(net.links)
        self._next_node = (max(self.used_nodes) // 10 + 1) * 10 + 1
        self._next_link = (max(self.used_links) // 100 + 1) * 100 + 1

    def node(self) -> int:
        while self._next_node in self.used_nodes:
            self._next_node += 1
        self.used_nodes.add(self._next_node)
        return self._next_node

    def link(self) -> int:
        while self._next_link in self.used_links:
            self._next_link += 1
        self.used_links.add(self._next_link)
        return self._next_link

    def split_link(self, link_id: int) -> int:
        cand = link_id + 100
        if cand in self.used_links:
            return self.link()
        self.used_links.add(cand)
        return cand


# ---------- small helpers ----------

def _segment_intersection(p1, p2, p3, p4):
    """Intersection point of segments p1p2 and p3p4, or None if they do not cross."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if not (0 < t < 1 and 0 < u < 1):
        return None
    return (round(x1 + t * (x2 - x1), 6), round(y1 + t * (y2 - y1), 6))


def _centroid(net: Network, nodes):
    pts = [net.xy(n) for n in nodes]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _add_node(net, nid, name, xy, node_type, ctrl, group, scenario, status):
    net.nodes[nid] = dict(node_id=nid, name=name, x_coord=round(xy[0], 6), y_coord=round(xy[1], 6),
                          node_type=node_type, ctrl_type=ctrl, scenario=scenario, control_group=group,
                          geometry_status=status)


def _add_link(net, proto, lid, a, b, name, group, status, geometry=None):
    pts = geometry or [net.xy(a), net.xy(b)]
    net.links[lid] = dict(proto, link_id=lid, name=name, from_node_id=a, to_node_id=b,
                          facility_type="intersection_connector", control_group=group,
                          control_status=status, geometry=pts, length=polyline_length_km(pts), layer=1)


def _drop_links(net, pred):
    gone = [i for i, l in net.links.items() if pred(l)]
    for i in gone:
        del net.links[i]
    return gone


def _finish(net, scenario):
    for n in net.nodes.values():
        n["scenario"] = scenario
    for l in net.links.values():
        l["scenario"] = scenario
    net.rebuild_movements()


def _host_links(terminals):
    return sorted({lid for t in terminals for lid in t.links.values()})


def _connector_proto(base: Network, terminals) -> dict:
    return dict(base.links[next(iter(terminals[0].connectors.values()))])


# ---------- diamond -> DDI ----------

def diamond_to_ddi(base: Network, terminals: list[Terminal], scenario: str):
    net = base.copy()
    reg = IdRegistry(base)
    log: list[str] = []
    # Deterministic order (by smallest port node), so IDs do not depend on how the model listed terminals.
    for t in sorted(terminals, key=lambda t: min(t.port_nodes())):
        b_in, b_out = t.ports["bridge_in"], t.ports["bridge_out"]

        # 1. carriageways cross over: swap the bridge-side ports
        (xi, yi), (xo, yo) = net.xy(b_in), net.xy(b_out)
        net.nodes[b_in].update(x_coord=xo, y_coord=yo)
        net.nodes[b_out].update(x_coord=xi, y_coord=yi)
        for l in net.links.values():
            if {l["from_node_id"], l["to_node_id"]} & {b_in, b_out}:
                net.refresh_link_geometry(l["link_id"])
        log.append(f"{t.name} ramp terminal: the two arterial directions swap sides (nodes {b_in} and {b_out}), "
                   "so traffic drives on the left between the terminals")

        # 2. split each through connector at the crossing point
        t1 = t.connectors[("approach_in", "bridge_out")]
        t2 = t.connectors[("bridge_in", "departure_out")]
        g1, g2 = net.links[t1]["geometry"], net.links[t2]["geometry"]
        if len(g1) != 2 or len(g2) != 2:
            raise TemplateError(f"{t.name}: through connectors must be straight two-point links")
        x = _segment_intersection(g1[0], g1[1], g2[0], g2[1])
        if x is None:
            raise TemplateError(f"{t.name}: through connectors {t1} and {t2} do not cross after the swap")
        for cid in (t1, t2):
            c = net.links[cid]
            nid = reg.node()
            _add_node(net, nid, f"X_{t.name}_{c['name']}", x, "crossover", "signal", t.group, scenario,
                      base.nodes[c["from_node_id"]]["geometry_status"])
            new_id = reg.split_link(cid)
            tail = dict(c, link_id=new_id, name=c["name"] + "_out", from_node_id=nid,
                        geometry=[x, c["geometry"][-1]], control_status="signal_required_timing_unknown")
            c.update(name=c["name"] + "_in", to_node_id=nid, geometry=[c["geometry"][0], x],
                     control_status="signal_required_timing_unknown")
            for part in (c, tail):
                part["length"] = polyline_length_km(part["geometry"])
            net.links[new_id] = tail
            log.append(f"{t.name} ramp terminal: through link {cid} now passes crossover node {nid} and continues as link {new_id}")

        # 3. turn connectors keep their ports; the signal now sits at the crossovers
        for cid in t.connectors.values():
            if cid not in (t1, t2):
                net.links[cid]["control_status"] = "control_to_verify"

    _finish(net, scenario)
    spec = {"target": "ddi", "families": ["ddi"], "host_links": _host_links(terminals),
            "movement_fact": "six_terminal_movements_per_terminal",
            "groups": {t.group: {"required": [(t.ports[a], t.ports[b], k) for a, b, k in DIAMOND_CONNECTORS],
                                 "forbidden": [(t.ports[a], t.ports[b], "U-turn or ramp-to-ramp") for a, b in FORBIDDEN_PAIRS]}
                       for t in terminals}}
    return net, log, spec


# ---------- diamond -> SPUI ----------

SPUI_HALF_LENGTH = 160.0     # m from the single point to the outer approach ports (schematic)
SPUI_RAMP_ALONG = 110.0      # m along the arterial from the single point to the ramp ports
SPUI_RAMP_OFFSET = 75.0      # m lateral offset of the ramp ports


def diamond_to_spui(base: Network, terminals: list[Terminal], scenario: str):
    if len(terminals) != 2:
        raise TemplateError("diamond -> SPUI needs exactly two bound terminals")
    net = base.copy()
    reg = IdRegistry(base)
    t1, t2 = sorted(terminals, key=lambda t: min(t.port_nodes()))
    c1, c2 = _centroid(base, t1.port_nodes()), _centroid(base, t2.port_nodes())
    cx, cy = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
    d = math.dist(c1, c2)
    ux, uy = (c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d
    nx, ny = -uy, ux                                   # left normal of the arterial axis

    def lateral(node):
        x, y = base.xy(node)
        return (x - cx) * nx + (y - cy) * ny

    def place(along, lat):
        return (round(cx + along * ux + lat * nx, 6), round(cy + along * uy + lat * ny, 6))

    log = ["Single intersection placed midway between the two old ramp terminals"]
    for t, sign in ((t1, -1), (t2, 1)):
        for role in ("approach_in", "departure_out"):
            n = t.ports[role]
            net.nodes[n]["x_coord"], net.nodes[n]["y_coord"] = place(sign * SPUI_HALF_LENGTH, lateral(n))
        for role in ("offramp_in", "onramp_out"):
            n = t.ports[role]
            side = 1 if lateral(n) > 0 else -1
            net.nodes[n]["x_coord"], net.nodes[n]["y_coord"] = place(sign * SPUI_RAMP_ALONG, side * SPUI_RAMP_OFFSET)
    proto = _connector_proto(base, terminals)
    ports = {n for t in terminals for n in t.port_nodes()}
    removed_nodes = {t.ports[r] for t in terminals for r in ("bridge_in", "bridge_out")}
    gone = _drop_links(net, lambda l: {l["from_node_id"], l["to_node_id"]} & removed_nodes
                       or (l["from_node_id"] in ports and l["to_node_id"] in ports))
    for n in removed_nodes:
        del net.nodes[n]
    log.append(f"Removed the freeway-side ports of both old terminals (nodes {', '.join(map(str, sorted(removed_nodes)))}) "
               f"and links {', '.join(map(str, sorted(gone)))}")
    for lid in _host_links(terminals):
        if lid in net.links:
            net.refresh_link_geometry(lid)
    log.append(f"Moved the arterial ports to {SPUI_HALF_LENGTH:g} m and the ramp ends to {SPUI_RAMP_ALONG:g} m from the centre")

    P = lambda t, r: t.ports[r]  # noqa: E731
    moves = [(P(t1, "approach_in"), P(t2, "departure_out"), "through"), (P(t2, "approach_in"), P(t1, "departure_out"), "through"),
             (P(t1, "approach_in"), P(t1, "onramp_out"), "right"), (P(t2, "approach_in"), P(t2, "onramp_out"), "right"),
             (P(t1, "approach_in"), P(t2, "onramp_out"), "left"), (P(t2, "approach_in"), P(t1, "onramp_out"), "left"),
             (P(t1, "offramp_in"), P(t1, "departure_out"), "right"), (P(t2, "offramp_in"), P(t2, "departure_out"), "right"),
             (P(t1, "offramp_in"), P(t2, "departure_out"), "left"), (P(t2, "offramp_in"), P(t1, "departure_out"), "left")]
    group = "SPUI"
    for a, b, kind in moves:
        lid = reg.link()
        status = "control_to_verify" if kind == "right" else "signal_required_timing_unknown"
        _add_link(net, proto, lid, a, b, f"SPUI_{kind}_{a}_{b}", group, status)
    for n in ports - removed_nodes:
        net.nodes[n]["control_group"] = group
    log.append(f"Added {len(moves)} intersection connectors through one signal-controlled intersection")
    _finish(net, scenario)
    forbidden = [(P(t1, "approach_in"), P(t1, "departure_out"), "U-turn"), (P(t2, "approach_in"), P(t2, "departure_out"), "U-turn")]
    forbidden += [(P(a, "offramp_in"), P(b, "onramp_out"), "ramp-to-ramp") for a in (t1, t2) for b in (t1, t2)]
    spec = {"target": "spui", "families": ["spui"], "host_links": _host_links(terminals),
            "movement_fact": "ten_spui_movements", "groups": {group: {"required": moves, "forbidden": forbidden}}}
    return net, log, spec


# ---------- diamond -> roundabout ramp terminals ----------

RING_RADIUS = 30.0           # m; inscribed circle diameter 60 m (197 ft), schematic
PORT_CLEARANCE = 2.2         # ports closer than this many radii are pushed out along their ray

ENTRY_ROLES = ("approach_in", "bridge_in", "offramp_in")
EXIT_ROLES = ("departure_out", "bridge_out", "onramp_out")


def diamond_to_roundabout(base: Network, terminals: list[Terminal], scenario: str):
    net = base.copy()
    reg = IdRegistry(base)
    proto = _connector_proto(base, terminals)
    log, groups = [], {}
    for t in sorted(terminals, key=lambda t: min(t.port_nodes())):
        ports = set(t.port_nodes())
        gone = _drop_links(net, lambda l: l["from_node_id"] in ports and l["to_node_id"] in ports)
        cx, cy = _centroid(base, ports)
        moved = []
        for n in ports:
            px, py = base.xy(n)
            d = math.dist((px, py), (cx, cy))
            if d < PORT_CLEARANCE * RING_RADIUS:
                k = PORT_CLEARANCE * RING_RADIUS / d
                net.nodes[n]["x_coord"], net.nodes[n]["y_coord"] = round(cx + (px - cx) * k, 6), round(cy + (py - cy) * k, 6)
                moved.append(n)
        for l in list(net.links.values()):
            if {l["from_node_id"], l["to_node_id"]} & set(moved):
                net.refresh_link_geometry(l["link_id"])
        if moved:
            log.append(f"{t.name} ramp terminal: moved ports {sorted(moved)} out to {PORT_CLEARANCE * RING_RADIUS:g} m from the ring centre")
        ring = []                                            # (angle, ring node, port role)
        for role in ENTRY_ROLES + EXIT_ROLES:
            px, py = base.xy(t.ports[role])
            ang = math.atan2(py - cy, px - cx) % (2 * math.pi)
            nid = reg.node()
            entry = role in ENTRY_ROLES
            _add_node(net, nid, f"R_{t.name}_{role}", (cx + RING_RADIUS * math.cos(ang), cy + RING_RADIUS * math.sin(ang)),
                      "roundabout", "yield" if entry else "", t.group, scenario, base.nodes[t.ports[role]]["geometry_status"])
            ring.append((ang, nid, role))
            lid = reg.link()
            if entry:
                _add_link(net, proto, lid, t.ports[role], nid, f"{t.name}_{role}_entry", t.group, "yield_entry")
            else:
                _add_link(net, proto, lid, nid, t.ports[role], f"{t.name}_{role}_exit", t.group, "exit")
        ring.sort()
        for (a0, n0, _), (a1, n1, _) in zip(ring, ring[1:] + ring[:1]):
            if a1 <= a0:
                a1 += 2 * math.pi                            # counter-clockwise, wrap past 0
            steps = max(2, int((a1 - a0) / math.radians(10)) + 1)
            pts = [net.xy(n0)] + [(round(cx + RING_RADIUS * math.cos(a0 + (a1 - a0) * k / steps), 6),
                                   round(cy + RING_RADIUS * math.sin(a0 + (a1 - a0) * k / steps), 6))
                                  for k in range(1, steps)] + [net.xy(n1)]
            _add_link(net, proto, reg.link(), n0, n1, f"{t.name}_circulating_{n0}_{n1}", t.group, "circulating", pts)
        log.append(f"{t.name} ramp terminal: replaced intersection connectors {sorted(gone)} with a counter-clockwise roundabout of "
                   f"{len(ring)} nodes (radius {RING_RADIUS:g} m) and yield-controlled entries")
        groups[t.group] = {"required": [(t.ports[a], t.ports[b], "entry-to-exit") for a in ENTRY_ROLES for b in EXIT_ROLES],
                           "forbidden": []}
    _finish(net, scenario)
    spec = {"target": "roundabout_diamond", "families": ["roundabout"], "host_links": _host_links(terminals),
            "movement_fact": "all_entries_reach_all_exits", "groups": groups}
    return net, log, spec


TEMPLATES = {("replace_facility", "diamond", "ddi"): diamond_to_ddi,
             ("replace_facility", "diamond", "spui"): diamond_to_spui,
             ("replace_facility", "diamond", "roundabout_diamond"): diamond_to_roundabout}


def baseline_spec(terminals: list[Terminal]) -> dict:
    """What the existing diamond already does, for running the same checks on the base network."""
    return {"target": "diamond", "families": [], "host_links": _host_links(terminals),
            "movement_fact": "six_terminal_movements_per_terminal",
            "groups": {t.group: {"required": [(t.ports[a], t.ports[b], k) for a, b, k in DIAMOND_CONNECTORS],
                                 "forbidden": [(t.ports[a], t.ports[b], "U-turn or ramp-to-ramp") for a, b in FORBIDDEN_PAIRS]}
                       for t in terminals}}
