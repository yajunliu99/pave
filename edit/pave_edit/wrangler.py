"""Network Wrangler interop: GMNS <-> Network Wrangler roadway tables, and project cards.

Network Wrangler (https://github.com/network-wrangler) keeps a roadway network as links,
nodes and shapes in WGS84, and records each project as a project card (YAML, validated by
the `projectcard` package). This module needs only the standard library (PyYAML if present);
tools/wrangler_roundtrip.py applies the card with network_wrangler itself.

Coordinates: the schematic network is placed on the site by a rigid transform (rotation about
the site centre, metres to WGS84 on a local tangent plane). It is exactly invertible, so a
round trip through Network Wrangler can be compared node by node. It is a placement of the
schematic drawing, not a survey.

What does not carry over:
  - movements: Network Wrangler has no movement table; they are rebuilt from the links with
    the same rule as gmns.Network.rebuild_movements;
  - unknown lane counts: the project card schema requires `lanes` >= 0 and the roadway table
    does not allow nulls, so an unknown count is written as LANES_PLACEHOLDER with
    `lanes_known: false` and restored to blank on the way back;
  - curved geometry of added links: a card adds a link as a straight A-B line.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .gmns import Network, polyline_length_km

EARTH_R = 6378137.0
KM_PER_MILE = 1.609344
LANES_PLACEHOLDER = 0
ROADWAY = {"freeway": "motorway", "arterial": "primary", "ramp": "motorway_link", "intersection_connector": "road"}
NO_WALK = {"motorway", "motorway_link"}          # pedestrians and bicycles are not allowed on freeways and ramps
NODE_ATTRS = ["name", "node_type", "ctrl_type", "control_group"]
LINK_ATTRS = ["name", "control_group", "control_status", "layer"]


class SitePlacement:
    """Schematic metres <-> WGS84. `x_bearing` is the compass bearing of the schematic +x axis;
    +y points 90 degrees counter-clockwise from it."""

    def __init__(self, lat: float, lon: float, x_bearing: float):
        self.lat0, self.lon0 = lat, lon
        bx, by = math.radians(x_bearing), math.radians(x_bearing - 90)
        self.ex, self.nx = math.sin(bx), math.cos(bx)        # east/north components of +x
        self.ey, self.ny = math.sin(by), math.cos(by)        # east/north components of +y
        self.k = EARTH_R * math.cos(math.radians(lat))

    @classmethod
    def for_scenario(cls, scenario: dict) -> "SitePlacement":
        site = scenario["site"]
        return cls(site["lat"], site["lon"], site["crossroad_bearing_deg"])

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        e, n = x * self.ex + y * self.ey, x * self.nx + y * self.ny
        return (round(self.lon0 + math.degrees(e / self.k), 9), round(self.lat0 + math.degrees(n / EARTH_R), 9))

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        e, n = math.radians(lon - self.lon0) * self.k, math.radians(lat - self.lat0) * EARTH_R
        return (round(e * self.ex + n * self.nx, 3), round(e * self.ey + n * self.ny, 3))


# ---------- GMNS -> Network Wrangler roadway tables ----------

def node_record(n: dict, place: SitePlacement) -> dict:
    lon, lat = place.to_lonlat(n["x_coord"], n["y_coord"])
    rec = {"model_node_id": n["node_id"], "X": lon, "Y": lat}
    rec.update({f"gmns_{k}" if k == "node_type" else k: n.get(k, "") for k in NODE_ATTRS})
    return rec


def link_record(l: dict) -> dict:
    roadway = ROADWAY.get(l["facility_type"], "road")
    lanes = l.get("lanes", "")
    known = str(lanes).strip() != ""
    rec = {"model_link_id": l["link_id"], "A": l["from_node_id"], "B": l["to_node_id"],
           "name": l.get("name") or "unknown", "roadway": roadway,
           "lanes": int(float(lanes)) if known else LANES_PLACEHOLDER, "lanes_known": known,
           "distance": round(l["length"] / KM_PER_MILE, 6),
           "drive_access": True, "walk_access": roadway not in NO_WALK, "bike_access": roadway not in NO_WALK,
           "gmns_facility_type": l["facility_type"]}
    rec.update({k: l.get(k, "") for k in LINK_ATTRS})
    return rec


def write_roadway(net: Network, place: SitePlacement, folder: Path) -> None:
    """links.json (records) and nodes.geojson, the layout network_wrangler.load_roadway_from_dir reads."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    links = [link_record(l) for l in net.links.values()]
    (folder / "links.json").write_text(json.dumps(links, indent=1))
    feats = []
    for n in net.nodes.values():
        rec = node_record(n, place)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [rec["X"], rec["Y"]]},
                      "properties": rec})
    (folder / "nodes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))


# ---------- project card ----------

def _ids(key, ids):
    return {key: sorted(ids), "ignore_missing": False}


def project_card(base: Network, after: Network, place: SitePlacement, *, project: str, tags: list[str],
                 notes: str) -> dict:
    """Describe base -> after as one Network Wrangler project card.

    Network Wrangler applies the changes in order:
      1. move each moved node (one change per node; it redraws the attached links). Moves come
         first, while every port still has links on both sides: network_wrangler 1.0b4 fails to
         move a node that is the A node (or the B node) of no link. A moved node that already
         lacks one side in the base is deleted and re-added instead;
      2. delete links that were removed or whose end nodes changed, and nodes that were removed;
      3. add new nodes, then new links and the re-routed links;
      4. set changed attributes on links that kept their identity;
      5. set changed attributes on nodes with a `pycode` change, because node property changes
         raise AttributeError in network_wrangler 1.0b4 (edit_node_property calls .get on a
         pydantic model).
    """
    changes = []
    b_links, a_links = base.links, after.links
    common_nodes = base.nodes.keys() & after.nodes.keys()
    moved = sorted(i for i in common_nodes if base.xy(i) != after.xy(i))
    one_sided = {i for i in moved if not base.in_links(i) or not base.out_links(i)}
    rerouted = {i for i in b_links.keys() & a_links.keys()
                if (b_links[i]["from_node_id"], b_links[i]["to_node_id"]) != (a_links[i]["from_node_id"], a_links[i]["to_node_id"])
                or {b_links[i]["from_node_id"], b_links[i]["to_node_id"]} & one_sided}

    for i in moved:
        if i in one_sided:
            continue
        lon, lat = place.to_lonlat(*after.xy(i))
        changes.append({"roadway_property_change": {"facility": {"nodes": _ids("model_node_id", [i])},
                                                    "property_changes": {"X": {"set": lon}, "Y": {"set": lat}}}})

    del_links = (b_links.keys() - a_links.keys()) | rerouted
    del_nodes = (base.nodes.keys() - after.nodes.keys()) | one_sided
    deletion = {}
    if del_links:
        deletion["links"] = _ids("model_link_id", del_links)
    if del_nodes:
        deletion["nodes"] = _ids("model_node_id", del_nodes)
    if deletion:                                 # no clean_shapes: the exported base carries no shapes table
        changes.append({"roadway_deletion": deletion})

    new_nodes = sorted((after.nodes.keys() - base.nodes.keys()) | one_sided)
    new_links = sorted((a_links.keys() - b_links.keys()) | rerouted)
    addition = {}
    if new_nodes:
        addition["nodes"] = [node_record(after.nodes[i], place) for i in new_nodes]
    if new_links:
        addition["links"] = [link_record(a_links[i]) for i in new_links]
    if addition:
        changes.append({"roadway_addition": addition})

    groups: dict[tuple, list[int]] = {}
    for i in sorted((b_links.keys() & a_links.keys()) - rerouted):
        for k, v in _attr_sets(b_links[i], a_links[i], LINK_ATTRS).items():
            groups.setdefault((k, json.dumps(v["set"])), []).append(i)
    for (k, v), ids in groups.items():
        changes.append({"roadway_property_change": {"facility": {"links": _ids("model_link_id", ids)},
                                                    "property_changes": {k: {"set": json.loads(v)}}}})

    ngroups: dict[tuple, list[int]] = {}
    for i in sorted(common_nodes - one_sided):
        for k, v in _attr_sets(base.nodes[i], after.nodes[i], NODE_ATTRS, gmns_prefix=True).items():
            ngroups.setdefault((k, json.dumps(v["set"])), []).append(i)
    if ngroups:
        lines = ["# Node attribute edits written as code: node property changes fail in network_wrangler 1.0b4.",
                 "_n = self.nodes_df.copy()"]
        for (k, v), ids in ngroups.items():
            lines.append(f"_n.loc[_n['model_node_id'].isin({sorted(ids)}), {k!r}] = {json.loads(v)!r}")
        lines.append("self.nodes_df = _n")
        changes.append({"pycode": "\n".join(lines)})
    return {"project": project, "tags": tags, "notes": notes, "changes": changes}


def _attr_sets(old: dict, new: dict, keys, gmns_prefix=False) -> dict:
    out = {}
    for k in keys:
        if str(old.get(k, "")) != str(new.get(k, "")):
            name = f"gmns_{k}" if gmns_prefix and k == "node_type" else k
            out[name] = {"set": new.get(k, "")}
    return out


def write_card(card: dict, path: Path) -> Path:
    path = Path(path)
    try:
        import yaml
        path.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=False, width=110))
    except ImportError:                          # JSON is valid YAML; keep the .yml name readable by projectcard
        path.write_text(json.dumps(card, indent=1))
    return path


# ---------- Network Wrangler -> GMNS ----------

def to_gmns(links: list[dict], nodes: list[dict], place: SitePlacement, scenario: str = "") -> Network:
    """Rebuild a GMNS network from Network Wrangler link and node records (e.g. DataFrame.to_dict('records'))."""
    gn = {}
    for r in nodes:
        x, y = place.to_xy(float(r["X"]), float(r["Y"]))
        nid = int(r["model_node_id"])
        gn[nid] = {"node_id": nid, "name": _s(r.get("name")), "x_coord": x, "y_coord": y,
                   "node_type": _s(r.get("gmns_node_type")), "ctrl_type": _s(r.get("ctrl_type")),
                   "scenario": scenario, "control_group": _s(r.get("control_group")),
                   "geometry_status": "schematic_not_georeferenced"}
    gl = {}
    for r in links:
        lid, a, b = int(r["model_link_id"]), int(r["A"]), int(r["B"])
        pts = [(gn[a]["x_coord"], gn[a]["y_coord"]), (gn[b]["x_coord"], gn[b]["y_coord"])]
        flag = r.get("lanes_known")
        flag = bool(flag) if flag is not None and flag == flag else True
        known = flag or int(r["lanes"]) != LANES_PLACEHOLDER      # a later card may set lanes without the flag
        gl[lid] = {"link_id": lid, "name": _s(r.get("name")), "from_node_id": a, "to_node_id": b,
                   "directed": "true", "dir_flag": 1, "length": polyline_length_km(pts),
                   "facility_type": _s(r.get("gmns_facility_type")) or r.get("roadway", ""),
                   "lanes": str(int(r["lanes"])) if known else "", "free_speed": "", "capacity": "",
                   "geometry": pts, "scenario": scenario, "control_group": _s(r.get("control_group")),
                   "control_status": _s(r.get("control_status")), "geometry_status": "schematic_not_georeferenced",
                   "operating_data_status": "unknown", "layer": int(float(r.get("layer") or 0))}
    net = Network(gn, gl)
    net.rebuild_movements()
    return net


def _s(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)
