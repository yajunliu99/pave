"""Section crop windows from GMNS link geometry.

Works for any number of legs at any angle. Lane counts and widths from the
network only place and size the crop; they are recorded as `crop_prior` and are
never part of a model prompt.
"""
from __future__ import annotations

import math

from ..hybrid.common import require
from .strips import Frame

DEFAULTS = {"lane_width_prior_m": 3.6, "crosswalk_margin_m": 5.0, "section_length_m": 28.0, "end_pad_m": 2.0,
            "lateral_margin_m": 7.0, "station_count": 5, "min_near_m": 8.0, "max_near_m": 35.0,
            "axis_probe_m": (25.0, 55.0), "driving_side": "right"}
CARDINALS = ("NB", "EB", "SB", "WB")
POINTS = ("NB", "NEB", "EB", "SEB", "SB", "SWB", "WB", "NWB")
ARM_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
ARM_TOLERANCE_DEG = 35.0


def parse_linestring(wkt):
    text = str(wkt).strip()
    require(text.upper().startswith("LINESTRING"), "Link geometry must be a LINESTRING")
    points = [tuple(float(v) for v in pair.split()[:2]) for pair in text[text.index("(") + 1:text.rindex(")")].split(",")]
    require(len(points) >= 2, "Link geometry needs two points")
    return points


def _int(value):
    return int(float(value))


def _truthy(value):
    return str(value).strip().lower() in ("1", "1.0", "true", "yes")


def _lanes(link):
    try:
        return max(1, _int(link.get("lanes")))
    except (TypeError, ValueError):
        return 1


def _metres(points, origin):
    scale = math.cos(math.radians(origin[1]))
    return [((lon - origin[0]) * 111320 * scale, (lat - origin[1]) * 110540) for lon, lat in points]


def _to_polyline(point, line):
    best = float("inf")
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        size = dx * dx + dy * dy
        t = 0. if not size else max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / size))
        best = min(best, math.hypot(point[0] - a[0] - dx * t, point[1] - a[1] - dy * t))
    return best


def shares_axis(link, links, tolerance_m=2.0):
    """A two-way road: flagged from_biway, or drawn as two directed links between the same nodes on one
    centreline (some networks leave the flag at 0)."""
    if _truthy(link.get("from_biway")):
        return True
    ends = (_int(link["to_node_id"]), _int(link["from_node_id"]))
    points = parse_linestring(link["geometry"])
    for other in links:
        if (_int(other["from_node_id"]), _int(other["to_node_id"])) != ends:
            continue
        line = _metres(parse_linestring(other["geometry"]), points[0])
        if sum(_to_polyline(p, line) for p in _metres(points, points[0])) / len(points) < tolerance_m:
            return True
    return False


def _motor(link):
    uses = str(link.get("allowed_uses") or "auto").lower()
    return "auto" in uses or uses in ("", "nan", "all")


def _along(points, distance):
    """Point `distance` pixels along a polyline, extrapolating past its end."""
    travelled = 0.0
    for a, b in zip(points, points[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length and travelled + length >= distance:
            t = (distance - travelled) / length
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        travelled += length
    a, b = points[-2], points[-1]
    length = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    t = (distance - travelled) / length
    return b[0] + (b[0] - a[0]) * t, b[1] + (b[1] - a[1]) * t


def _unit(a, b):
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    require(length > 0, "Degenerate link geometry")
    return (b[0] - a[0]) / length, (b[1] - a[1]) / length


def _at_projection(points, origin, unit, distance):
    """Polyline point lying `distance` pixels from `origin` along `unit`, and whether it is past the link's end."""
    def station(p):
        return (p[0] - origin[0]) * unit[0] + (p[1] - origin[1]) * unit[1]
    for a, b in zip(points, points[1:]):
        ta, tb = station(a), station(b)
        if ta < tb and ta <= distance <= tb:
            f = (distance - ta) / (tb - ta)
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), False
    beyond = distance > station(points[-1])
    end = points[-1] if beyond else points[0]
    gap = distance - station(end)
    return (end[0] + unit[0] * gap, end[1] + unit[1] * gap), beyond


def _heading(a, b):
    """Clockwise-from-up direction of a->b in image pixels."""
    return math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360


def cardinal(heading):
    return CARDINALS[round(heading / 90) % 4]


def compass_point(heading):
    return POINTS[round(heading / 45) % 8]


def _gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def road_arms(bearings, tolerance=ARM_TOLERANCE_DEG):
    """Arm name per leg: legs leaving the node within `tolerance` of each other are one road arm.

    bearings: clockwise-from-up direction in which each leg leaves the node. Arms are named by the
    eight-point compass of their mean bearing, numbered when two share a name.
    """
    arms = []
    for i in sorted(range(len(bearings)), key=bearings.__getitem__):
        arm = next((a for a in arms if _gap(bearings[i], a["bearing"]) <= tolerance), None)
        if arm is None:
            arms.append({"legs": [i], "bearing": bearings[i]})
        else:
            arm["legs"].append(i)
            vectors = [(math.sin(math.radians(bearings[j])), math.cos(math.radians(bearings[j]))) for j in arm["legs"]]
            arm["bearing"] = math.degrees(math.atan2(sum(v[0] for v in vectors), sum(v[1] for v in vectors))) % 360
    names = [ARM_NAMES[round(a["bearing"] / 45) % 8] for a in arms]
    out = [None] * len(bearings)
    for k, arm in enumerate(arms):
        name = names[k] if names.count(names[k]) == 1 else f"{names[k]}{names[:k + 1].count(names[k])}"
        for i in arm["legs"]:
            out[i] = (name, round(arm["bearing"], 1))
    return out


def frames_from_network(node_id, node_lonlat, links, to_pixel, mpp, image_size, policy=None):
    """One section spec per motor link entering or leaving `node_id`.

    links: GMNS rows (dicts) with link_id, from_node_id, to_node_id, geometry, lanes,
           allowed_uses and from_biway. to_pixel(lon, lat) -> source pixel.

    Distances are measured from the node along each leg's axis, not along the link
    polyline: after consolidation a link may start at a former sub-node several
    metres from the node and reach its carriageway through a diagonal stub.
    """
    p = {**DEFAULTS, **(policy or {})}
    side = 1 if p["driving_side"] == "right" else -1
    node = to_pixel(*node_lonlat)
    legs = []
    for link in links:
        inbound, outbound = _int(link["to_node_id"]) == node_id, _int(link["from_node_id"]) == node_id
        if inbound == outbound or not _motor(link):
            continue
        points = [to_pixel(lon, lat) for lon, lat in parse_linestring(link["geometry"])]
        if inbound:
            points.reverse()  # walk away from the junction in both cases
        shared = shares_axis(link, links)
        lanes = _lanes(link)
        probe = _along(points, p["axis_probe_m"][0] / mpp)
        unit = _unit(probe, _along(points, p["axis_probe_m"][1] / mpp))
        offset = abs((probe[0] - node[0]) * unit[1] - (probe[1] - node[1]) * unit[0]) * mpp
        # Reach of this road across the junction when it is the crossing road: from the node to its
        # axis, then one direction's lanes (shared axis) or half of this carriageway (separate links).
        half = offset + lanes * p["lane_width_prior_m"] * (1 if shared else .5)
        legs.append({"link": link, "inbound": inbound, "points": points, "shared": shared, "lanes": lanes,
                     "unit": unit, "axis": _heading((0, 0), unit) % 180, "half_width_m": half, "axis_offset_m": offset})
    require(legs, f"No motor links at node {node_id}")
    # The direction a link leaves the junction, not the bearing to a point on it: the two carriageways
    # of a divided road sit either side of the node yet leave it in parallel.
    arms = road_arms([_heading((0, 0), leg["unit"]) for leg in legs])
    specs = []
    for leg, (arm, arm_bearing) in zip(legs, arms):
        crossing = [o["half_width_m"] for o in legs
                    if 45 <= abs((o["axis"] - leg["axis"] + 90) % 180 - 90)]
        near = min(p["max_near_m"], max(p["min_near_m"], max(crossing, default=0) + p["crosswalk_margin_m"]))
        far = near + p["section_length_m"]
        a, _ = _at_projection(leg["points"], node, leg["unit"], near / mpp)
        b, beyond = _at_projection(leg["points"], node, leg["unit"], far / mpp)
        heading = _heading(b, a) if leg["inbound"] else _heading(a, b)
        carriageway = leg["lanes"] * p["lane_width_prior_m"]
        shift = side * carriageway / 2 / mpp if leg["shared"] else 0.0
        h = math.radians(heading)
        center = ((a[0] + b[0]) / 2 + shift * math.cos(h), (a[1] + b[1]) / 2 + shift * math.sin(h))
        width = 2 * math.ceil((carriageway / 2 + p["lateral_margin_m"]) / mpp)
        pad = round(p["end_pad_m"] / mpp)
        height = 2 * math.ceil((p["section_length_m"] / 2 + p["end_pad_m"]) / mpp)
        frame = Frame((round(center[0], 2), round(center[1], 2)), round(heading, 3), width, height)
        count = p["station_count"]
        stations = [round(pad + (height - 2 * pad) * i / (count - 1), 1) for i in range(count)]
        inside = all(0 <= x <= image_size[0] and 0 <= y <= image_size[1] for x, y in frame.corners())
        kind = "inbound_stopbar" if leg["inbound"] else "outbound_receiving"
        specs.append({"link_id": _int(leg["link"]["link_id"]), "kind": kind, "heading_deg": round(heading, 3),
                      "arm": arm, "arm_bearing_deg": arm_bearing,
                      "frame": frame, "stations": stations, "ground_mpp": mpp, "usable": inside,
                      "unusable_reason": None if inside else "crop_window_outside_satellite_image",
                      "reference_kind": "estimated_junction_envelope", "extrapolated_beyond_link": beyond,
                      # for the anchoring step: where the junction is and which way the leg leaves it
                      "node_px": [round(node[0], 2), round(node[1], 2)], "away_unit": [round(v, 6) for v in leg["unit"]],
                      "carriageway_m": round(carriageway, 1), "shared_axis": leg["shared"],
                      "link_px": [[round(x, 1), round(y, 1)] for x, y in leg["points"]],
                      "centerline_reference": "shared_road_axis" if leg["shared"] else "directed_carriageway",
                      "crop_prior": {"lanes": leg["lanes"], "near_m": round(near, 1), "far_m": round(far, 1),
                                     "axis_offset_m": round(leg["axis_offset_m"], 1),
                                     "role": "crop_window_only_not_sent_to_model"}})
    # Four-point compass names; legs that would share one get eight-point names, then link IDs.
    prefix = {"inbound_stopbar": "in_", "outbound_receiving": "out_"}
    names = [prefix[s["kind"]] + cardinal(s["heading_deg"]) for s in specs]
    for namer in (compass_point, None):
        clashing = [i for i, n in enumerate(names) if names.count(n) > 1]
        for i in clashing:
            spec = specs[i]
            names[i] = prefix[spec["kind"]] + (namer(spec["heading_deg"]) if namer else f"link_{spec['link_id']}")
    for spec, name in zip(specs, names):
        spec["id"], spec["direction"] = name, name.split("_", 1)[1]
    return sorted(specs, key=lambda s: s["id"])
