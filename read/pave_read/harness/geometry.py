"""Arm windows and lane regions for the lean Read pipeline.

One travel-up window per road arm holds both carriageways: traffic entering the junction on that arm moves UP, the
junction lies beyond the top edge, and the leaving carriageway runs down the other side of the road. Only the
network's link geometry places a window (the arm's direction, where its link or links run, how far apart the two
directed links of a divided road lie). Lane counts and widths from the network are not used: the window is a fixed
corridor around the road axis, so its size says nothing about how many lanes the road has.
"""
from __future__ import annotations

import math

from ..autoloop.frames import _along, _heading, _int, _motor, _unit, cardinal, parse_linestring, road_arms, shares_axis
from ..autoloop.strips import Frame
from ..hybrid.common import require

DEFAULTS = {"corridor_half_width_m": 26.0, "near_m": 0.0, "far_m": 78.0,
            "stations_m": [10.0, 17.5, 25.0, 32.5, 40.0, 50.0, 60.0, 72.0], "axis_probe_m": (25.0, 55.0),
            "driving_side": "right"}
MOTOR = ("motor_vehicle_lane", "shared_rail_motor_lane", "shared_bicycle_motor_lane")


def _legs(node_id, node_px, links, to_pixel, mpp, probe_m):
    legs = []
    for link in links:
        inbound, outbound = _int(link["to_node_id"]) == node_id, _int(link["from_node_id"]) == node_id
        if inbound == outbound or not _motor(link):
            continue
        points = [to_pixel(lon, lat) for lon, lat in parse_linestring(link["geometry"])]
        if inbound:
            points.reverse()  # walk away from the junction in both cases
        probe = _along(points, probe_m[0] / mpp)
        unit = _unit(probe, _along(points, probe_m[1] / mpp))
        legs.append({"link_id": _int(link["link_id"]), "name": link.get("name"), "inbound": inbound, "unit": unit,
                     "probe": probe, "shared": shares_axis(link, links)})
    require(legs, f"No motor links at node {node_id}")
    return legs


def arm_windows(node_id, node_lonlat, links, to_pixel, mpp, image_size, policy=None):
    """One window per road arm: its frame, station rows, legs and section names."""
    p = {**DEFAULTS, **(policy or {})}
    node = to_pixel(*node_lonlat)
    legs = _legs(node_id, node, links, to_pixel, mpp, p["axis_probe_m"])
    names = road_arms([_heading((0, 0), leg["unit"]) for leg in legs])
    arms = {}
    for leg, (arm, bearing) in zip(legs, names):
        arms.setdefault(arm, {"arm": arm, "bearing_deg": bearing, "legs": []})["legs"].append(leg)
    windows = []
    for arm in arms.values():
        ux = sum(l["unit"][0] for l in arm["legs"]); uy = sum(l["unit"][1] for l in arm["legs"])
        size = math.hypot(ux, uy) or 1.0
        unit = (ux / size, uy / size)
        heading = _heading((0, 0), (-unit[0], -unit[1]))  # traffic entering on this arm moves toward the node
        h = math.radians(heading)
        right = (math.cos(h), math.sin(h))  # driver-right of entering traffic, in source pixels
        lateral = {l["inbound"]: (l["probe"][0] - node[0]) * right[0] + (l["probe"][1] - node[1]) * right[1]
                   for l in arm["legs"]}
        centre_offset = sum(lateral.values()) / len(lateral)
        separation = abs(lateral.get(True, centre_offset) - lateral.get(False, centre_offset))
        width = 2 * math.ceil((separation / 2 + p["corridor_half_width_m"] / mpp))
        near, far = p["near_m"], p["far_m"]
        # Shorten the window until it lies inside the image (the satellite covers a fixed radius around the node).
        while True:
            mid = (near + far) / 2 / mpp
            centre = (node[0] + unit[0] * mid + right[0] * centre_offset, node[1] + unit[1] * mid + right[1] * centre_offset)
            frame = Frame((round(centre[0], 2), round(centre[1], 2)), round(heading, 3), width, round((far - near) / mpp))
            if all(0 <= x <= image_size[0] and 0 <= y <= image_size[1] for x, y in frame.corners()) or far - near < 30:
                break
            far -= 2.0
        stations = [round((s - near) / mpp, 1) for s in p["stations_m"] if near < s < far]
        require(len(stations) >= 3, f"Arm {arm['arm']} window too short for stations")
        sections = {}
        for leg in arm["legs"]:
            travel = heading if leg["inbound"] else (heading + 180) % 360
            sections["inbound" if leg["inbound"] else "outbound"] = {
                "section_id": ("in_" if leg["inbound"] else "out_") + cardinal(travel),
                "direction": cardinal(travel), "link_id": leg["link_id"], "road_name": leg["name"],
                "travel_heading_deg": round(travel, 1), "shared_axis": leg["shared"]}
        road = next((l["name"] for l in arm["legs"] if l.get("name")), None)
        windows.append({"arm": arm["arm"], "bearing_deg": arm["bearing_deg"], "road_name": road, "frame": frame,
                        "stations": stations, "stations_m": [s for s in p["stations_m"] if near < s < far], "mpp": mpp,
                        "near_m": near, "far_m": far, "sections": sections, "driving_side": p["driving_side"],
                        "node_px": [round(node[0], 2), round(node[1], 2)]})
    return sorted(windows, key=lambda w: w["bearing_deg"])


def _x_at(xs, stations, y):
    """x of a boundary at row y, interpolated between its defined stations (None outside them)."""
    pts = [(s, x) for s, x in zip(stations, xs) if x is not None]
    if not pts:
        return None
    if y <= pts[0][0]:
        return pts[0][1] if len(pts) == 1 else pts[0][1] + (pts[1][1] - pts[0][1]) * (y - pts[0][0]) / (pts[1][0] - pts[0][0])
    for (s0, x0), (s1, x1) in zip(pts, pts[1:]):
        if s0 <= y <= s1:
            return x0 + (x1 - x0) * (y - s0) / (s1 - s0)
    return None


def strip_rows(left, right, stations, start_y):
    """Station rows from the first one past start_y (downstream of the stop bar or exit start) where both
    boundaries are drawn, while they stay drawn: the strip's extent along the road."""
    rows = []
    for i, y in enumerate(stations):
        if y <= start_y:
            continue
        if left[i] is None or right[i] is None:
            if rows:
                break
            continue
        rows.append(i)
    return rows


def stop_line_ends(trace):
    """The stop line as (y where it meets the road-centre side of the inbound lanes, y where it meets the curb side), or
    None. A trace that gives one y (stop_line_y) has both ends at it; a trace that spells the curb end y_kerb is read
    as y_curb."""
    s = trace.get("stop_line")
    y_side = (s.get("y_curb") if s.get("y_curb") is not None else s.get("y_kerb")) if isinstance(s, dict) else None
    if isinstance(s, dict) and s.get("y_centre") is not None and y_side is not None:
        return float(s["y_centre"]), float(y_side)
    if trace.get("stop_line_y") is not None:
        return float(trace["stop_line_y"]), float(trace["stop_line_y"])
    return None


def _first_x(xs, stations, after_y):
    """x of a boundary at the first station past after_y where it is drawn (else at any station where it is)."""
    for x, y in zip(xs, stations):
        if x is not None and y > after_y:
            return x
    return next((x for x in xs if x is not None), None)


def stop_line_at(trace, stations):
    """Function x -> y of the stop line: straight between its two ends, which lie on the outermost boundaries of the
    inbound carriageway (road-centre side = smaller x, curb side = larger x). None without a stop line."""
    ends = stop_line_ends(trace)
    if ends is None:
        return None
    y_centre, y_curb = ends
    own = [k for k, s in enumerate(trace["strips"]) if s["carriageway"] == "inbound"]
    edges = sorted({k for k in own} | {k + 1 for k in own})
    xs = [x for x in (_first_x(trace["boundaries"][k]["xs"], stations, min(ends)) for k in edges) if x is not None]
    if not xs or max(xs) - min(xs) < 1e-6 or y_centre == y_curb:
        return lambda x: (y_centre + y_curb) / 2
    lo, hi = min(xs), max(xs)
    return lambda x: y_centre + (y_curb - y_centre) * min(max((x - lo) / (hi - lo), 0.0), 1.0)


def regions(trace, window):
    """Lane regions of both carriageways from a trace answer.

    Inbound lanes are what reaches the stop line. The stop line may cross the road at an angle (a skewed junction), so
    each boundary is looked at on the first station past the stop line at that boundary's own position. A boundary not
    drawn there (a line that only exists upstream) joins its two neighbouring strips into one lane, and a lane is
    measured on the first station where both its boundaries have passed the stop line. Outbound lanes are read where
    most of the exit's lines are drawn, on a station past the exit start. Each lane's region runs from there along the
    road while both of its boundaries are drawn. Inbound regions are numbered from the driver's left (the road-centre
    side, smallest x); outbound traffic moves down the image, so its driver's left is the largest x."""
    stations, frame = window["stations"], window["frame"]
    xs = [b["xs"] for b in trace["boundaries"]]
    out = {"inbound": [], "outbound": []}
    for side in ("inbound", "outbound"):
        if side == "inbound":
            at = stop_line_at(trace, stations)
            if at is None:
                continue
            ends = stop_line_ends(trace)
            gate_of, start_of = {}, {}
            for k, b in enumerate(xs):
                x = _first_x(b, stations, min(ends))
                start_of[k] = at(x) if x is not None else max(ends)
                after = [i for i, y in enumerate(stations) if y > start_of[k]]
                if after:
                    gate_of[k] = after[0]
            present = [k for k in gate_of if xs[k][gate_of[k]] is not None]
        else:
            start = trace.get("exit_start_y")
            if start is None:
                continue
            after = [k for k, y in enumerate(stations) if y > start]
            if not after:
                continue
            # Exit lane lines often start some way past the crosswalk: read the exit where most of its lines are drawn.
            own = [k for k, s in enumerate(trace["strips"]) if s["carriageway"] == side]
            edges = sorted({k for k in own} | {k + 1 for k in own})
            gate = max(after, key=lambda row: (sum(xs[k][row] is not None for k in edges), -row))
            present = [k for k, b in enumerate(xs) if b[gate] is not None]
            gate_of = {k: gate for k in present}
            start_of = {k: start for k in present}
        for a, b in zip(present, present[1:]):
            between = [trace["strips"][i]["carriageway"] for i in range(a, b)]
            if any(c not in (side, "none") for c in between):
                continue  # a span that would join the two carriageways is a missing boundary, not a lane
            parts = [(i, trace["strips"][i]) for i in range(a, b) if trace["strips"][i]["carriageway"] == side]
            if not parts:
                continue
            # Strips joined by a boundary missing at the gate form one lane, a motor lane if any part is one: shared with a
            # streetcar if a part is, else plain if a part is, else shared with bicycles.
            kinds = [s["surface_type"] for _, s in parts]
            surface = next((t for t in (MOTOR[1], MOTOR[0], MOTOR[2]) if t in kinds), kinds[0])
            # Measure where both boundaries have passed the stop line (or the exit gate) and are drawn.
            row = next((r for r in range(max(gate_of[a], gate_of[b]), len(stations))
                        if xs[a][r] is not None and xs[b][r] is not None), None)
            if row is None:
                continue
            start = max(start_of[a], start_of[b]) if side == "inbound" else start_of[a]
            rows = strip_rows(xs[a], xs[b], stations, min(start, stations[row] - 1e-6))
            top_a, top_b = max(start_of[a], 0.0), max(start_of[b], 0.0)
            left_top, right_top = _x_at(xs[a], stations, top_a), _x_at(xs[b], stations, top_b)
            local = ([(left_top, top_a)] if left_top is not None else []) + [(xs[a][k], stations[k]) for k in rows]
            local += [(xs[b][k], stations[k]) for k in reversed(rows)] + ([(right_top, top_b)] if right_top is not None else [])
            out[side].append({"strip": parts[0][0] if len(parts) == 1 else [i for i, _ in parts], "boundaries": [a, b],
                              "surface_type": surface, "confidence": min((s.get("confidence") or "low" for _, s in parts),
                                                                         key=["low", "medium", "high"].index),
                              "evidence": " | ".join(s.get("evidence", "") for _, s in parts), "at_gate": True, "rows": rows,
                              "width_m": round((xs[b][row] - xs[a][row]) * window["mpp"], 2),
                              "x_mid": (xs[a][row] + xs[b][row]) / 2,
                              "polygon_px": [[round(v, 1) for v in frame.to_source(x, y)] for x, y in local]})
    for side, order in (("inbound", False), ("outbound", True)):
        out[side].sort(key=lambda r: r["x_mid"] if r["x_mid"] is not None else 0, reverse=order)
        motor = [r for r in out[side] if r["surface_type"] in MOTOR and r["at_gate"] and r["polygon_px"]]
        for k, region in enumerate(motor):
            region["region_id"] = f"r{k + 1}"
            region["motor_lane_index"] = k + 1
    return out


def motor_regions(found, side):
    return [r for r in found[side] if r.get("region_id")]


def trace_issues(found, sections, low_m=2.5, high_m=5.0):
    """Measured issues plus the motor lanes the trace itself marked low-confidence: each is worth a second look."""
    issues = width_issues(found, sections, low_m, high_m)
    for side in ("inbound", "outbound"):
        for r in motor_regions(found, side):
            if r.get("confidence") == "low":
                issues.append(f"{sections[side]['section_id']} {r['region_id']} (between boundaries b{r['boundaries'][0] + 1} and"
                              f" b{r['boundaries'][1] + 1}) is a motor lane you marked low-confidence: check it in the street views.")
    return issues


def width_issues(found, sections, low_m=2.5, high_m=5.0):
    """Measured facts worth a second look: a motor lane at its gate narrower or wider than one lane usually is.
    The model decides what the strip really is; nothing here changes the trace."""
    issues = []
    for side in ("inbound", "outbound"):
        for r in motor_regions(found, side):
            if r["width_m"] is None:
                continue
            where = f"{sections[side]['section_id']} {r['region_id']} (between boundaries b{r['boundaries'][0] + 1} and b{r['boundaries'][1] + 1})"
            if r["width_m"] > high_m:
                issues.append(f"{where} is {r['width_m']:.1f} m wide at its first station, wider than one lane usually is: a lane line may be"
                              " hidden by shadow, wear or vehicles, or the strip may hold a lane and a bicycle lane, buffer or parking.")
            elif r["width_m"] < low_m:
                issues.append(f"{where} is {r['width_m']:.1f} m wide at its first station, narrower than a motor lane usually is: it may be"
                              " a bicycle lane, buffer or gutter, a boundary may be misplaced, or a line that is not a lane line (such as a rail"
                              " of a streetcar track) may have been traced as one.")
    regs = found.get("inbound", [])
    for i, r in enumerate(regs):
        # a bicycle strip as wide as a lane between two traced lanes
        if (r.get("surface_type") == "bicycle_lane" and not r.get("region_id") and (r.get("width_m") or 0) >= low_m
                and 0 < i < len(regs) - 1 and regs[i - 1].get("region_id") and regs[i + 1].get("region_id")):
            issues.append(f"{sections['inbound']['section_id']}: the strip traced as a bicycle lane between {regs[i - 1]['region_id']} and"
                          f" {regs[i + 1]['region_id']} is {r['width_m']:.1f} m wide at its first station, as wide as a motor lane: it may be a"
                          " motor lane (for example a turn lane), or hold one beside a narrower bicycle lane.")
    return issues
