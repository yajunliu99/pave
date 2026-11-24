"""Anchor each section's crop window on what the satellite image shows, before tracing.

The network places a window from estimates: the junction envelope from the crossing roads' lane counts and the
road direction from the link polyline. Both can be off by many metres (wide junctions, set-back stop bars,
curves, consolidated links that reach the road through a diagonal stub). One model call per section looks at a
long, wide probe strip around the estimate and reports where the stop bar (or, for an exit, the start of the
receiving lanes) is and where the target carriageway runs at two cross-sections. Code turns those points into
the window: it starts just beyond that line and follows the carriageway's own direction. When the model cannot
see the line, the network window is kept and the reason recorded.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from ..hybrid.common import require
from ..hybrid.geometry import font
from . import render
from .strips import Frame, is_number

DEFAULTS = {"probe_length_m": 65.0, "probe_back_m": 25.0, "probe_margin_m": 14.0, "probe_min_width_m": 40.0, "second_section_m": 25.0,
            "stop_offset_m": 1.0, "min_carriageway_m": 2.5, "max_carriageway_m": 36.0, "arm_tolerance_m": 8.0}
STOP_KINDS = ("stop_bar", "crosswalk_edge", "junction_edge", "not_visible")
CONFIDENCE = ("high", "medium", "low")
MARGIN = 58
GRID = (255, 255, 255, 70)
NODE, LINK = "#ff3df2", "#ffd400"


def heading_of(vx, vy):
    """Degrees clockwise from image-up of a vector in image pixels (y down)."""
    return math.degrees(math.atan2(vx, -vy)) % 360


def inbound(spec):
    return spec["kind"] == "inbound_stopbar"


def probe_frame(spec, policy=None):
    """A long, wide travel-up window along the leg's direction, from past the network junction centre outward.

    It starts `probe_back_m` beyond the network's junction centre, because that point can lie well off the painted
    junction (a consolidated node, an offset side street): the whole junction then stays in view."""
    p = {**DEFAULTS, **(policy or {})}
    mpp = spec["ground_mpp"]
    ux, uy = spec["away_unit"]
    back, ahead = p["probe_back_m"] / mpp, p["probe_length_m"] / mpp
    length = back + ahead
    width = max(p["probe_min_width_m"], spec["carriageway_m"] + 2 * p["probe_margin_m"]) / mpp
    nx, ny = spec["node_px"]
    centre = (nx + ux * (ahead - back) / 2, ny + uy * (ahead - back) / 2)
    heading = heading_of(-ux, -uy) if inbound(spec) else heading_of(ux, uy)
    return Frame((round(centre[0], 2), round(centre[1], 2)), round(heading, 3), 2 * math.ceil(width / 2), 2 * math.ceil(length / 2))


def gridded(raw, probe, spec, step=100):
    """Probe crop with x/y rulers in crop pixels on all sides, a faint grid, the network junction centre and link."""
    body = raw.convert("RGBA")
    layer = Image.new("RGBA", body.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(layer)
    for x in range(step, body.width, step):
        for y in range(0, body.height, 7):
            g.point((x, y), fill=GRID)
    for y in range(step, body.height, step):
        for x in range(0, body.width, 7):
            g.point((x, y), fill=GRID)
    link = [probe.to_local(x, y) for x, y in spec.get("link_px", [])]
    render._dashed(g, link, LINK, on=10, off=10, width=1)
    nx, ny = probe.to_local(*spec["node_px"])
    nx, ny = min(max(nx, 6), body.width - 6), min(max(ny, 6), body.height - 6)
    g.line((nx - 9, ny, nx + 9, ny), fill=NODE, width=3)
    g.line((nx, ny - 9, nx, ny + 9), fill=NODE, width=3)
    body = Image.alpha_composite(body, layer).convert("RGB")
    canvas = Image.new("RGB", (body.width + 2 * MARGIN, body.height + 2 * MARGIN), render.PAPER)
    canvas.paste(body, (MARGIN, MARGIN))
    d = ImageDraw.Draw(canvas)
    for x in range(0, body.width + 1, 20):
        px, major = MARGIN + x, x % step == 0
        for base, sign in ((MARGIN, -1), (MARGIN + body.height, 1)):
            d.line((px, base, px, base + sign * (10 if major else 4)), fill="white")
        if major:
            d.text((px, 4), str(x), font=font(16), fill="white", anchor="ma")
            d.text((px, MARGIN + body.height + 14), str(x), font=font(16), fill="white", anchor="ma")
    for y in range(0, body.height + 1, 20):
        py, major = MARGIN + y, y % step == 0
        for base, sign in ((MARGIN, -1), (MARGIN + body.width, 1)):
            d.line((base, py, base + sign * (10 if major else 4), py), fill="white")
        if major:
            d.text((MARGIN - 12, py), str(y), font=font(16), fill="white", anchor="rm")
            d.text((MARGIN + body.width + 12, py), str(y), font=font(16), fill="white", anchor="lm")
    return canvas


def anchor_prompt(spec, probe, shaded=False):
    mpp = spec["ground_mpp"]
    second = round(DEFAULTS["second_section_m"] / mpp)
    side = spec.get("driving_side", "right")
    other = "left" if side == "right" else "right"
    if inbound(spec):
        where = ("traffic on the target approach drives UP toward the junction, which lies toward the TOP. The magenta cross is where "
                 "the road network puts the junction centre; it can be 20 m or more off the painted junction, so the crop runs about "
                 "25 m past it and the whole junction should be in view.")
        target = (f"the approach carriageway: the lanes on which traffic moves UP toward the junction. Traffic keeps to the {side}, "
                  f"so on a two-way road it lies on the {side} of the centre line and the opposing lanes, moving down, are on its {other}.")
        line = ("stop_line: the y of the stop bar across the target lanes, the transverse white line where vehicles wait. If no stop bar "
                "is painted, give the upstream edge of the crosswalk (kind crosswalk_edge), or else the line where the target lanes "
                "end at the junction (kind junction_edge).")
        rows = f"one at the stop line and one about 25 m upstream of it, i.e. about {second} px BELOW it"
    else:
        where = ("traffic on the target receiving carriageway drives UP, away from the junction, which lies toward the BOTTOM. The "
                 "magenta cross is where the road network puts the junction centre; it can be 20 m or more off the painted junction, so "
                 "the crop runs about 25 m past it and the whole junction should be in view.")
        target = (f"the receiving carriageway: the lanes on which traffic leaves the junction moving UP. Traffic keeps to the {side}, "
                  f"so on a two-way road it lies on the {side} of the centre line and the lanes moving down toward the junction are on its {other}.")
        line = ("stop_line: the y where the receiving lanes begin, i.e. the downstream edge (the side away from the junction, toward "
                "the top) of the crosswalk across them (kind crosswalk_edge); with no crosswalk, the line where the lanes leave the "
                "junction (kind junction_edge). It lies above the junction area, never inside it or on the junction's far side; a stop bar "
                "of the opposing traffic beside them is not this line.")
        rows = f"one at that line and one about 25 m further from the junction, i.e. about {second} px ABOVE it"
    shade = (" Shaded parts of the image have been lightened, so their colours are not true." if shaded else "")
    return f"""Locate a road section on a satellite crop before its lanes are traced.
The crop is rotated so that {where} The dashed yellow line is the network's centre line for this road; it can be offset from the painted road or cut across a corner. Rulers on all sides and the faint dotted grid give x and y in crop pixels (1 px is about {mpp:.3f} m).{shade}
Target: {target}
First find the junction itself: the area where the crossing road meets this road, usually with a crosswalk on each side of it. Then report:
1. {line} Give null with kind not_visible if none of these can be seen.
2. cross_sections: two rows across the target carriageway, {rows}. For each give the x of its left limit (centre line, median edge or left kerb) and its right limit (kerb or pavement edge). Include every lane of the target direction: turn bays, a lane shared with rail, and a right-turn lane lying between a bicycle lane and the kerb. Leave out the opposing lanes, sidewalks and parking aisles off the road.
Read positions from the image itself, not from the dashed network line.
Return JSON {{"stop_line":{{"y":number or null,"kind":"{'|'.join(STOP_KINDS)}","evidence":"what marks it"}},"cross_sections":[{{"y":number,"left_x":number,"right_x":number,"evidence":"what bounds the carriageway here"}},{{"y":number,"left_x":number,"right_x":number,"evidence":"..."}}],"confidence":"{'|'.join(CONFIDENCE)}","notes":"anything that limits the reading"}}.
"""


def validate_anchor(value, spec, probe, policy=None):
    p = {**DEFAULTS, **(policy or {})}
    mpp = spec["ground_mpp"]
    require(isinstance(value, dict), "Return one JSON object")
    stop = value.get("stop_line")
    require(isinstance(stop, dict) and stop.get("kind") in STOP_KINDS, f"stop_line.kind must be one of {', '.join(STOP_KINDS)}")
    y = stop.get("y")
    require((y is None) == (stop["kind"] == "not_visible"), "Give stop_line.y as a number unless kind is not_visible, and null only then")
    require(y is None or (is_number(y) and 0 <= y <= probe.height), f"stop_line.y must lie in 0..{probe.height}")
    require(isinstance(stop.get("evidence"), str), "stop_line.evidence must be text")
    rows = value.get("cross_sections")
    require(isinstance(rows, list) and len(rows) == 2, "Give exactly two cross_sections")
    for r in rows:
        require(isinstance(r, dict) and all(is_number(r.get(k)) for k in ("y", "left_x", "right_x")), "Each cross-section needs y, left_x and right_x numbers")
        require(0 <= r["y"] <= probe.height and 0 <= r["left_x"] < r["right_x"] <= probe.width,
                f"Cross-section points must lie in the crop (x 0..{probe.width}, y 0..{probe.height}) with left_x < right_x")
        width = (r["right_x"] - r["left_x"]) * mpp
        require(p["min_carriageway_m"] <= width <= p["max_carriageway_m"],
                f"A carriageway {width:.1f} m wide is implausible; give the limits of the target direction's lanes only")
        require(isinstance(r.get("evidence"), str), "Each cross-section needs evidence text")
    near, far = rows
    gap = (far["y"] - near["y"]) * (1 if inbound(spec) else -1) * mpp
    require(10 <= gap <= 40, "The second cross-section must lie 10-40 m from the first, "
            + ("below it (upstream)" if inbound(spec) else "above it (away from the junction)"))
    if y is not None:
        require(abs(near["y"] - y) * mpp <= 4, "The first cross-section must lie at the stop line")
    require(value.get("confidence") in CONFIDENCE, "confidence must be high, medium or low")
    return value


def line_distance(spec, probe, value):
    """Distance (m) from the network junction centre, along the leg, of the reported line on the carriageway's centre."""
    if value["stop_line"]["y"] is None:
        return None
    a, b = _centre_line(probe, value)
    near_y, far_y = value["cross_sections"][0]["y"], value["cross_sections"][1]["y"]
    point = a + (b - a) * (value["stop_line"]["y"] - near_y) / (far_y - near_y)
    return _along(spec, point)


def _centre_line(probe, value):
    pts = [(probe.to_source(r["left_x"], r["y"]), probe.to_source(r["right_x"], r["y"])) for r in value["cross_sections"]]
    mids = [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in pts]
    return np.array(mids[0]), np.array(mids[1])


def _along(spec, point):
    return float(np.dot(np.asarray(point) - np.asarray(spec["node_px"]), spec["away_unit"])) * spec["ground_mpp"]


def harmonise(entries, tolerance=None):
    """Start distance and how it was chosen, for every read section: {section id: (metres from the junction centre, how)}.

    On one road arm the approach's stop line and the exit's start both sit at that arm's crosswalk, so their distances
    from the network junction centre agree within `tolerance`. A trusted line is one read at medium or high confidence.
    Two trusted lines that disagree: the higher confidence wins, the approach's stop bar on a tie (it is the clearer
    mark). A section whose own line is untrusted takes its arm partner's trusted line; with neither trusted, two
    low-confidence lines that agree are used; otherwise the network's estimate stays (None)."""
    tolerance = DEFAULTS["arm_tolerance_m"] if tolerance is None else tolerance
    rank = {"high": 2, "medium": 1, "low": 0}
    read = {}
    for spec, probe, value in entries:
        if value is not None:
            read[spec["id"]] = (spec, line_distance(spec, probe, value), value["confidence"])
    arms = {}
    for sid, (spec, _, _) in read.items():
        arms.setdefault(spec.get("arm"), []).append(sid)
    out = {}
    for sids in arms.values():
        trusted = {s: read[s] for s in sids if read[s][1] is not None and read[s][2] != "low"}
        best = max(trusted, key=lambda s: (rank[trusted[s][2]], inbound(trusted[s][0])), default=None)
        for s in sids:
            spec, d, conf = read[s]
            if s in trusted and (best is None or abs(d - trusted[best][1]) <= tolerance or s == best):
                out[s] = (d, "anchored")
            elif best is not None:
                out[s] = (trusted[best][1], "from_arm_partner")
            else:
                others = [read[o][1] for o in sids if o != s and read[o][1] is not None]
                if d is not None and others and abs(d - others[0]) <= tolerance:
                    out[s] = (d, "agreed_low_confidence")
                else:
                    out[s] = (None, "aligned_only")
    return out


def anchored_frame(spec, probe, value, frame_policy, start_m=None, how=None):
    """The section window placed from the model's reading.

    The two cross-sections give the carriageway's own direction and lateral position. The window starts at `start_m`
    metres from the network junction centre along the leg (see `harmonise`), just beyond that line; with no start
    distance it keeps the network's start, carried onto the new centre line ("aligned_only")."""
    mpp = spec["ground_mpp"]
    pts = [(probe.to_source(r["left_x"], r["y"]), probe.to_source(r["right_x"], r["y"])) for r in value["cross_sections"]]
    a, b = _centre_line(probe, value)
    away = (b - a) / np.linalg.norm(b - a)          # along the carriageway, away from the junction
    travel = -away if inbound(spec) else away
    length = frame_policy["section_length_m"]
    if start_m is None and how is None and value["stop_line"]["y"] is not None and value["confidence"] != "low":
        start_m, how = line_distance(spec, probe, value), "anchored"
    if start_m is not None:
        # the point of the centre line that lies start_m from the junction centre along the leg
        rate = float(np.dot(away, spec["away_unit"]))
        start = a + away * (start_m - _along(spec, a)) / mpp / (rate if abs(rate) > .2 else 1.0)
        offset = DEFAULTS["stop_offset_m"]
    else:
        old = spec["frame"]
        pad = frame_policy["end_pad_m"] / mpp
        first = np.array(old.to_source(old.width / 2, pad if inbound(spec) else old.height - pad))  # junction-side station
        start, how, offset = a + away * float(np.dot(first - a, away)), "aligned_only", 0.0
    centre = start + away * (offset + length / 2) / mpp
    carriageway = max(math.dist(l, r) for l, r in pts) * mpp
    width = 2 * math.ceil((carriageway / 2 + frame_policy["lateral_margin_m"]) / mpp)
    height = 2 * math.ceil((length / 2 + frame_policy["end_pad_m"]) / mpp)
    return Frame((round(float(centre[0]), 2), round(float(centre[1]), 2)), round(heading_of(*travel), 3), width, height), {
        "start_px": [round(float(v), 1) for v in start], "carriageway_m": round(carriageway, 1), "placed": how or "anchored",
        "start_m": None if start_m is None else round(start_m, 1)}


def compare(old, new, spec):
    """How far anchoring moved the window: along the road, across it, and in direction."""
    mpp = spec["ground_mpp"]
    turn = (new.heading_deg - old.heading_deg + 180) % 360 - 180
    h = math.radians(new.heading_deg)
    dx, dy = new.center[0] - old.center[0], new.center[1] - old.center[1]
    along = (dx * math.sin(h) - dy * math.cos(h)) * mpp
    across = (dx * math.cos(h) + dy * math.sin(h)) * mpp
    return {"along_m": round(along, 1), "across_m": round(across, 1), "turn_deg": round(turn, 1)}
