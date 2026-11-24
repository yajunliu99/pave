"""Lane-strip geometry in a travel-up section frame, and the edits a review may apply.

A section has K horizontal stations (local y, top to bottom) and N longitudinal
boundaries, each holding one x per station, ordered driver-left to driver-right.
The N-1 strips between neighbouring boundaries are neutral candidate regions;
nothing here decides which of them are motor-vehicle lanes.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict

from ..hybrid.common import ValidationError, digest, require

MIN_GAP = 4.0
EDIT_OPS = ("move_edge", "split_region", "merge_regions", "add_region", "drop_region")


@dataclass(frozen=True)
class Frame:
    """Crop window in a source image, rotated so that travel points up.

    `heading_deg` is the travel direction in the source image, clockwise from
    image-up. Coordinates are continuous, top-left origin, in both frames.
    """
    center: tuple
    heading_deg: float
    width: int
    height: int

    def _axes(self):
        h = math.radians(self.heading_deg)
        return (math.cos(h), math.sin(h)), (math.sin(h), -math.cos(h))  # driver-right, travel

    def to_source(self, x, y):
        (rx, ry), (ux, uy) = self._axes()
        dx, dy = x - self.width / 2, y - self.height / 2
        return self.center[0] + dx * rx - dy * ux, self.center[1] + dx * ry - dy * uy

    def to_local(self, x, y):
        (rx, ry), (ux, uy) = self._axes()
        dx, dy = x - self.center[0], y - self.center[1]
        return dx * rx + dy * ry + self.width / 2, -(dx * ux + dy * uy) + self.height / 2

    def affine(self):
        """Coefficients for PIL's AFFINE transform (output pixel -> source pixel)."""
        (rx, ry), (ux, uy) = self._axes()
        ox, oy = self.to_source(0, 0)
        return tuple(round(v, 9) for v in (rx, -ux, ox, ry, -uy, oy))

    def corners(self):
        return [self.to_source(x, y) for x, y in ((0, 0), (self.width, 0), (self.width, self.height), (0, self.height))]

    def expanded(self, pad):
        return Frame(self.center, self.heading_deg, self.width + 2 * pad, self.height)

    def pixels_preserved(self):
        return self.heading_deg % 90 == 0

    def serialize(self):
        return asdict(self)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def edge_meta(strips):
    return strips.get("meta") or [{} for _ in strips["edges"]]


def validate_strips(strips, width, min_gap=MIN_GAP):
    stations, edges = strips["stations"], strips["edges"]
    require(len(stations) >= 2 and all(a < b for a, b in zip(stations, stations[1:])), "Stations must run down the image")
    require(len(edges) >= 2, "At least two boundaries are needed to form a region")
    for xs in edges:
        require(len(xs) == len(stations) and all(is_number(x) for x in xs), "Each boundary needs one x per station")
        require(all(0 <= x <= width for x in xs), "A boundary leaves the crop")
    for left, right in zip(edges, edges[1:]):
        require(all(b - a >= min_gap for a, b in zip(left, right)), "Boundaries cross or touch")
    return strips


def region_widths(strips):
    return [[b - a for a, b in zip(left, right)] for left, right in zip(strips["edges"], strips["edges"][1:])]


def region_polygon(strips, index):
    """Local polygon of region `index` (0-based): down the left boundary, back up the right."""
    stations = strips["stations"]
    left, right = strips["edges"][index], strips["edges"][index + 1]
    return list(zip(left, stations)) + list(zip(reversed(right), reversed(stations)))


def section_regions(strips, frame, kind):
    """Regions in the hybrid config schema: source-pixel polygons, gate at the junction end."""
    stations = strips["stations"]
    end = 0 if kind == "inbound_stopbar" else -1  # travel is up, so an inbound stop bar is the top station
    regions = []
    for index, (left, right) in enumerate(zip(strips["edges"], strips["edges"][1:])):
        gate = frame.to_source((left[end] + right[end]) / 2, stations[end])
        regions.append({"id": f"r{index + 1}",
                        "polygon": [[round(v, 1) for v in frame.to_source(x, y)] for x, y in region_polygon(strips, index)],
                        "gate_point": [round(v, 1) for v in gate]})
    return regions


def signature(strips):
    return digest([[round(x) for x in xs] for xs in strips["edges"]])


def change_px(before, after):
    """Largest boundary displacement, or None when the number of boundaries changed."""
    if len(before["edges"]) != len(after["edges"]):
        return None
    return max(abs(a - b) for xs, ys in zip(before["edges"], after["edges"]) for a, b in zip(xs, ys))


def _ref(value, prefix, count):
    require(isinstance(value, str) and value[:1] == prefix and value[1:].isdigit(), f"Expected an ID like {prefix}1")
    index = int(value[1:])
    require(1 <= index <= count, f"{value} is not in the reviewed geometry")
    return index


def _xs(value, stations):
    require(isinstance(value, list) and len(value) == stations and all(is_number(x) for x in value),
            "xs needs one number per station")
    return [float(x) for x in value]


def _position(work, token):
    for i, edge in enumerate(work):
        if edge["token"] == token:
            return i
    raise ValidationError("A boundary this edit refers to was removed by an earlier edit")


def _at(xs):
    return f"x≈{sum(xs) / len(xs):.0f}"


def _apply_one(work, edit, count, stations, fresh, origin):
    """Apply one edit in place and describe its effect by position, not by ID."""
    require(isinstance(edit, dict) and edit.get("op") in EDIT_OPS, "Unknown edit operation")
    op = edit["op"]
    new = {"token": fresh, "meta": {"kind": "unknown", "origin": origin, "evidence": edit.get("evidence", "")}}
    if op == "move_edge":
        edge = work[_position(work, _ref(edit.get("edge"), "e", count) - 1)]
        before = _at(edge["xs"])
        if edit.get("xs") is not None:
            edge["xs"] = _xs(edit["xs"], stations)
        else:
            dx = edit.get("dx")
            require(is_number(dx) or isinstance(dx, list), "move_edge needs xs or dx")
            shift = [float(dx)] * stations if is_number(dx) else _xs(dx, stations)
            edge["xs"] = [x + d for x, d in zip(edge["xs"], shift)]
        edge["meta"] = {**edge["meta"], "revised_by": origin}
        return f"moved the boundary at {before} to {_at(edge['xs'])}"
    elif op == "split_region":
        k = _ref(edit.get("region"), "r", count - 1)
        i, j = _position(work, k - 1), _position(work, k)
        require(j == i + 1, "That region was already changed by an earlier edit")
        xs = _xs(edit["xs"], stations) if edit.get("xs") is not None else [(a + b) / 2 for a, b in zip(work[i]["xs"], work[j]["xs"])]
        work.insert(j, {**new, "xs": xs})
        return f"added a boundary at {_at(xs)}"
    elif op == "merge_regions":
        ids = edit.get("regions")
        require(isinstance(ids, list) and len(ids) == 2, "merge_regions needs two region IDs")
        a, b = sorted(_ref(v, "r", count - 1) for v in ids)
        require(b == a + 1, "Only neighbouring regions can be merged")
        i = _position(work, a)
        require(0 < i < len(work) - 1, "That boundary is now an outer limit and cannot be merged away")
        return f"removed the boundary at {_at(work.pop(i)['xs'])}"
    elif op == "add_region":
        require(edit.get("side") in ("left", "right"), "add_region needs side left or right")
        entry = {**new, "xs": _xs(edit.get("xs"), stations)}
        work.insert(0 if edit["side"] == "left" else len(work), entry)
        return f"added an outer boundary on the {edit['side']} at {_at(entry['xs'])}"
    else:
        k = _ref(edit.get("region"), "r", count - 1)
        require(k in (1, count - 1), "Only an outermost region can be dropped; merge interior regions instead")
        i = _position(work, 0 if k == 1 else count - 1)
        require(i in (0, len(work) - 1), "That boundary is no longer an outer limit")
        return f"removed the outer boundary at {_at(work.pop(i)['xs'])}"


def apply_edits(strips, edits, width, min_gap=MIN_GAP, origin="review"):
    """Apply edits one at a time. An edit that cannot be applied is rejected, never guessed.

    IDs in every edit refer to the geometry as supplied, so several edits from one
    review stay valid while earlier ones insert or remove boundaries.
    """
    stations = len(strips["stations"])
    count = len(strips["edges"])
    work = [{"token": i, "xs": [float(x) for x in xs], "meta": copy.deepcopy(meta)}
            for i, (xs, meta) in enumerate(zip(strips["edges"], edge_meta(strips)))]
    applied, rejected = [], []
    for n, edit in enumerate(edits):
        trial = copy.deepcopy(work)
        try:
            effect = _apply_one(trial, edit, count, stations, count + n, origin)
            validate_strips({"stations": strips["stations"], "edges": [e["xs"] for e in trial]}, width, min_gap)
        except ValidationError as error:
            rejected.append({"edit": edit, "reason": str(error)})
            continue
        work = trial
        applied.append({**edit, "effect": effect})
    result = {"stations": list(strips["stations"]), "edges": [e["xs"] for e in work], "meta": [e["meta"] for e in work]}
    return result, applied, rejected
