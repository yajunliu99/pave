"""Project satellite lane regions into street views, and align the projection to the image.

Projection uses the camera's recorded position, heading and pitch over flat ground at an
assumed height. Those are not calibrated, so a projection can sit a lane or more off the
paint. Alignment splits the work: the model says where the real feature each projected
boundary stands for crosses a few image rows, and a pose fit (sideways offset, heading,
height) turns those readings into a corrected projection. Lane lines repeat every lane
width, so geometry alone locks onto the wrong lane; the strip labels let the model resolve
which line is which.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from ..hybrid.common import ValidationError, read_json, require
from ..hybrid.geometry import font
from .lane_labels import label
from .render import _dashed

NEAR = .8          # metres ahead of the camera; ground closer than this is cut off
HEIGHT = 2.5       # assumed camera height when nothing better is known
FEATURES = ("lane_line", "bicycle_line", "kerb", "median_edge", "rail_edge", "other")
ROW, MARGIN, RULER = "#ff3df2", 52, 30
def _circular_mean(degrees):
    x, y = sum(math.sin(math.radians(d)) for d in degrees), sum(math.cos(math.radians(d)) for d in degrees)
    return math.degrees(math.atan2(x, y)) % 360


def _gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def road_arms(regions, tolerance=35.):
    """Group sections into road arms by the bearing of their regions from the junction centre.

    Works from the regions' local coordinates alone, so any number of arms at any angle is handled.
    An arm's inbound heading is the direction of travel toward the centre.
    """
    points = {}
    for region in regions:
        points.setdefault(region["section_id"], []).extend(region["polygon_local_m"])
    bearing = {sid: math.degrees(math.atan2(*np.mean(np.asarray(p, float), axis=0))) % 360 for sid, p in points.items()}
    arms = []
    for sid in sorted(points, key=bearing.get):
        arm = next((a for a in arms if _gap(bearing[sid], a["bearing"]) <= tolerance), None)
        if arm is None:
            arms.append({"sections": [sid], "bearing": bearing[sid]})
        else:
            arm["sections"].append(sid)
            arm["bearing"] = _circular_mean([bearing[x] for x in arm["sections"]])
    for arm in arms:
        inbound = sorted(x for x in arm["sections"] if x.startswith("in_"))
        arm["name"] = (inbound or arm["sections"])[0].split("_", 1)[1]
        arm["inbound_heading"] = (arm["bearing"] + 180) % 360
    return arms


def arm_of(view, arms):
    """The arm whose bearing from the centre is closest to the camera's."""
    x, y = view["local_xy"]
    return min(arms, key=lambda a: _gap(math.degrees(math.atan2(x, y)) % 360, a["bearing"]))


def facing(view, arm):
    """+1 when the camera looks along the arm's inbound travel, -1 when it looks back against it."""
    return 1. if math.cos(math.radians(view["compass_heading_deg"] - arm["inbound_heading"])) >= 0 else -1.


def load_run(atlas, run):
    """Atlas data (camera poses, regions in local metres) and the evidence run's lane predictions."""
    data = read_json(atlas / "atlas_data.json")
    lanes = {(s["section_id"], r["region_id"]): (r, s["kind"] == "inbound_stopbar") for s in read_json(run / "lanes.json") for r in s["regions"]}
    return data, lanes


def leg_regions(data, lanes, view, arms):
    """Labelled regions of the road arm the camera stands on: its incoming and outgoing carriageways."""
    leg = set(arm_of(view, arms)["sections"])
    out = []
    for region in data["regions"]:
        key = (region["section_id"], region["region_id"])
        if region["section_id"] in leg and key in lanes and "polygon_local_m" in region:
            lines, color = label(*lanes[key])
            out.append((region, [region["section_id"].split("_", 1)[1]] + lines, color))
    return out


def view_image(atlas, view):
    image = Image.open(atlas / view["asset"]).convert("RGB")
    return image if list(image.size) == list(view["image_size"]) else image.resize(tuple(view["image_size"]))


def camera_xyz(points, view, dx=0., dh=0., height=HEIGHT):
    """Ground points (east, north) as camera right, up, forward, for a pose shifted by dx metres and dh degrees."""
    p = np.asarray(points, float)
    heading, pitch = math.radians(view["compass_heading_deg"] + dh), math.radians(view["pitch_deg"])
    base = math.radians(view["compass_heading_deg"])
    cx = view["local_xy"][0] + dx * math.cos(base)
    cy = view["local_xy"][1] - dx * math.sin(base)
    ex, ny = p[..., 0] - cx, p[..., 1] - cy
    ahead = ex * math.sin(heading) + ny * math.cos(heading)
    right = ex * math.cos(heading) - ny * math.sin(heading)
    return right, -ahead * math.sin(pitch) - height * math.cos(pitch), ahead * math.cos(pitch) - height * math.sin(pitch)


def focal(view):
    return view["image_size"][0] / 2 / math.tan(math.radians(view["hfov_deg"]) / 2)


def clip_near(polygon):
    """Sutherland-Hodgman against the plane forward = NEAR."""
    out = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(tuple(p + (q - p) * t for p, q in zip(a, b)))
    return out


def project_polygon(points, view, pose=(0., 0., HEIGHT)):
    w, h = view["image_size"]
    f = focal(view)
    right, up, forward = camera_xyz(points, view, *pose)
    return [(w / 2 + f * x / z, h / 2 - f * y / z) for x, y, z in clip_near(list(zip(right, up, forward)))]


def draw_regions(view, image, regions, pose=(0., 0., HEIGHT), note=None):
    """Regions as (region, label lines, colour), drawn in perspective with their labels and a caption."""
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tags = []
    for region, lines, color in regions:
        polygon = project_polygon(region["polygon_local_m"], view, pose)
        if len(polygon) < 3:
            continue
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        d.polygon(polygon, fill=rgb + (60,), outline=rgb + (255,), width=3)
        inside = [p for p in polygon if 0 <= p[0] < image.width and 0 <= p[1] < image.height]
        if inside:
            x = sum(p[0] for p in inside) / len(inside)
            tags.append(((x, min(max(p[1] for p in inside) - 40, image.height - 60)), " ".join(lines), color))
    out = Image.alpha_composite(image.convert("RGBA"), layer)
    d = ImageDraw.Draw(out)
    for (x, y), text, color in tags:
        half = d.textlength(text, font=font(22)) / 2 + 7
        d.rectangle((x - half, y - 15, x + half, y + 15), fill=color, outline="black")
        d.text((x, y), text, font=font(22), fill="black", anchor="mm")
    caption = Image.new("RGB", (out.width, 74), "#161a22")
    c = ImageDraw.Draw(caption)
    c.text((10, 8), f"{view['id']} | {view['capture_date']} | {view.get('distance_to_center_m', 0):.0f} m from centre | heading {view['compass_heading_deg']:.0f}",
           font=font(20), fill="white")
    c.text((10, 40), note or f"Satellite lane regions projected onto flat ground, camera height {pose[2]} m assumed; positions uncalibrated.",
           font=font(17), fill="#aebdcb")
    sheet = Image.new("RGB", (out.width, out.height + 74))
    sheet.paste(out.convert("RGB"), (0, 0))
    sheet.paste(caption, (0, out.height))
    return sheet, len(tags)


def boundaries(regions):
    """Unique strip edges of the regions, and each region's (left, right) edge indices."""
    edges, sides = [], []
    for region in regions:
        polygon = region["polygon_local_m"]
        k = len(polygon) // 2
        pair = []
        for edge in (polygon[:k], polygon[k:][::-1]):
            match = next((i for i, e in enumerate(edges) if len(e) == len(edge) and np.allclose(e, edge, atol=.05)), None)
            if match is None:
                edges.append(edge)
                match = len(edges) - 1
            pair.append(match)
        sides.append(tuple(pair))
    return edges, sides


def dense(edge, n=80, extend=0.):
    """Evenly spaced points along a boundary, optionally continued straight past both ends."""
    edge = np.asarray(edge, float)
    if extend:
        head, tail = edge[0] - edge[1], edge[-1] - edge[-2]
        edge = np.vstack([edge[0] + head / np.linalg.norm(head) * extend, edge, edge[-1] + tail / np.linalg.norm(tail) * extend])
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))]
    s = np.linspace(0, along[-1], n)
    return np.c_[np.interp(s, along, edge[:, 0]), np.interp(s, along, edge[:, 1])]


def crossings(samples, view, rows, pose):
    """Image x where each boundary crosses each row (nan where it does not), shape (boundaries, rows)."""
    w, h = view["image_size"]
    f = focal(view)
    right, up, forward = camera_xyz(np.stack(samples), view, *pose)
    out = np.full((len(samples), len(rows)), np.nan)
    for i in range(len(samples)):
        ok = forward[i] >= NEAR
        if ok.sum() < 2:
            continue
        u, v = w / 2 + f * right[i][ok] / forward[i][ok], h / 2 - f * up[i][ok] / forward[i][ok]
        order = np.argsort(v)
        u, v = u[order], v[order]
        for j, row in enumerate(rows):
            if v[0] <= row <= v[-1]:
                out[i, j] = np.interp(row, v, u)
    return out


def choose_rows(samples, view, pose=(0., 0., HEIGHT), count=3):
    """Rows spread over the stretch of ground where the projected boundaries are visible."""
    w, h = view["image_size"]
    f = focal(view)
    right, up, forward = camera_xyz(np.stack(samples), view, *pose)
    ok = forward >= NEAR
    u, v = w / 2 + f * right[ok] / forward[ok], h / 2 - f * up[ok] / forward[ok]
    v = v[(u >= 0) & (u < w) & (v >= 0) & (v < h)]
    require(len(v) > 10, "No projected boundary is visible in this view")
    top = max(float(v.min()), h / 2 - f * math.tan(math.radians(view["pitch_deg"])) + .04 * h)
    bottom = min(float(v.max()), h - 1)
    require(bottom - top > .08 * h, "Projected boundaries cover too little of the ground in this view")
    return [round(top + (bottom - top) * t) for t in np.linspace(.2, .92, count)]


def left_to_right(samples, sides, view, pose=(0., 0., HEIGHT)):
    """Renumber boundaries by their direction from the camera, driver-left first, so IDs follow the picture."""
    right, _, forward = camera_xyz(np.stack(samples), view, *pose)
    keys = [float(np.median(r[f >= NEAR] / f[f >= NEAR])) if (f >= NEAR).sum() else np.inf for r, f in zip(right, forward)]
    order = sorted(range(len(samples)), key=lambda i: keys[i])
    position = {old: new for new, old in enumerate(order)}
    return [samples[i] for i in order], [(position[a], position[b]) for a, b in sides]


def render_for_model(image, view, samples, sides, labels, rows, pose):
    """Image with numbered projected boundaries, strip labels, dashed rows R1..Rk and pixel rulers in margins."""
    w, h = image.size
    canvas = Image.new("RGB", (MARGIN + w + 12, RULER + h + RULER), "#161a22")
    canvas.paste(image, (MARGIN, RULER))
    d = ImageDraw.Draw(canvas)
    for x in range(0, w + 1, 20):
        major = x % 100 == 0
        d.line((MARGIN + x, RULER - (8 if major else 4), MARGIN + x, RULER - 1), fill="white")
        d.line((MARGIN + x, RULER + h, MARGIN + x, RULER + h + (8 if major else 4)), fill="white")
        if major:
            d.text((MARGIN + x, 1), str(x), font=font(15), fill="white", anchor="ma")
            d.text((MARGIN + x, RULER + h + 10), str(x), font=font(15), fill="white", anchor="ma")
    for j, row in enumerate(rows):
        _dashed(d, [(MARGIN, RULER + row), (MARGIN + w - 1, RULER + row)], ROW, on=10, off=10)
        d.text((4, RULER + row), f"R{j + 1}", font=font(18), fill=ROW, anchor="lm")
    xs = crossings(samples, view, list(range(0, h, 6)), pose)
    for i in range(len(samples)):
        points = [(MARGIN + x, RULER + y) for x, y in zip(xs[i], range(0, h, 6)) if np.isfinite(x) and 0 <= x < w]
        if len(points) >= 2:
            _dashed(d, points, "#16d5ec", on=14, off=8, width=3)
            x, y = points[-1]
            d.rectangle((x - 16, y - 24, x + 16, y - 2), fill="#16d5ec")
            d.text((x, y - 13), f"b{i + 1}", font=font(16), fill="black", anchor="mm")
    label_row = rows[len(rows) // 2]
    at = crossings(samples, view, [label_row], pose)[:, 0]
    for (left, right), (text, color) in zip(sides, labels):
        if np.isfinite(at[left]) and np.isfinite(at[right]):
            x = (at[left] + at[right]) / 2
            if 0 <= x < w:
                half = d.textlength(text, font=font(17)) / 2 + 5
                d.rectangle((MARGIN + x - half, RULER + label_row - 34, MARGIN + x + half, RULER + label_row - 12), fill=color)
                d.text((MARGIN + x, RULER + label_row - 23), text, font=font(17), fill="black", anchor="mm")
    return canvas


def visible_ids(samples, view, rows, pose):
    xs = crossings(samples, view, rows, pose)
    w = view["image_size"][0]
    return [f"b{i + 1}" for i in range(len(samples)) if np.any((xs[i] >= 0) & (xs[i] < w))]


def align_prompt(view, ids, rows, review=False):
    rows_text = ", ".join(f"R{j + 1} at y={r}" for j, r in enumerate(rows))
    task = ("The cyan lines now show a pose fitted to your earlier readings. Check each one again." if review else
            "The cyan lines come from an approximate camera pose and can be shifted sideways by up to two lanes, slightly rotated or scaled.")
    return f"""Match projected lane boundaries to the real road in street view {view['id']} (camera heading {view['compass_heading_deg']:.0f}°, captured {view['capture_date']}).
Dashed cyan lines {', '.join(ids)} are lane boundaries projected from a satellite tracing. {task} Each strip between them carries the label the pipeline gave it: BIKE for a bicycle strip, L/T/R plus a lane number for a motor lane with that predicted use, a lane number alone where use was not established, +RAIL for a lane shared with rails, OUT for a receiving lane.
Dashed magenta rows cross the road: {rows_text}. Rulers in the margins give x in image pixels.
For every boundary, find the real feature it is meant to follow (painted lane line, bicycle-lane line, kerb, median edge or rail-zone edge). Use the labels to decide which real line that is: a BIKE strip belongs on the pavement with bicycle markings or green paint, an L or R strip on the lane carrying that arrow, a +RAIL strip on the lane with rails. Report where that real feature crosses each row, in image pixels. Use null where it is hidden or outside the image. Report the real feature, not the cyan line.
If the scene lacks a strip the projection shows (for example no lane beyond the bicycle strip), or has one it does not show, say so in layout_mismatches instead of forcing a match.
Return JSON {{"boundaries":[{{"id":"b#","feature":"{'|'.join(FEATURES)}","xs":[{len(rows)} values, number or null, one per row],"evidence":"what identifies the real feature"}}],"layout_mismatches":[],"cannot_determine":[]}}
"""


def validate_readings(value, ids, rows, width):
    require(isinstance(value, dict) and isinstance(value.get("boundaries"), list), "Response requires boundaries[]")
    got = [b.get("id") if isinstance(b, dict) else None for b in value["boundaries"]]
    require(sorted(got) == sorted(ids), "Answer each shown boundary exactly once")
    readings = {}
    for b in value["boundaries"]:
        xs = b.get("xs")
        require(isinstance(xs, list) and len(xs) == len(rows), "xs needs one value per row")
        require(all(x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= width) for x in xs),
                "xs values must be null or inside the image")
        require(b.get("feature") in FEATURES, "Unknown feature")
        require(isinstance(b.get("evidence"), str), "Evidence text missing")
        readings[b["id"]] = [None if x is None else float(x) for x in xs]
    order = sorted(ids, key=lambda i: int(i[1:]))
    for j in range(len(rows)):
        seen = [readings[i][j] for i in order if readings[i][j] is not None]
        require(all(b > a for a, b in zip(seen, seen[1:])), "Real features must keep the left-to-right order of the boundaries")
    for key in ("layout_mismatches", "cannot_determine"):
        require(isinstance(value.get(key, []), list), f"{key} must be a list")
    return {"readings": readings, "features": {b["id"]: b["feature"] for b in value["boundaries"]},
            "evidence": {b["id"]: b["evidence"] for b in value["boundaries"]},
            "layout_mismatches": value.get("layout_mismatches", []), "cannot_determine": value.get("cannot_determine", [])}


def observed_matrix(samples, readings):
    observed = np.full((len(samples), len(next(iter(readings.values()), [None]))), np.nan)
    for key, xs in readings.items():
        observed[int(key[1:]) - 1] = [np.nan if x is None else x for x in xs]
    return observed


def levenberg_marquardt(residuals, x, low, high, iterations=80):
    """Minimise |residuals(x)|^2 within bounds, with a forward-difference Jacobian."""
    x = np.array(x, float)
    damping, r = 1e-2, residuals(x)
    for _ in range(iterations):
        jacobian = np.stack([(residuals(x + e) - r) / 1e-4 for e in np.eye(len(x)) * 1e-4], axis=1)
        normal = jacobian.T @ jacobian
        step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), -jacobian.T @ r)
        trial = np.clip(x + step, low, high)
        r_trial = residuals(trial)
        if r_trial @ r_trial < r @ r:
            x, r, damping = trial, r_trial, damping / 3
            if np.abs(step).max() < 1e-4:
                break
        else:
            damping *= 4
            if damping > 1e8:
                break
    return x


def fit_pose(samples, view, rows, readings, limits=(6., 6., (1.9, 3.3))):
    """Least-squares pose (dx m, dh deg, height m) from boundary readings: coarse grid, then Levenberg-Marquardt."""
    observed = observed_matrix(samples, readings)
    usable = np.isfinite(observed)
    require(usable.sum() >= 4 and usable.any(axis=1).sum() >= 2,
            "Too few readings to fit a pose: need four points on at least two boundaries")

    def cost(pose):
        predicted = crossings(samples, view, rows, pose)
        both = usable & np.isfinite(predicted)
        if both.sum() < max(3, usable.sum() // 2):
            return np.inf, predicted
        return float(np.mean(np.minimum((predicted[both] - observed[both]) ** 2, 120. ** 2))), predicted

    dmax, hmax, (hlow, hhigh) = limits
    best = (np.inf, (0., 0., HEIGHT))
    for dx in np.arange(-dmax, dmax + 1e-9, .25):
        for dh in np.arange(-hmax, hmax + 1e-9, .5):
            for height in np.arange(hlow, hhigh + 1e-9, .2):
                c = cost((dx, dh, height))[0]
                if c < best[0]:
                    best = (c, (float(dx), float(dh), float(height)))
    # Refine on the readings within 120 px of the best grid cell; the rest are outliers there.
    keep = usable & (np.abs(np.nan_to_num(cost(best[1])[1] - observed, nan=1e9)) < 120)

    def residuals(pose):
        predicted = crossings(samples, view, rows, tuple(pose))
        return np.where(np.isfinite(predicted[keep]), predicted[keep] - observed[keep], 120.)

    x = levenberg_marquardt(residuals, best[1], [-dmax, -hmax, hlow], [dmax, hmax, hhigh])
    if cost(tuple(x))[0] < best[0]:
        best = (cost(tuple(x))[0], tuple(x))
    pose = tuple(round(float(v), 3) for v in best[1])
    before, after = cost((0., 0., HEIGHT)), cost(pose)
    residual = np.abs(after[1] - observed)
    at_limit = abs(pose[0]) >= dmax - .05 or abs(pose[1]) >= hmax - .05 or pose[2] <= hlow + .02 or pose[2] >= hhigh - .02
    return {"pose": {"lateral_shift_m": pose[0], "heading_shift_deg": pose[1], "camera_height_m": pose[2]},
            "rms_px_before": round(math.sqrt(before[0]), 1) if np.isfinite(before[0]) else None,
            "rms_px_after": round(math.sqrt(after[0]), 1), "points": int(usable.sum()),
            "max_residual_px": round(float(np.nanmax(np.where(usable, residual, np.nan))), 1),
            "at_search_limit": bool(at_limit)}


def fit_group(entries, tolerance=1.5, limits=(4., 6., (1.9, 3.3))):
    """Shared sideways offset and camera height for views from one drive, with a heading correction per view.

    entries: dicts with samples, view, rows, readings and fit (that view's own fit_pose result).
    A view whose own offset is more than `tolerance` metres from the group median read its lines
    against the wrong lanes; it is left out of the fit and reported as an outlier.
    """
    # Offsets are compared in the leg frame: a camera looking back has its right-hand side reversed.
    shifts = [e.get("facing", 1.) * e["fit"]["pose"]["lateral_shift_m"] for e in entries]
    centre = float(np.median(shifts))
    inliers = [i for i, d in enumerate(shifts) if abs(d - centre) <= tolerance]
    outliers = [entries[i]["view"]["id"] for i in range(len(entries)) if i not in inliers]
    if len(inliers) < 2:
        return {"inliers": [entries[i]["view"]["id"] for i in inliers], "outliers": outliers, "shared": None}
    data = []
    for i in inliers:
        e = entries[i]
        observed = observed_matrix(e["samples"], e["readings"])
        p = e["fit"]["pose"]
        predicted = crossings(e["samples"], e["view"], e["rows"], (p["lateral_shift_m"], p["heading_shift_deg"], p["camera_height_m"]))
        keep = np.isfinite(observed) & (np.abs(np.nan_to_num(predicted - observed, nan=1e9)) < 120)
        data.append((e, observed, keep))

    def residuals(x):
        out = []
        for k, (e, observed, keep) in enumerate(data):
            predicted = crossings(e["samples"], e["view"], e["rows"], (e.get("facing", 1.) * x[0], x[2 + k], x[1]))
            out.append(np.where(np.isfinite(predicted[keep]), predicted[keep] - observed[keep], 120.))
        return np.concatenate(out)

    reach, hmax, (hlow, hhigh) = limits
    headings = [entries[i]["fit"]["pose"]["heading_shift_deg"] for i in inliers]
    start = [centre, float(np.median([entries[i]["fit"]["pose"]["camera_height_m"] for i in inliers]))] + headings
    # Bounds around the starting values, which after a whole-lane correction can sit beyond a fixed range.
    x = levenberg_marquardt(residuals, start, [centre - reach, hlow] + [h - hmax for h in headings],
                            [centre + reach, hhigh] + [h + hmax for h in headings])
    views = {}
    for k, (e, observed, keep) in enumerate(data):
        predicted = crossings(e["samples"], e["view"], e["rows"], (e.get("facing", 1.) * x[0], x[2 + k], x[1]))
        both = np.isfinite(observed) & np.isfinite(predicted)
        views[e["view"]["id"]] = {"heading_shift_deg": round(float(x[2 + k]), 3),
                                  "rms_px": round(float(np.sqrt(np.mean(np.minimum((predicted[both] - observed[both]) ** 2, 120. ** 2)))), 1)}
    return {"inliers": [entries[i]["view"]["id"] for i in inliers], "outliers": outliers,
            "shared": {"leg_shift_m": round(float(x[0]), 3), "camera_height_m": round(float(x[1]), 3),
                       "frame": "metres toward the driver's right of the approach's travel direction"}, "views": views}


def refine_pose(samples, view, rows, readings, start, reach=(4., 6., (1.9, 3.3))):
    """Levenberg-Marquardt from a given pose, without the grid search, for readings already matched near it.

    The bounds are centred on the start: after a whole-lane correction the start itself can lie
    several metres beyond the range the first grid search covers.
    """
    observed = observed_matrix(samples, readings)
    usable = np.isfinite(observed)
    require(usable.sum() >= 4 and usable.any(axis=1).sum() >= 2,
            "Too few readings to fit a pose: need four points on at least two boundaries")

    def residuals(pose):
        predicted = crossings(samples, view, rows, tuple(pose))
        return np.where(np.isfinite(predicted[usable]), predicted[usable] - observed[usable], 120.)
    dx, dh, (hlow, hhigh) = reach
    x = levenberg_marquardt(residuals, start, [start[0] - dx, start[1] - dh, hlow], [start[0] + dx, start[1] + dh, hhigh])
    r = residuals(x)
    return {"pose": {"lateral_shift_m": round(float(x[0]), 3), "heading_shift_deg": round(float(x[1]), 3), "camera_height_m": round(float(x[2]), 3)},
            "rms_px_after": round(float(np.sqrt(np.mean(np.minimum(r ** 2, 120. ** 2)))), 1), "points": int(usable.sum())}


def reassociate(samples, view, rows, readings, pose, gate=.45):
    """Give each read line position to the projected boundary nearest to it at that row.

    The positions the model read are kept; only which boundary each belongs to is decided again.
    A position farther than `gate` of the local boundary spacing from every boundary is dropped.
    """
    predicted = crossings(samples, view, rows, pose)
    out = {}
    for xs in readings.values():
        for j, x in enumerate(xs):
            column = predicted[:, j]
            finite = np.flatnonzero(np.isfinite(column))
            if x is None or len(finite) < 2:
                continue
            spacing = float(np.median(np.diff(np.sort(column[finite]))))
            b = finite[np.argmin(np.abs(column[finite] - x))]
            if abs(column[b] - x) > gate * spacing:
                continue
            slot = out.setdefault(f"b{b + 1}", [None] * len(rows))
            if slot[j] is None or abs(column[b] - x) < abs(column[b] - slot[j]):
                slot[j] = x
    return out


def realign_group(entries, lane_offset, iterations=3):
    """After a whole-lane correction, rematch every view's readings and refit the drive's pose.

    entries: dicts with samples, view, rows, facing, step (lane width, m), pose (camera frame) and
    raw_readings (the model's first readings). Poses are updated in place. Returns the last group fit.
    """
    for e in entries:
        e["pose"] = (e["pose"][0] + e["facing"] * lane_offset * e["step"], e["pose"][1], e["pose"][2])
    group = None
    for _ in range(iterations):
        for e in entries:
            e["readings"] = reassociate(e["samples"], e["view"], e["rows"], e["raw_readings"], e["pose"])
            try:
                e["fit"] = refine_pose(e["samples"], e["view"], e["rows"], e["readings"], e["pose"])
                e["pose"] = tuple(e["fit"]["pose"].values())
            except ValidationError:  # too few readings matched near this pose; keep it as it is
                e["fit"] = {"pose": dict(zip(("lateral_shift_m", "heading_shift_deg", "camera_height_m"), e["pose"]))}
        usable = [e for e in entries if sum(x is not None for xs in e["readings"].values() for x in xs) >= 4]
        if len(usable) >= 2:
            group = fit_group(usable)
            if group["shared"]:
                for e in usable:
                    if e["view"]["id"] in group["views"]:
                        e["pose"] = (e["facing"] * group["shared"]["leg_shift_m"], group["views"][e["view"]["id"]]["heading_shift_deg"],
                                     group["shared"]["camera_height_m"])
    return group


def lane_offset(votes):
    """The drive's whole-lane correction from the views' choices, in lanes toward the arm's inbound driver-right.

    A majority of two or more confident votes decides. A single confident vote stands when no other
    confident vote disagrees; that is weaker and is reported as such.
    """
    confident = {v: k for v, (k, c) in votes.items() if c in ("high", "medium")}
    tally = Counter(confident.values()).most_common()
    if tally and tally[0][1] >= 2 and tally[0][1] > len(confident) / 2:
        return tally[0][0], "group_vote"
    if len(tally) == 1 and tally[0][1] == 1:
        return tally[0][0], "single_vote"
    return None, None


LETTERS = "ABCDE"


def lane_width(regions):
    """Median width of the motor strips on a leg, in metres, for stepping the projection one lane at a time."""
    widths = []
    for region, lines, _ in regions:
        if lines and lines[-1] in ("BIKE", "BUFFER", "PARKING", "NOT ROAD", "RAIL"):
            continue
        polygon = np.asarray(region["polygon_local_m"], float)
        k = len(polygon) // 2
        widths.append(float(np.median(np.linalg.norm(polygon[:k] - polygon[k:][::-1], axis=1))))
    return float(np.median(widths)) if widths else 3.6


def offset_sheet(view, image, regions, pose, step, steps=(-2, -1, 0, 1, 2)):
    """Candidate overlays shifted by whole lanes along the road, lettered A-E, two per row."""
    panels = []
    for letter, k in zip(LETTERS, steps):
        trial = (pose[0] + k * step, pose[1], pose[2])  # k lanes toward the camera's right
        panel, _ = draw_regions(view, image, regions, trial, f"Panel {letter}")
        top = int(panel.height * .38)
        panel = panel.crop((0, top, panel.width, panel.height))
        panel.thumbnail((760, 520))
        d = ImageDraw.Draw(panel)
        d.rectangle((0, 0, 70, 52), fill="#161a22")
        d.text((35, 26), letter, font=font(40), fill="white", anchor="mm")
        panels.append(panel)
    w, h = max(p.width for p in panels), max(p.height for p in panels)
    sheet = Image.new("RGB", (2 * w + 10, 3 * h + 20), "#0d1117")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * (w + 10), (i // 2) * (h + 10)))
    return sheet, dict(zip(LETTERS, steps))


def offset_prompt(view):
    return f"""Choose the overlay that puts every labelled strip on the right part of the road in street view {view['id']} (camera heading {view['compass_heading_deg']:.0f}°, captured {view['capture_date']}).
Panels A-E show the same picture with lanes traced on satellite imagery projected into it, the whole overlay shifted sideways by a different number of lane widths in each. Labels: L/T/R plus a number = motor lane with that predicted use; a number alone = motor lane of unknown use; BIKE = bicycle strip; +RAIL = lane shared with rails; OUT = receiving lane of the opposite carriageway; BUFFER, PARKING as named. The first word of each label names the approach or exit section the strip belongs to.
Judge by content, not by how neatly lines meet: an L strip should sit on a lane with a left arrow, an R strip on a lane with a right arrow, BIKE on the bicycle lane, +RAIL on the lane carrying rails, and strips of one approach should not cross the centre line or median into the other carriageway.
Return JSON {{"panel":"A|B|C|D|E|none","confidence":"high|medium|low","reasons":["visible evidence, naming strips and markings"],"remaining_problems":["strips still misplaced in the chosen panel, if any"]}}. Use "none" if no panel places the strips on matching lanes.
"""


def validate_offset(value):
    require(isinstance(value, dict) and value.get("panel") in tuple(LETTERS) + ("none",), "panel must be A-E or none")
    require(value.get("confidence") in ("high", "medium", "low"), "Invalid confidence")
    require(isinstance(value.get("reasons"), list) and value["reasons"], "Give the visible reasons for the choice")
    require(isinstance(value.get("remaining_problems", []), list), "remaining_problems must be a list")
    return {"panel": value["panel"], "confidence": value["confidence"], "reasons": value["reasons"],
            "remaining_problems": value.get("remaining_problems", [])}


def visible_strips(samples, sides, labels, view, rows, pose):
    """Labels of the strips whose middle lies inside the image on the label row of render_for_model."""
    at = crossings(samples, view, [rows[len(rows) // 2]], pose)[:, 0]
    w = view["image_size"][0]
    return [text for (left, right), (text, _) in zip(sides, labels)
            if np.isfinite(at[left]) and np.isfinite(at[right]) and 0 <= (at[left] + at[right]) / 2 < w]


def check_prompt(view, ids, rows, strips):
    rows_text = ", ".join(f"R{j + 1} at y={r}" for j, r in enumerate(rows))
    return f"""Check a lane overlay on street view {view['id']} (camera heading {view['compass_heading_deg']:.0f}°, captured {view['capture_date']}).
The cyan lines {', '.join(ids)} bound lanes traced on satellite imagery and projected into this picture with a corrected camera pose. Each strip carries the label the pipeline gave it: BIKE for a bicycle strip, L/T/R plus a lane number for a motor lane with that predicted use, a lane number alone where use was not established, +RAIL for a lane shared with rails, OUT for a receiving lane. Dashed magenta rows: {rows_text}. Rulers give x in image pixels.
For every labelled strip you can see ({', '.join(strips)}), say whether the pavement inside it agrees with its label: a BIKE strip should hold bicycle symbols or green paint between bicycle-lane lines; an L or R strip should hold that arrow if any arrow is painted in it; a +RAIL strip should hold rails; any motor lane should be travel pavement, not a median, rail zone, kerb or sidewalk. Answer consistent, inconsistent or unknown, with the visible evidence. An arrow that belongs to a neighbouring strip makes both strips inconsistent.
If any strip is inconsistent the overlay is misplaced. Then give, for every boundary, where the real feature it should follow crosses each row, reading from the labels which real line that is, so the pose can be refitted. If no strip is inconsistent, return "boundaries": [].
Return JSON {{"strip_checks":[{{"strip":"label as shown","verdict":"consistent|inconsistent|unknown","evidence":"what is visible"}}],"boundaries":[{{"id":"b#","feature":"{'|'.join(FEATURES)}","xs":[{len(rows)} values, number or null],"evidence":"..."}}],"layout_mismatches":[],"cannot_determine":[]}}
"""


def validate_check(value, ids, rows, width, strips):
    require(isinstance(value, dict) and isinstance(value.get("strip_checks"), list), "Response requires strip_checks[]")
    checks = {}
    for c in value["strip_checks"]:
        require(isinstance(c, dict) and c.get("strip") in strips, "strip_checks must name strips shown in the image")
        require(c.get("verdict") in ("consistent", "inconsistent", "unknown"), "Invalid strip verdict")
        require(isinstance(c.get("evidence"), str) and c["evidence"].strip(), "Each strip check needs evidence")
        checks[c["strip"]] = {"verdict": c["verdict"], "evidence": c["evidence"]}
    require(checks, "Check at least one strip")
    bad = any(c["verdict"] == "inconsistent" for c in checks.values())
    if bad:
        corrected = validate_readings({**value, "boundaries": value.get("boundaries", [])}, ids, rows, width)
    else:
        require(not value.get("boundaries"), "Return no boundary readings when no strip is inconsistent")
        corrected = None
    for key in ("layout_mismatches", "cannot_determine"):
        require(isinstance(value.get(key, []), list), f"{key} must be a list")
    return {"strip_checks": checks, "inconsistent": bad, "corrected": corrected,
            "layout_mismatches": value.get("layout_mismatches", []), "cannot_determine": value.get("cannot_determine", [])}


def readings_json(readings):
    return json.dumps(readings, sort_keys=True)
