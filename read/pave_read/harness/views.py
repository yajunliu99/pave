"""Street views in the lean Read pipeline.

An approach uses up to three views: the near look-back (taken near the stop bar, facing away from the junction, so it
shows the lanes at the stop bar with traffic coming toward the camera), and the mid and far forward views. A camera is
marked on the satellite crop from its recorded position and heading, and traced boundaries or lanes are drawn into a
view by projecting satellite ground points through that pose over flat ground (camera height assumed, see
autoloop/street_panel). The pose is not calibrated, so a projection can sit up to about a lane to one side; it shows
which painted line or lane a trace stands for, not where it is to the centimetre.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from ..autoloop import render, street_panel
from ..hybrid.common import resource_path
from ..hybrid.geometry import font

POSITIONS = (("near", True), ("mid", False), ("far", False))
PAPER = "#161a22"
# A lane can look unlike its neighbours, or be something else upstream of the stop bar. Every stage that decides what a
# strip is (trace, blind reading, lane use) is told both, so the model judges a strip by what it is at the stop bar.
PAVING = ("A lane may be paved differently from its neighbours (brick, concrete or coloured surfacing): tell a lane from a median or buffer"
          " by whether it is raised or curbed, whether vehicles stand or drive in it and whether the stop bar and lane lines reach it, not by its colour.")
AT_STOP = ("A strip can change along the road: parking or a shoulder upstream can become a turn lane near the stop bar, and turning traffic can"
           " enter a bicycle lane to turn right. What it is at the stop bar is what counts.")


def approach_views(base, direction, root):
    """The approach's near look-back, mid and far forward views that were acquired, labelled C1, C2, ..."""
    out = []
    for position, reverse in POSITIONS:
        for sid, s in base.get("sources", {}).items():
            if (s.get("role") == "gsv" and s.get("direction") == direction and bool(s.get("reverse_view")) == reverse
                    and s.get("sampling_position") == position and s.get("path") and s.get("actual_lat") is not None):
                out.append({**s, "id": sid, "path": str(resource_path(root, s["path"]))})
                break
    for i, v in enumerate(out):
        v["label"] = f"C{i + 1}"
    return out


def caption(v):
    way = "look-back, facing away from the junction" if v.get("reverse_view") else "forward, facing the junction"
    return f"{v['label']} {v['sampling_position']} {way}: {v.get('distance_to_center_m')} m from the centre, {v.get('capture_date') or 'date unknown'}"


def mark_cameras(canvas, window, views, to_pixel, scale):
    """Each camera of `views` that stands on or near the crop, with its field of view, on a ruled crop."""
    frame = window["frame"]
    for v in views:
        at, relative, _ = street_panel.camera_on_crop(v, frame, to_pixel)
        if -frame.width * .2 <= at[0] <= frame.width * 1.2 and -frame.height * .2 <= at[1] <= frame.height * 1.2:
            canvas = street_panel.mark_camera(canvas, at, relative, v.get("view_settings", {}).get("fov", 90), scale, label=v["label"])
    return canvas


def _dense(points, steps=12):
    out = []
    for a, b in zip(points, points[1:] + points[:1]):
        out += [(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps) for k in range(steps)]
    return out


def _visible(points, size):
    return [p for p in points if p is not None and -200 < p[0] < size[0] + 200 and -200 < p[1] < size[1] + 200]


def with_boundaries(v, root, window, trace, to_pixel, mpp, extend_px=40):
    """The view with every traced boundary b1..bN projected onto the road and labelled at its near end."""
    image = street_panel.load(v, root)
    camera = to_pixel(v["actual_lon"], v["actual_lat"])
    d = ImageDraw.Draw(image)
    stations, frame = window["stations"], window["frame"]
    for i, b in enumerate(trace["boundaries"]):
        pts = [(x, y) for x, y in zip(b["xs"], stations) if x is not None]
        if len(pts) < 2:
            continue
        pts = [(pts[0][0], pts[0][1] - extend_px)] + pts + [(pts[-1][0], pts[-1][1] + extend_px)]
        seen = _visible(street_panel.project([frame.to_source(x, y) for x, y in pts], camera, v, mpp), image.size)
        if len(seen) < 2:
            continue
        colour = street_panel.EDGE_COLORS[i % len(street_panel.EDGE_COLORS)]
        render._dashed(d, seen, colour, on=14, off=8, width=3)
        low = max(seen, key=lambda p: p[1])
        d.text((min(max(low[0], 18), image.width - 18), min(max(low[1] - 16, 16), image.height - 16)), f"b{i + 1}",
               font=font(20), fill=colour, stroke_width=3, stroke_fill="black", anchor="mm")
    return image


def with_regions(v, root, regions, to_pixel, mpp, colour="#ffd400"):
    """The view with lane regions (label, source-pixel polygon) projected onto the road and labelled."""
    image = street_panel.load(v, root)
    camera = to_pixel(v["actual_lon"], v["actual_lat"])
    d = ImageDraw.Draw(image)
    for label, polygon in regions:
        if len(polygon) < 3:
            continue
        seen = _visible(street_panel.project(_dense([tuple(p) for p in polygon]), camera, v, mpp), image.size)
        if len(seen) < 3:
            continue
        for a, b in zip(seen, seen[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 400:
                d.line((a, b), fill=colour, width=3)
        cx = sum(p[0] for p in seen) / len(seen); cy = max(p[1] for p in seen) - 30
        cx, cy = min(max(cx, 20), image.width - 20), min(max(cy, 20), image.height - 20)
        d.text((cx, cy), label, font=font(20), fill=colour, stroke_width=3, stroke_fill="black", anchor="mm")
    return image


def column(cells, width=600):
    """Views stacked top to bottom, each under its caption."""
    out = []
    for image, text in cells:
        im = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (width, im.height + 30), PAPER)
        cell.paste(im, (0, 30))
        ImageDraw.Draw(cell).text((6, 6), text, font=font(15), fill="white")
        out.append(cell)
    sheet = Image.new("RGB", (width, sum(c.height for c in out) + 10 * max(len(out) - 1, 0)), PAPER)
    y = 0
    for c in out:
        sheet.paste(c, (0, y)); y += c.height + 10
    return sheet


def reading_sheet(raw, views, root, path, height=1200, view_width=420):
    """Blind reading: the unmarked arm crop (travel up) beside the approach's street views, no lane information."""
    sat = raw.resize((round(raw.width * height / raw.height), height), Image.Resampling.LANCZOS)
    col = column([(street_panel.load(v, root), caption(v)) for v in views], view_width) if views else None
    width = sat.width + (10 + col.width if col else 0)
    sheet = Image.new("RGB", (width, max(height, col.height if col else 0) + 34), PAPER)
    ImageDraw.Draw(sheet).text((8, 7), "satellite, traffic on this approach drives UP", font=font(18), fill="white")
    sheet.paste(sat, (0, 34))
    if col:
        sheet.paste(col, (sat.width + 10, 34))
    sheet.save(path, quality=90)
    return path


def reading_prompt(captions):
    return f"""This image shows one approach to a signalised intersection.
Left: a satellite crop rotated so that traffic on this approach drives UP toward the intersection; the stop bar and crosswalk are near the top. The road's other carriageway runs down the left side.
Right: street views of this approach ({'; '.join(captions) or 'none available'}). A forward photo looks along the direction of travel toward the junction. A look-back photo is taken near the stop bar facing away from the junction: the approach's traffic comes toward the camera, so the driver's left is on the RIGHT of that photo. A street view may be old, taken from the opposite carriageway, or partly blocked; say so when it is.
Read the lane layout of THIS approach at the stop bar, counting from the driver's left.
- Count only lanes open to general motor traffic in the direction of travel. Turn pockets count; a lane shared with a streetcar counts.
- {AT_STOP} So a curbside turn lane counts where it opens from parking or a shoulder or lies beside a bicycle lane, and so does a bicycle lane that turning motor traffic also uses at the stop bar.
- {PAVING}
- Do not count opposing lanes, medians, rail- or bus-only lanes, gutters, or bicycle lanes, parking and shoulders that motor traffic does not use at the stop bar; report those separately.
- For each counted lane give the movements its pavement arrows allow (left, through, right, u_turn), or an empty list if no arrow is legible.
- If shadow, occlusion, image age or camera position prevents a reliable count, set count_settled to false instead of guessing.
Return JSON only:
{{"lanes":[{{"lane":1,"turns":["left"],"evidence":"what shows this lane and its arrows"}}],"bicycle_lanes":0,"rail_or_bus_only_lanes":0,"parking_lanes":0,"count_confidence":"high|medium|low","count_settled":true,"limitations":"what limited the reading"}}"""
