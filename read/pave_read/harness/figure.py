"""The result figure: every traced region on the satellite image, labelled, with an arrow for each proposed turn.

Everything drawn is the run's own output (lanes.json, movements.json); nothing is added or corrected here. A region's
polygon starts and ends at its junction end (the stop line of an approach lane, the exit start of an exit lane), so
an arrow runs from the middle of that end of the approach lane to the middle of that end of the exit lane it reaches.
"""
import math

from PIL import Image, ImageDraw

from pave_read.hybrid.geometry import font

MOTOR, BIKE, OTHER, RAIL, REJECTED, UNSURE = "#16d5ec", "#35c94a", "#8fa3b8", "#c58bff", "#8a8a8a", "#ffe14d"
TURN_COLOR = {"left": "#ff5cf4", "through": "#ffffff", "right": "#ffe14d", "u_turn": "#ff8a3d"}
LETTER = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}
TURN_NAME = {"left": "left turn", "through": "through", "right": "right turn", "u_turn": "U-turn"}
PLAIN = {"bicycle_lane": ("BIKE", BIKE), "parking": ("PARKING", OTHER), "median_or_buffer": ("BUFFER", OTHER),
         "shoulder": ("SHOULDER", OTHER), "sidewalk_or_verge": ("VERGE", OTHER), "rail_or_bus_only": ("RAIL/BUS", RAIL),
         "unknown": ("?", UNSURE)}
SHARED = {"shared_rail_motor_lane": "+RAIL", "shared_bicycle_motor_lane": "+BIKE"}
LEGEND = {MOTOR: "motor lane: L left, T through, R right, U U-turn, and its number from the driver's left; OUT exit lane",
          BIKE: "bicycle lane", OTHER: "buffer, median, parking, shoulder or verge", RAIL: "rail or bus only",
          REJECTED: "traced, but judged not a lane by lane use", UNSURE: "motor strip that does not reach the stop line; unknown strip"}


def label(region, inbound):
    """(text, colour) of one region."""
    kind, existence = region["surface_type"], region.get("existence")
    if existence == "rejected_by_lane_use":
        return "NOT A LANE", REJECTED
    if existence == "not_at_gate":
        return "LANE?", UNSURE
    if not region.get("region_id"):
        return PLAIN.get(kind, (kind.replace("_", " ").upper(), OTHER))
    turns = (region.get("lane_use_prediction") or {}).get("allowed_turns") or []
    letters = "".join(LETTER[t] for t in ("left", "through", "right", "u_turn") if t in turns) if inbound else "OUT"
    return f"{letters}{region.get('motor_lane_index') or ''}" + (" " + SHARED[kind] if kind in SHARED else ""), MOTOR


def gate(polygon):
    """The middle of the region's junction end (its first and last points)."""
    return (polygon[0][0] + polygon[-1][0]) / 2, (polygon[0][1] + polygon[-1][1]) / 2


def axis(polygon):
    """Unit vector along the region, toward the junction."""
    cx, cy = sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon)
    gx, gy = gate(polygon)
    length = math.hypot(gx - cx, gy - cy) or 1
    return (gx - cx) / length, (gy - cy) / length


def crossing(p, u, q, v):
    """Where the two lane axes meet, or the midpoint when they are parallel (a through movement)."""
    det = u[0] * v[1] - u[1] * v[0]
    if abs(det) < .2:
        return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    t = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / det
    return p[0] + u[0] * t, p[1] + u[1] * t


def curve(start, control, end, steps=24):
    return [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0],
             (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1]) for t in (i / steps for i in range(steps + 1))]


def arrow(d, points, color, width=5):
    d.line(points, fill="black", width=width + 4)
    d.line(points, fill=color, width=width)
    (ax, ay), (bx, by) = points[-3], points[-1]
    angle = math.atan2(by - ay, bx - ax)
    d.polygon([(bx, by)] + [(bx - 26 * math.cos(angle + s), by - 26 * math.sin(angle + s)) for s in (.42, -.42)], fill=color, outline="black")


def overview(image, sections, movements, margin=70):
    """The labelled figure, cropped to what it shows, with a legend above it."""
    canvas = image.convert("RGB")
    d = ImageDraw.Draw(canvas)
    f = font(22)
    shown, tags, colours = [], [], []
    for section in sections:
        inbound = section["kind"] == "inbound_stopbar"
        for i, region in enumerate(section["regions"]):
            polygon = [tuple(p) for p in region.get("polygon_px") or []]
            if len(polygon) < 3:
                continue
            text, color = label(region, inbound)
            d.line(polygon + [polygon[0]], fill=color, width=4)
            shown.extend(polygon)
            colours.append(color)
            # The tag sits beyond the far end of the strip, on a leader line. Along steep strips neighbouring tags step
            # through three rows (exits through three more) so narrow strips stay legible.
            ux, uy = axis(polygon)
            gx, gy = gate(polygon)
            depth = max((gx - p[0]) * ux + (gy - p[1]) * uy for p in polygon)
            row = (i % 3 + (0 if inbound else 3)) if abs(uy) > abs(ux) else 0
            half = d.textlength(text, font=f) / 2 + 7
            reach = depth + 26 + 36 * row if abs(uy) > abs(ux) else depth + 14 + half
            # A tag that would overlap one already placed moves further out along the strip.
            for _ in range(20):
                x, y = gx - ux * reach, gy - uy * reach
                box = (x - half - 3, y - 18, x + half + 3, y + 18)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for *_, b in tags):
                    break
                reach += 34
            tags.append(((gx - ux * depth, gy - uy * depth), (x, y), half, text, color, box))
    for (fx, fy), (x, y), half, text, color, _ in tags:
        d.line((fx, fy, x, y), fill=color, width=2)
        d.rectangle((x - half, y - 15, x + half, y + 15), fill=color, outline="black")
        d.text((x, y), text, font=f, fill="black", anchor="mm")
        shown.extend([(x - half, y - 15), (x + half, y + 15)])
    regions = {(s["section_id"], r["region_id"]): r for s in sections for r in s["regions"] if r.get("region_id")}
    turns = []
    for m in movements:
        if m.get("status") != "candidate":
            continue
        for pair in m.get("lane_pairs") or []:
            a, b = regions.get((m["in_section_id"], pair["in_region_id"])), regions.get((m["out_section_id"], pair["out_region_id"]))
            if not a or not b:
                continue
            p, q = gate(a["polygon_px"]), gate(b["polygon_px"])
            points = curve(p, crossing(p, axis(a["polygon_px"]), q, axis(b["polygon_px"])), q)
            arrow(d, points, TURN_COLOR[m["turn"]])
            shown.extend(points)
            turns.append(m["turn"])
    if not shown:
        return canvas
    x1, y1 = max(0, min(p[0] for p in shown) - margin), max(0, min(p[1] for p in shown) - margin)
    x2, y2 = min(canvas.width, max(p[0] for p in shown) + margin), min(canvas.height, max(p[1] for p in shown) + margin)
    body = canvas.crop((int(x1), int(y1), int(x2), int(y2)))
    rows = [(c, LEGEND[c]) for c in LEGEND if c in colours]
    rows += [(TURN_COLOR[t], f"arrow: {TURN_NAME[t]}, from the approach lane to the exit lane it reaches")
             for t in ("left", "through", "right", "u_turn") if t in turns]
    band = 20 + 34 * len(rows)
    out = Image.new("RGB", (body.width, body.height + band), "#161a22")
    out.paste(body, (0, band))
    o = ImageDraw.Draw(out)
    for i, (color, text) in enumerate(rows):
        o.rectangle((14, 12 + 34 * i, 44, 34 + 34 * i), fill=color)
        o.text((56, 23 + 34 * i), text, font=f, fill="white", anchor="lm")
    return out
