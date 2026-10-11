"""Label a finished visual-evidence run: lane number, facility type and movements.

Writes one travel-up panel per section, a sheet of all panels and a junction
overview with movement arrows. Everything shown is the model's prediction from
that run; nothing is added or corrected here. With --exits, the exits are shown as
the exit audit read them (as in the atlas), not as the classification step did.
"""
import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop import render
from movement_fixer.autoloop.lane_labels import LEGEND, TURN_COLOR, label
from movement_fixer.autoloop.strips import Frame
from movement_fixer.fusion.downstream import with_exit_audits
from movement_fixer.hybrid.common import read_json
from movement_fixer.hybrid.geometry import font
from movement_fixer.hybrid.legs import order

def section_frame(section, side=44, end=40):
    heading = section.get("heading_deg", section.get("rotation_ccw", 0))
    probe = Frame((0, 0), heading, 0, 0)
    local = [probe.to_local(*p) for r in section["regions"] for p in r["polygon"]]
    xs, ys = [p[0] for p in local], [p[1] for p in local]
    centre = probe.to_source((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    return Frame(centre, heading, 2 * math.ceil((max(xs) - min(xs)) / 2 + side), 2 * math.ceil((max(ys) - min(ys)) / 2 + end))


def fitted(d, box, lines, fill):
    """Fill a label box and write its lines in the largest font that fits."""
    x1, y1, x2, y2 = box
    d.rectangle(box, fill=fill)
    for size in range(26, 9, -1):
        f = font(size)
        if all(d.textlength(t, font=f) <= x2 - x1 - 4 for t in lines) and size * 1.15 * len(lines) <= y2 - y1 - 2:
            break
    step = (y2 - y1) / len(lines)
    for i, text in enumerate(lines):
        d.text(((x1 + x2) / 2, y1 + step * (i + .5)), text, font=f, fill="black", anchor="mm")


def movement_lines(section, movements):
    lines = []
    for m in movements:
        if m["from_direction"] != section["direction"]:
            continue
        # ASCII only: the fallback font has no arrow glyph.
        pairs = ", ".join(f"{p['ib_lane']} to {p['ob_lane']}" for p in m["lane_pairs"])
        lines.append(f"{m['turn'].replace('_', '-')} into {m['to_direction']}: " + (f"lane {pairs}" if pairs else m["status"]))
    return lines


def panel(image, section, lanes, movements, scale):
    inbound = section["kind"] == "inbound_stopbar"
    frame = section_frame(section)
    body = render.enlarge(render.crop(image, frame), scale)
    d = ImageDraw.Draw(body)
    predicted = {r["region_id"]: r for r in lanes["regions"]}
    band = 34 * scale // 2
    for region in section["regions"]:
        lines, color = label(predicted[region["id"]], inbound)
        points = [tuple(v * scale for v in frame.to_local(*p)) for p in region["polygon"]]
        d.line(points + [points[0]], fill=color, width=3)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        # The label sits just beyond the junction end of the strip, so no lane pixels are covered.
        end = min(ys) if inbound else max(ys)
        near = sorted(points, key=lambda p: abs(p[1] - end))[:2]
        left, right = min(p[0] for p in near), max(p[0] for p in near)
        fitted(d, (left + 1, end - band if inbound else end + 2, right - 1, end - 2 if inbound else end + band), lines, color)
    notes = movement_lines(section, movements) if inbound else []
    count = lanes["count"]["model_motor_count"]
    head = f"{section['id']}   {count} motor lane{'s' if count != 1 else ''}"
    canvas = Image.new("RGB", (max(body.width, 420), 44 + body.height + 26 * len(notes) + 10), "#161a22")
    c = ImageDraw.Draw(canvas)
    c.text((8, 8), head, font=font(26), fill="white")
    canvas.paste(body, ((canvas.width - body.width) // 2, 44))
    for i, text in enumerate(notes):
        c.text((8, 44 + body.height + 6 + 26 * i), text, font=font(20), fill="#cfd8e3")
    return canvas


def arrow(d, start, control, end, color, width=5):
    points = [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0],
               (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1]) for t in (i / 24 for i in range(25))]
    d.line(points, fill="black", width=width + 4)
    d.line(points, fill=color, width=width)
    (ax, ay), (bx, by) = points[-3], points[-1]
    angle = math.atan2(by - ay, bx - ax)
    head = [(bx, by)] + [(bx - 26 * math.cos(angle + s), by - 26 * math.sin(angle + s)) for s in (.42, -.42)]
    d.polygon(head, fill=color, outline="black")


def crossing(p, u, q, v):
    """Intersection of the two lane axes, or the midpoint when they are parallel (a through movement)."""
    det = u[0] * v[1] - u[1] * v[0]
    if abs(det) < .2:
        return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    t = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / det
    return p[0] + u[0] * t, p[1] + u[1] * t


def axis(region):
    cx = sum(p[0] for p in region["polygon"]) / len(region["polygon"])
    cy = sum(p[1] for p in region["polygon"]) / len(region["polygon"])
    gx, gy = region["gate_point"]
    length = math.hypot(gx - cx, gy - cy) or 1
    return (gx - cx) / length, (gy - cy) / length


def overview(image, config, lanes, movements):
    canvas = image.copy()
    d = ImageDraw.Draw(canvas)
    byid = {s["section_id"]: {r["region_id"]: r for r in s["regions"]} for s in lanes}
    geometry = {s["id"]: {r["id"]: r for r in s["regions"]} for s in config["sections"]}
    tags = []
    for section in config["sections"]:
        inbound = section["kind"] == "inbound_stopbar"
        for region in section["regions"]:
            lines, color = label(byid[section["id"]][region["id"]], inbound)
            points = [tuple(p) for p in region["polygon"]]
            d.line(points + [points[0]], fill=color, width=4)
            tags.append((section, region, " ".join(lines), color))
    for section, region, text, color in tags:
        # Tag beyond the far end of the strip; neighbours cycle through three rows so narrow strips stay legible.
        ux, uy = axis(region)
        points = region["polygon"]
        far = min(points, key=lambda p: p[0] * ux + p[1] * uy)
        depth = (region["gate_point"][0] - far[0]) * ux + (region["gate_point"][1] - far[1]) * uy
        # Inbound and outbound sections sit side by side, so outbound tags use the outer pair of rows.
        row = section["regions"].index(region) % 3 + (0 if section["kind"] == "inbound_stopbar" else 3) if abs(uy) > abs(ux) else 0
        reach = depth + 26 + 36 * row if abs(uy) > abs(ux) else depth + 14 + d.textlength(text, font=font(22)) / 2
        x, y = region["gate_point"][0] - ux * reach, region["gate_point"][1] - uy * reach
        half = d.textlength(text, font=font(22)) / 2 + 7
        d.line((region["gate_point"][0] - ux * depth, region["gate_point"][1] - uy * depth, x, y), fill=color, width=2)
        d.rectangle((x - half, y - 15, x + half, y + 15), fill=color, outline="black")
        d.text((x, y), text, font=font(22), fill="black", anchor="mm")
    for m in movements:
        for pair in m["lane_pairs"]:
            a, b = geometry[m["in_section_id"]][pair["in_region_id"]], geometry[m["out_section_id"]][pair["out_region_id"]]
            u, v = axis(a), axis(b)
            arrow(d, a["gate_point"], crossing(a["gate_point"], u, b["gate_point"], v), b["gate_point"], TURN_COLOR[m["turn"]])
    rows = LEGEND + [(TURN_COLOR[t], f"arrow: {t.replace('_', '-')} movement candidate") for t in ("left", "through", "right")]
    d.rectangle((16, 16, 44 + max(d.textlength(t, font=font(22)) for _, t in rows) + 40, 30 + 34 * len(rows)), fill="#161a22")
    for i, (color, text) in enumerate(rows):
        d.rectangle((28, 30 + 34 * i, 58, 52 + 34 * i), fill=color)
        d.text((70, 28 + 34 * i), text, font=font(22), fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--exits", type=Path, help="exit audit predictions.json; ignored when missing")
    args = parser.parse_args()
    config = read_json(args.run / "inference_config.json")
    lanes = read_json(args.run / "lanes.json")
    if args.exits and args.exits.exists():
        lanes = with_exit_audits(lanes, read_json(args.exits))
    movements = read_json(args.run / "movements.json")
    image = Image.open(args.run / "sources" / "satellite.png").convert("RGB")
    out = args.run / "labels"
    out.mkdir(exist_ok=True)
    bysection = {s["section_id"]: s for s in lanes}
    panels = {}
    for section in config["sections"]:
        panels[section["id"]] = panel(image, section, bysection[section["id"]], movements, args.scale)
        panels[section["id"]].save(out / (section["id"] + ".png"))
    rows = [[panels[s["id"]] for s in sorted(config["sections"], key=order) if s["kind"] == kind]
            for kind in ("inbound_stopbar", "outbound_receiving")]
    rows = [row for row in rows if row]
    gap = 18
    width = max(sum(p.width for p in row) + gap * (len(row) + 1) for row in rows)
    sheet = Image.new("RGB", (width, sum(max(p.height for p in row) + gap for row in rows) + gap), "#0d1117")
    y = gap
    for row in rows:
        x = gap
        for p in row:
            sheet.paste(p, (x, y))
            x += p.width + gap
        y += max(p.height for p in row) + gap
    sheet.save(out / "sections.png")
    overview(image, config, lanes, movements).save(out / "overview.png")
    print({"panels": len(panels), "sheet": sheet.size, "output": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
