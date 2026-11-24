"""Images, prompts and answer checks of the lean Read pipeline's model requests.

trace     one per road arm: stop line, exit start and every longitudinal boundary of both carriageways, with what
          lies between each pair (merges anchoring, tracing and the surface audit of the earlier pipeline)
retrace   one per arm whose independent street-view reading sees more lanes than the trace, or with a measured issue;
          the trace's boundaries are projected into the approach's street views
use       one per approach: whether each traced lane exists, its allowed turns and the exit lanes each turn feeds,
          with the lanes projected into the approach's street views and the traced cross-section in the prompt; it
          also reports a lane it sees with no outline of its own (missing_lanes), and temporary traffic control
          (temporary_control) apart from the permanent layout it reads
detail    one per approach that asked for native-resolution crops in its `use` answer
addlane   a re-trace (retrace_prompt with `missing`) for an approach whose `use` answer reported a missing lane
leftlane  a re-trace (retrace_prompt with `no_left`) for an approach where no lane allows a left turn although an
          exit lies to its left
keepuse   `use` asked again (use_prompt `kept`) where it judged a traced lane not to exist and the independent
          reading does not count fewer lanes
The independent street-view reading (harness/views.reading_prompt) is blind: it sees neither the trace nor any reference.
Prompts carry only what the images show and measurable geometry; no lane count or reference is ever part of one.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from ..autoloop import render
from ..hybrid.common import require
from ..hybrid.geometry import font
from . import geometry
from . import views as street

KINDS = ("curb_or_pavement_edge", "lane_line", "centre_line", "median_edge", "bicycle_line", "buffer_or_parking_edge",
         "rail_edge", "unknown")
# An answer that spells the curb "kerb" (y_kerb, kerb_or_pavement_edge) is read under the curb names.
LEGACY_KINDS = {"kerb_or_pavement_edge": "curb_or_pavement_edge"}
SURFACES = ("motor_vehicle_lane", "shared_rail_motor_lane", "shared_bicycle_motor_lane", "bicycle_lane", "parking", "median_or_buffer",
            "shoulder", "sidewalk_or_verge", "rail_or_bus_only", "unknown")
SURFACE_TEXT = {"motor_vehicle_lane": "motor lane", "shared_rail_motor_lane": "motor lane shared with a streetcar",
                "shared_bicycle_motor_lane": "bicycle lane that motor traffic also uses", "bicycle_lane": "bicycle lane",
                "parking": "parking", "median_or_buffer": "median or buffer", "shoulder": "shoulder", "sidewalk_or_verge": "sidewalk or verge",
                "rail_or_bus_only": "rail or bus only", "unknown": "unknown strip"}
PAVING, AT_STOP = street.PAVING, street.AT_STOP
TEMPORARY = ("The network records the permanent layout. Cones, barrels, drums, delineator posts, temporary barriers, portable signs on stands"
             " or weighted bases and work-zone fencing are temporary traffic control: a lane closed or a turn banned only by them keeps its"
             " permanent existence and turns. Judge exists and turns from the permanent markings and fixed signs, and report any temporary"
             " control in temporary_control.")
SEEN = ("yes", "no", "uncertain")
SIDES = ("inbound", "outbound", "none")
CONFIDENCE = ("high", "medium", "low")
TURNS = ("left", "through", "right", "u_turn")
BASIS = ("observed", "inferred", "unknown")
EXISTS = ("yes", "no", "uncertain")
LEFT, TOP, BOTTOM, RIGHT = 46, 54, 32, 64
STATION, EDGE, PAPER = "#ff3df2", "#16d5ec", "#161a22"


def ruled_xy(raw, stations, scale, labels=None):
    """Crop at `scale` with x rulers (top, bottom) and a y ruler (right) in crop units, and dashed station lines."""
    body = raw.resize((round(raw.width * scale), round(raw.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (LEFT + body.width + RIGHT, TOP + body.height + BOTTOM), PAPER)
    canvas.paste(body, (LEFT, TOP))
    d = ImageDraw.Draw(canvas)
    base, edge = TOP + body.height, LEFT + body.width
    for x in range(0, raw.width + 1, 10):
        px, major = LEFT + x * scale, x % 50 == 0
        d.line((px, TOP - (9 if major else 4), px, TOP - 1), fill="white")
        d.line((px, base, px, base + (9 if major else 4)), fill="white")
        if major and x % 100 == 0:
            d.text((px, 2), str(x), font=font(16), fill="white", anchor="ma")
            d.text((px, base + 10), str(x), font=font(16), fill="white", anchor="ma")
    for y in range(0, raw.height + 1, 10):
        py, major = TOP + y * scale, y % 50 == 0
        d.line((edge, py, edge + (9 if major else 4), py), fill="white")
        if major and y % 100 == 0:
            d.text((edge + 12, py), str(y), font=font(15), fill="white", anchor="lm")
    for i, y in enumerate(stations):
        py = TOP + y * scale
        for x0 in range(LEFT, edge, 20):
            d.line((x0, py, min(x0 + 8, edge), py), fill=STATION, width=2)
        d.text((4, py), (labels or [f"S{k + 1}" for k in range(len(stations))])[i], font=font(16), fill=STATION, anchor="lm")
    return canvas


def side_by_side(images, gap=12, colour=PAPER):
    h = max(i.height for i in images)
    sheet = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), h), colour)
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width + gap
    return sheet


def trace_sheet(raw, window, scale, overlay=None, cameras=None, photos=None):
    """The arm crop with rulers and stations, plus a shade-lifted copy when shadow covers part of it.
    overlay: an earlier trace, drawn as dashed boundaries b1..bN for a re-trace.
    cameras: (views, to_pixel), marked on each panel; photos: [(image, caption)] in a column on the right."""
    panels = [raw]
    lifted = render.shade_lifted(raw) or render.lit(raw)
    if lifted is not None:
        panels.append(lifted)
    out = []
    for panel in panels:
        canvas = ruled_xy(panel, window["stations"], scale)
        if overlay:
            d = ImageDraw.Draw(canvas)
            for k, b in enumerate(overlay["boundaries"]):
                pts = [(LEFT + x * scale, TOP + y * scale) for x, y in zip(b["xs"], window["stations"]) if x is not None]
                for a, c in zip(pts, pts[1:]):
                    d.line((a, c), fill=EDGE, width=2)
                if pts:
                    d.text((pts[0][0], TOP - 26), f"b{k + 1}", font=font(14), fill=EDGE, anchor="ma")
            at = geometry.stop_line_at(overlay, window["stations"])
            if at is not None:  # the stop line across the inbound half, straight between its two ends
                xs_in = [x for k, s in enumerate(overlay["strips"]) if s["carriageway"] == "inbound" for x in
                         (geometry._first_x(overlay["boundaries"][k]["xs"], window["stations"], 0), geometry._first_x(overlay["boundaries"][k + 1]["xs"], window["stations"], 0))
                         if x is not None]
                x0, x1 = (min(xs_in), max(xs_in)) if xs_in else (panel.width / 2, panel.width)
                d.line((LEFT + x0 * scale, TOP + at(x0) * scale, LEFT + x1 * scale, TOP + at(x1) * scale), fill="#ffd400", width=2)
                d.text((LEFT + x0 * scale + 6, TOP + at(x0) * scale - 4), "stop line", font=font(14), fill="#ffd400", anchor="ld")
            y = overlay.get("exit_start_y")
            if y is not None:
                d.line((LEFT, TOP + y * scale, LEFT + panel.width * scale, TOP + y * scale), fill="#ffd400", width=2)
                d.text((LEFT + 6, TOP + y * scale - 4), "exit start", font=font(14), fill="#ffd400", anchor="ld")
        if cameras:
            canvas = street.mark_cameras(canvas, window, cameras[0], cameras[1], scale)
        out.append(canvas)
    if photos:
        out.append(street.column(photos, 600))
    return side_by_side(out), lifted is not None


def _stations_text(window):
    return ", ".join(f"S{i + 1} y={y:g} ({m:g} m from the junction centre)"
                     for i, (y, m) in enumerate(zip(window["stations"], window["stations_m"])))


def trace_prompt(window, shaded, site_name, photos=None):
    sections = window["sections"]
    has_in, has_out = "inbound" in sections, "outbound" in sections
    which = []
    if has_in:
        which.append(f"the inbound carriageway ({sections['inbound']['section_id']}: traffic entering the junction, moving UP, on the right of the road centre)")
    if has_out:
        which.append(f"the outbound carriageway ({sections['outbound']['section_id']}: traffic leaving the junction, moving DOWN, on the left of the road centre)")
    lifted = (" A second panel shows the same crop with its shaded parts lightened (colours there are not true; dark cars and new asphalt are"
              " lightened too) with the same rulers; use it to see paint in shadow.") if shaded else ""
    seen = (f" On the right are street views of the inbound approach ({'; '.join(photos)}); their cameras are marked in yellow on the crop"
            " with their field of view (positions from street-view metadata, off by a few metres at most). Use them to see lane lines,"
            " arrows, bicycle lanes and curbs that shadow or vehicles hide on the satellite. A look-back photo faces away from the"
            " junction: inbound traffic comes toward the camera, so the driver's left is on the RIGHT of that photo.") if photos else ""
    return f"""Trace the {window['arm']} arm of the signalised intersection {site_name}{(' (' + window['road_name'] + ')') if window.get('road_name') else ''}.
The image is a satellite crop rotated so that traffic ENTERING the junction on this arm moves UP. The crop starts at the junction centre, so the junction, its crosswalk and the stop bar lie near the TOP edge. Traffic keeps to the {window['driving_side']}.
It holds {' and '.join(which)}.{lifted}{seen}
Rulers give x (top and bottom) and y (right side) in crop units. Dashed magenta lines are stations: {_stations_text(window)}.

1. stop_line: where the stop bar crosses the inbound lanes (or, without a painted stop bar, the upstream edge of the crosswalk), as two y values: y_centre where it meets the road-centre side of the inbound lanes and y_curb where it meets their curb side. A stop bar square to the road has the same y at both ends; at a skewed junction they differ. Give it whenever this arm has an inbound carriageway: where paint is hidden, place it where vehicles wait or where the crosswalk begins, and say so in notes. null only if this arm has no inbound carriageway.
2. exit_start_y: the y where the outbound lanes begin after the junction (the downstream edge of the crosswalk). null if this arm has no outbound carriageway.
3. boundaries: every longitudinal line or edge across the whole road, listed left to right, from the outer curb of the left side to the outer curb of the right side: curbs or pavement edges, lane lines, centre lines, median edges, bicycle-lane lines, buffer or parking edges, and the edges of a track bed or rail-only strip that cars do not use. The rails of a streetcar or light-rail track laid in a traffic lane are not boundaries: that lane, from one lane line to the next, is one strip. For each give its kind and its x on every station; use null at a station where it does not exist (inside the junction, upstream of where a turn pocket opens, or where it ends). Where vehicles, shadow or worn paint hide a line that does exist, interpolate between its visible parts and name those stations in interpolated.
4. strips: one entry per pair of neighbouring boundaries (strip k lies between boundary k and k+1), in the same order: which carriageway it belongs to (inbound, outbound, none) and its surface_type where the lanes are read: at the stop line for an inbound strip, just past the exit start for an outbound one. {AT_STOP} So a strip that is parking upstream and a turn lane at the stop line is a motor_vehicle_lane, and a bicycle lane that turning motor traffic also uses at the stop line is a shared_bicycle_motor_lane; say in its evidence how the strip changes. A right-turn lane often lies between a bicycle lane and the curb near the stop bar; it is part of the carriageway. {PAVING} A concrete gutter pan (about 0.5-0.8 m wide, often lighter) runs along many curbs: its edge is not a lane line, and the gutter belongs to the curbside lane or shoulder. A lane with rails laid in it, which cars also drive in, is one shared_rail_motor_lane strip. A wide strip with no dividing line is one strip; do not invent lines that are not painted.

Do not count opposing traffic as inbound or outbound by its position alone: say which way it runs from arrows, stop bars, vehicles and the road centre.
Return JSON only:
{{"stop_line": {{"y_centre": 140.0, "y_curb": 140.0}}, "exit_start_y": 150.0,
 "boundaries": [{{"kind": "{'|'.join(KINDS)}", "xs": [null, 212.0, 211.5, ...], "interpolated": ["S6"], "visibility": "clear|partial|occluded"}}],
 "strips": [{{"carriageway": "inbound|outbound|none", "surface_type": "{'|'.join(SURFACES)}", "confidence": "high|medium|low", "evidence": "what shows it"}}],
 "notes": "what limited the trace"}}"""


def validate_trace(value, window, min_gap=4.0):
    require(isinstance(value, dict), "Expected a JSON object")
    stations, height = window["stations"], window["frame"].height
    width = window["frame"].width
    stop = value.get("stop_line")
    if isinstance(stop, dict) and "y_curb" not in stop and "y_kerb" in stop:
        stop["y_curb"] = stop.pop("y_kerb")
    for b in value.get("boundaries") or []:
        if isinstance(b, dict) and b.get("kind") in LEGACY_KINDS:
            b["kind"] = LEGACY_KINDS[b["kind"]]
    if "inbound" in window["sections"]:
        require(isinstance(stop, dict) and all(isinstance(stop.get(k), (int, float)) and 0 <= stop[k] < height for k in ("y_centre", "y_curb")),
                "stop_line must give y_centre and y_curb, each a y inside the crop")
    else:
        require(stop is None, "stop_line must be null: this arm has no inbound carriageway")
    y = value.get("exit_start_y")
    if "outbound" in window["sections"]:
        require(isinstance(y, (int, float)) and 0 <= y < height, "exit_start_y must be a y inside the crop")
    else:
        require(y is None, "exit_start_y must be null: this arm has no outbound carriageway")
    bounds = value.get("boundaries")
    require(isinstance(bounds, list) and len(bounds) >= 2, "At least two boundaries are needed")
    for b in bounds:
        require(b.get("kind") in KINDS, f"Unknown boundary kind {b.get('kind')}")
        xs = b.get("xs")
        require(isinstance(xs, list) and len(xs) == len(stations), f"Each boundary needs one x (or null) per station ({len(stations)})")
        require(all(x is None or (isinstance(x, (int, float)) and -1 <= x <= width + 1) for x in xs), "A boundary x leaves the crop")
        require(any(x is not None for x in xs), "A boundary must exist at one station at least")
    for i in range(len(stations)):
        row = [b["xs"][i] for b in bounds if b["xs"][i] is not None]
        require(all(b - a >= min_gap for a, b in zip(row, row[1:])), f"Boundaries cross, touch or are out of order at S{i + 1}")
    strips = value.get("strips")
    require(isinstance(strips, list) and len(strips) == len(bounds) - 1, "Give one strip per pair of neighbouring boundaries")
    for s in strips:
        require(s.get("carriageway") in SIDES, "carriageway must be inbound, outbound or none")
        require(s.get("surface_type") in SURFACES, f"Unknown surface_type {s.get('surface_type')}")
        require(s.get("confidence") in CONFIDENCE, "confidence must be high, medium or low")
    for side in ("inbound", "outbound"):
        if side not in window["sections"]:
            require(not any(s["carriageway"] == side for s in strips), f"This arm has no {side} carriageway")
    return value


def retrace_prompt(window, shaded, site_name, trace, found, reading, issues=(), disagrees=False, photos=None, missing=(), no_left=()):
    """A second look at an arm: its earlier trace drawn back, the measured issues, and the independent reading (only
    passed when it saw more lanes than the trace). missing: lanes the lane-use step saw at the stop bar with no strip of
    their own, as (where, evidence). no_left: the exits that lie to the approach's left (dicts with section_id, compass
    and turn_angle) when the lane-use step found no inbound lane that allows a left turn."""
    traced = len([r for r in found["inbound"] if r.get("region_id")])
    lines = []
    if missing:
        lines.append("A lane-use check of this approach, which saw your inbound lanes outlined on the satellite and projected into the street views,"
                     " reported general motor lane(s) at the stop bar that have no strip of their own:\n"
                     + "\n".join(f"- {where}: {evidence}" for where, evidence in missing)
                     + "\nAdd the boundary that separates such a lane where the pixels show it; keep the trace as it is where they do not.")
    if no_left:
        lines.append("The junction has an exit to the left of this approach ("
                     + "; ".join(f"{e['section_id']}, heading {e['compass']}, turning {e['turn_angle']:+.0f} deg" for e in no_left)
                     + "), but a lane-use check of this approach, which saw your inbound lanes outlined on the satellite and projected into the"
                       " street views, found no inbound lane that allows a left turn. Look again at the road-centre side of the inbound carriageway"
                       " for a left-turn lane or pocket that the trace leaves out, merges into a neighbouring strip or gives another surface type"
                       " (such as a median or buffer, or a rail- or bus-only strip). Where the pixels show such a lane at the stop line, give it"
                       " its own strip as a motor lane; where they show none (left turns may share a lane or be banned), keep the trace as it is.")
    if reading:
        lanes = " ".join("".join(t[0].upper() for t in l["turns"] if t != "u_turn") or "?" for l in reading["lanes"])
        extra = ", ".join(f"{reading.get(k, 0)} {n}" for k, n in (("bicycle_lanes", "bicycle"), ("rail_or_bus_only_lanes", "rail or bus only"),
                                                                  ("parking_lanes", "parking")) if reading.get(k))
        state = "a settled" if reading.get("count_settled") and reading.get("count_confidence") != "low" else "an unsettled, low-confidence"
        lines.append(f"A separate reading of this approach from its street views and an unmarked satellite crop, made without seeing your trace, gave"
                     f" {state} count of {len(reading['lanes'])} general motor lane(s) at the stop bar ({lanes}{'; also ' + extra if extra else ''});"
                     f" its own limits: {reading.get('limitations') or 'none stated'}."
                     + (" It differs from your trace, so one of the two is wrong." if disagrees else ""))
    if issues:
        lines.append("Measured from your trace:\n" + "\n".join(f"- {i}" for i in issues))
    projected = ("\nIn the street views on the right, your boundaries b1..bN are projected onto the road from each camera's recorded position and"
                 " heading (camera height 2.5 m assumed); the projection can sit up to about a lane to one side or skew with distance. Use it to"
                 " tell which painted line each boundary stands for: a lane in a photo with no strip of its own between two projected boundaries"
                 " means a missing boundary; a projected strip over a curb, gutter, sidewalk, bay or the opposing lanes means a spurious one.") if photos else ""
    return trace_prompt(window, shaded, site_name, photos) + f"""

This is a second look. The dashed cyan lines b1..bN on the crop are your earlier boundaries, and the yellow lines your stop line and exit start. The earlier trace gives the inbound carriageway {traced} general motor lane(s) at the stop line.
{chr(10).join(lines)}{projected}
Look again, using the lightened panel where shadow covers the road: a boundary may be missing, spurious or on the wrong side of the road centre, or a strip may hold the wrong surface. Return the complete corrected answer in the same format. Keep the trace where the pixels support it, even if {'it still differs from the separate reading or ' if reading else ''}a strip stays unusually wide or narrow."""


def approach_crop(image, window, found, margin_m=4.0, length_m=45.0):
    """Native-resolution crop of the inbound carriageway from just past the stop line upstream, and its box in arm-crop
    coordinates (left, top, right, bottom)."""
    regions = [r for r in found["inbound"] if r.get("region_id")]
    require(regions, "No inbound motor lanes to read")
    frame, mpp = window["frame"], window["mpp"]
    local = [frame.to_local(*p) for r in regions for p in r["polygon_px"]]
    left = max(0, min(x for x, _ in local) - margin_m / mpp)
    right = min(frame.width, max(x for x, _ in local) + margin_m / mpp)
    top = max(0, min(y for _, y in local) - margin_m / mpp)
    bottom = min(frame.height, top + (length_m + margin_m) / mpp)
    return render.crop(image, frame).crop((round(left), round(top), round(right), round(bottom))), (left, top, right, bottom)


def label_regions(canvas, polygons, offset, scale, colour="#ffd400", width=2, size=18):
    d = ImageDraw.Draw(canvas)
    for label, poly in polygons:
        pts = [((x - offset[0]) * scale, (y - offset[1]) * scale) for x, y in poly]
        if len(pts) >= 3:
            d.polygon(pts, outline=colour, width=width)
            cx = sum(p[0] for p in pts) / len(pts); cy = min(p[1] for p in pts) + 30
            box = d.textbbox((cx, cy), label, font=font(size), anchor="mm")
            d.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill="black")
            d.text((cx, cy), label, font=font(size), fill=colour, anchor="mm")
    return canvas


def use_sheet(image, window, found, overview_regions, node_px, mpp, photos, scale=2.0, overview_m=110.0, overview_px=1100):
    """Approach crop (travel up, lanes labelled r#), north-up junction overview (every exit's lanes labelled), and the
    approach's forward street views. Returns the sheet and the crop box used for detail requests."""
    crop, box = approach_crop(image, window, found)
    frame = window["frame"]
    lanes = [(r["region_id"], [frame.to_local(*p) for p in r["polygon_px"]]) for r in found["inbound"] if r.get("region_id")]
    big = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    panels = [label_regions(big.copy(), lanes, (box[0], box[1]), scale)]
    lifted = render.shade_lifted(crop) or render.lit(crop)
    if lifted is not None:
        lifted = lifted.resize(big.size, Image.Resampling.LANCZOS)
        panels.append(label_regions(lifted, lanes, (box[0], box[1]), scale))
    half = overview_m / 2 / mpp
    ov = image.crop((round(node_px[0] - half), round(node_px[1] - half), round(node_px[0] + half), round(node_px[1] + half)))
    k = overview_px / ov.width
    ov = ov.resize((overview_px, round(ov.height * k)), Image.Resampling.LANCZOS)
    shifted = [(label, [(x, y) for x, y in poly]) for label, poly in overview_regions]
    label_regions(ov, shifted, (node_px[0] - half, node_px[1] - half), k, colour="#38e8ff", width=2, size=15)
    panels.append(ov)
    column = photos
    if column:
        panels.append(street.column(photos, 560))
    titled = []
    titles = ["approach, traffic drives UP (lanes r#)"] + (["same, shadows lightened (colours not true)"] if lifted is not None else []) + \
             ["junction overview, north up (exit lanes)"] + (["street views of this approach"] if column else [])
    for title, panel in zip(titles, panels):
        cell = Image.new("RGB", (panel.width, panel.height + 34), PAPER)
        cell.paste(panel, (0, 34))
        ImageDraw.Draw(cell).text((8, 7), title, font=font(18), fill="white")
        titled.append(cell)
    return side_by_side(titled), {"crop": crop, "box": box, "overview": ov, "lifted": lifted is not None}


def cross_section(found):
    """The traced inbound strips at the stop line from the driver's left, lanes and the rest: 'r1 3.3 m | bicycle lane 1.7 m | r2 ...'."""
    parts = []
    for r in found["inbound"]:
        width = f" {r['width_m']:g} m" if r.get("width_m") is not None else ""
        parts.append(f"{r['region_id']}{width}" if r.get("region_id") else f"{SURFACE_TEXT.get(r['surface_type'], r['surface_type'])}{width} (no r#)")
    return " | ".join(parts)


def curb_wide(found, low_m=2.5):
    """Lanes r# that lie between a bicycle lane and the curb and measure at least low_m wide at the stop line."""
    out, after_bike = [], False
    for r in (found or {}).get("inbound", []):
        if r.get("surface_type") in ("bicycle_lane", "shared_bicycle_motor_lane") and not r.get("region_id"):
            after_bike = True
        elif after_bike and r.get("region_id") and (r.get("width_m") or 0) >= low_m:
            out.append(r)
    return out


def shared_inferred(answer):
    """Lanes an answer gives both left and through, inferred (not observed), while another lane also allows through."""
    kept = [l for l in answer["lanes"] if l.get("exists") != "no"]
    return [l["region_id"] for l in kept if {"left", "through"} <= set(l.get("turns") or []) and l.get("basis") != "observed"
            and any("through" in (x.get("turns") or []) for x in kept if x is not l)]


def wide_bicycle(found, low_m=2.5):
    """Bicycle strips (no r#) at least low_m wide that lie between two lanes r#: (width, left r#, right r#)."""
    regs = (found or {}).get("inbound", [])
    return [(r["width_m"], regs[i - 1]["region_id"], regs[i + 1]["region_id"]) for i, r in enumerate(regs)
            if r.get("surface_type") == "bicycle_lane" and not r.get("region_id") and (r.get("width_m") or 0) >= low_m
            and 0 < i < len(regs) - 1 and regs[i - 1].get("region_id") and regs[i + 1].get("region_id")]


def use_prompt(direction, site_name, lanes, exits, reading, captions, shaded, projected=True, found=None, kept=(), kept_why="",
               left_note=(), low_m=2.5, wide_note=(), shared_note=()):
    """kept: region IDs an earlier answer judged not to exist that stay lanes because the independent reading does not
    count fewer lanes; kept_why: one sentence on what that reading gave. left_note: exits to the left that no lane served
    and that a left-lane re-trace found no lane for (step 8): the answer is asked to look at the leftmost lane again.
    A lane between a bicycle lane and the curb that measures at least low_m wide gets a note on curbside parking; a bicycle
    strip at least low_m wide between two lanes gets a note to check it for a lane. wide_note: (r#, width) of lanes an
    earlier answer judged not to exist although they measure as wide as a motor lane (step 7b)."""
    wide = curb_wide(found, low_m)
    names = ", ".join(f"{r['region_id']} ({r['width_m']:g} m)" for r in wide)
    curb_text = (f"\n{names}"
                 f" {'lies' if len(wide) == 1 else 'lie'} between a bicycle lane and the curb and {'is' if len(wide) == 1 else 'are'} as wide as a motor lane"
                 f" ({low_m:g} m or more at the stop line). Curbside parking ends short of a junction, so such a strip is normally where right-turning"
                 " traffic waits at the stop bar, even where cars park in it upstream and no arrow is painted. Judge it not a general motor lane only"
                 " if something at the stop bar keeps cars out of it (a curb extension or island, posts, a bus stop, hatching, or a stop bar that"
                 " ends short of it), and say what.") if wide else ""
    bike_text = "".join(f"\nThe trace has a bicycle lane {wb:g} m wide between {ra} and {rb}, wider than a bicycle lane usually is: if a"
                        " general motor lane (for example a turn lane) lies within that strip at the stop bar, report it in missing_lanes;"
                        " otherwise leave it." for wb, ra, rb in wide_bicycle(found, low_m))
    one_w = len(wide_note) == 1
    wide_text = (f"\nAn earlier answer judged {', '.join(f'{rid} ({wd:g} m)' for rid, wd in wide_note)} not to be"
                 f" {'a general motor lane' if one_w else 'general motor lanes'}, but {'it is' if one_w else 'they are'} as wide as a motor lane"
                 f" at the stop line ({low_m:g} m or more). Keep {'it' if one_w else 'them'} as {'a lane' if one_w else 'lanes'} unless something at the"
                 " stop bar keeps cars out (a curb extension or island, posts, a bus stop, hatching, or a stop bar that ends short of it); if so,"
                 " say what.") if wide_note else ""
    one_s = len(shared_note) == 1
    shared_text = (f"\nAn earlier answer gave {', '.join(shared_note)} both left and through, inferred rather than read from an arrow or sign,"
                   " while other lanes also carry through traffic. Where an approach has more than one through lane, the lane on the"
                   " road-centre side that takes left turns is usually a left-turn lane, and a left-turn lane may lie on the road-centre side"
                   " outside the outlines, on paving that looks like a median. Look again: give"
                   f" {'it' if one_s else 'them'} only the turns that arrows, signs or position show, and report in missing_lanes a left-turn"
                   " lane that has no outline of its own.") if shared_note else ""
    left_text = ("\nAn exit lies to the left of this approach (" + "; ".join(f"{e['section_id']}, heading {e['compass']}" for e in left_note)
                 + "), but an earlier answer gave no lane a left turn, and a second trace found no further lane on the road-centre side."
                 " Look again at the leftmost lane: say whether it also allows left turns (for example a lane shared by left and through"
                 " traffic) or whether a sign or marking bans left turns here, and give left where nothing rules it out.") if left_note else ""
    lane_text = "\n".join(f"- {r['region_id']}: lane {r['motor_lane_index']} from the driver's left, about {r['width_m']} m wide"
                          + (f", traced as a {SURFACE_TEXT[r['surface_type']]}" if r.get("surface_type") not in (None, "motor_vehicle_lane") else "")
                          for r in lanes)
    if found:
        lane_text += f"\nThe trace across the approach at the stop line, from the driver's left: {cross_section(found)}."
    exit_text = "\n".join(f"- {e['section_id']}: traffic leaving the junction heading {e['compass']}, turning {e['turn_angle']:+.0f} deg from this approach"
                          f" ({e['geometric_hint']}); lanes {', '.join(e['regions']) or 'none traced'} from its driver's left (the road centre side)"
                          for e in exits)
    turns = " ".join("".join(t[0].upper() for t in l["turns"] if t != "u_turn") or "?" for l in reading["lanes"]) if reading else None
    seen = (f"A separate reading of this approach from its street views (made without seeing the lanes r#) counted {len(reading['lanes'])}"
            f" general motor lane(s) with arrows {turns} from the driver's left; its limits: {reading.get('limitations') or 'none stated'}."
            " Use it for arrows and signs; where it counts fewer lanes than the r#, judge each lane from the images, not from that count.") if reading else \
        "No separate street-view reading is available."
    one = len(kept) == 1
    kept_text = (f"\nAn earlier answer judged {', '.join(kept)} not to be {'a general motor lane' if one else 'general motor lanes'} at the"
                 f" stop bar. {kept_why} So nothing independent of the outlines confirms that, and {'it stays a lane' if one else 'they stay lanes'}:"
                 f" treat {'it' if one else 'them'} as {'a lane' if one else 'lanes'} (exists yes, or uncertain where the images cannot settle it)"
                 f" and give the turns {'it allows' if one else 'each allows'}.") if kept else ""
    return f"""Read lane use and turning movements for the {direction} approach of the signalised intersection {site_name}.
The image shows: the approach (satellite, traffic drives UP toward the junction, its general motor lanes outlined in yellow and labelled r#){', the same with shadows lightened' if shaded else ''}; a north-up overview of the junction with the lanes of every exit outlined in cyan and labelled exit:r#; and street views of this approach ({'; '.join(captions) or 'none'}){' with the lanes r# outlined in yellow where they project onto the road from each camera (camera height assumed; the outline can sit up to about a lane to one side)' if projected else ''}. A look-back photo faces away from the junction: the approach's traffic comes toward the camera, so the driver's left is on the RIGHT of that photo.
Approach lanes:
{lane_text}
Exits:
{exit_text}
{seen}{kept_text}{curb_text}{bike_text}{wide_text}{shared_text}{left_text}

1. For every lane r# first say whether it exists as a general motor lane at the stop bar (yes; no: it is really a gutter, shoulder, parking, bay, a bicycle lane that motor traffic does not use there, sidewalk or part of another lane; uncertain), judged from the satellite and the street views. {AT_STOP} {PAVING} Then give the turns it allows (left, through, right, u_turn; none for a lane that does not exist), the basis (observed: an arrow, ONLY word or sign read for this lane; inferred: from lane order, lane lines or the junction layout; unknown) and the evidence. Bind a marking to a lane by its position in the images, not by vehicle positions alone. The curbside lane may also turn right unless a permanent marking or fixed sign rules it out; this applies only to a strip that is a lane. {TEMPORARY}
2. For each turn (left, through, right, u_turn) name the exit it leads to (or none), whether you propose it (candidate), cannot settle it (unresolved) or see no such movement (not_proposed), and its lane pairs: which approach lane feeds which exit lane. A lane that allows a turn must appear in that turn's pairs, and only such lanes may. A pair names lanes by their IDs (approach r#, exit r#).
3. If a small arrow, word or sign could settle a lane's use, ask for up to three native-resolution crops in detail_requests (source: approach, overview, or a street-view ID; bbox_xyxy as 0-1 fractions of that panel). Otherwise leave detail_requests empty.
4. missing_lanes: a general motor lane you see at the stop bar that has no r# of its own (for example a turn pocket or a curbside turn lane the outlines leave out, or two lanes inside one outline): where it lies (for example "between r4 and the curb") and what shows it. Leave it empty when every lane has its own r#.
5. temporary_control: whether you see temporary traffic control on this approach (yes, no, uncertain), which r# it affects, what it does to them (for example closes a lane or bans a turn for now), and what shows it in which image. The lanes and turns above stay the permanent ones.
Return JSON only:
{{"lanes": [{{"region_id": "r1", "exists": "yes|no|uncertain", "turns": ["left"], "basis": "{'|'.join(BASIS)}", "confidence": "high|medium|low", "evidence": "..."}}],
 "movements": [{{"turn": "left", "to": "out_NB", "status": "candidate|unresolved|not_proposed", "lane_pairs": [["r1", "r1"]], "reason": "..."}}],
 "detail_requests": [{{"source": "approach", "bbox_xyxy": [0.1, 0.0, 0.4, 0.3], "why": "..."}}],
 "missing_lanes": [{{"where": "between r4 and the curb", "evidence": "..."}}],
 "temporary_control": {{"seen": "yes|no|uncertain", "lanes": ["r2"], "effect": "...", "evidence": "..."}},
 "notes": "..."}}"""


def validate_use(value, lanes, exits, sources, allow_details=True, must_exist=(), keep_all=False):
    """must_exist: lanes that must exist and allow a turn; keep_all: no lane may be judged not to exist."""
    require(isinstance(value, dict), "Expected a JSON object")
    ids = [r["region_id"] for r in lanes]
    got = value.get("lanes")
    require(isinstance(got, list) and [l.get("region_id") for l in got] == ids, f"Give lanes for exactly {ids}, in order")
    for lane in got:
        require(lane.get("exists") in EXISTS, "exists must be yes, no or uncertain")
        require(isinstance(lane.get("turns"), list) and set(lane["turns"]) <= set(TURNS), "turns must be left, through, right, u_turn")
        require(lane["exists"] != "no" or not lane["turns"], f"{lane.get('region_id')} does not exist, so it allows no turn")
        require(lane["region_id"] not in must_exist or (lane["exists"] != "no" and lane["turns"]),
                f"{lane['region_id']} stays a lane (nothing independent confirms it is not one): give exists yes or uncertain, and the turns it allows")
        require(not keep_all or lane["exists"] != "no",
                f"Every lane stays a lane in this answer: give {lane['region_id']} exists yes or uncertain")
        require(lane.get("basis") in BASIS, "basis must be observed, inferred or unknown")
        require(lane.get("confidence") in CONFIDENCE, "confidence must be high, medium or low")
    exit_ids = {e["section_id"]: set(e["regions"]) for e in exits}
    moves = value.get("movements")
    require(isinstance(moves, list) and sorted(m.get("turn") for m in moves) == sorted(TURNS), "Give one movement per turn: left, through, right, u_turn")
    allowed = {l["region_id"]: set(l["turns"]) for l in got}
    for m in moves:
        require(m.get("status") in ("candidate", "unresolved", "not_proposed"), "status must be candidate, unresolved or not_proposed")
        pairs = m.get("lane_pairs") or []
        require(isinstance(pairs, list) and all(isinstance(p, list) and len(p) == 2 for p in pairs), "lane_pairs must be [approach r#, exit r#] pairs")
        if m["status"] == "candidate":
            require(m.get("to") in exit_ids, f"A candidate {m['turn']} must name one of the exits {sorted(exit_ids)}")
            require(pairs, f"A candidate {m['turn']} needs lane pairs")
            for a, b in pairs:
                require(a in allowed, f"Unknown approach lane {a}")
                require(b in exit_ids[m["to"]], f"Unknown lane {b} of {m['to']}")
                require(m["turn"] in allowed[a], f"{a} feeds the {m['turn']} but does not allow it")
            fed = {a for a, _ in pairs}
            missing = [r for r, t in allowed.items() if m["turn"] in t and r not in fed]
            require(not missing, f"{', '.join(missing)} allow(s) {m['turn']} but feed(s) no exit lane in it")
        else:
            require(not pairs, f"Only a candidate movement has lane pairs ({m['turn']})")
    for r, t in allowed.items():
        for turn in t:
            require(any(m["turn"] == turn and m["status"] in ("candidate", "unresolved") for m in moves),
                    f"{r} allows {turn}, but that movement is not_proposed")
    missing = value.get("missing_lanes") or []
    require(isinstance(missing, list) and len(missing) <= 3 and all(isinstance(m, dict) and isinstance(m.get("where"), str) and m["where"].strip()
                                                                    and isinstance(m.get("evidence"), str) for m in missing),
            "missing_lanes must be a list of at most three {where, evidence} entries (empty when every lane has an r#)")
    temporary = value.get("temporary_control")
    require(isinstance(temporary, dict) and temporary.get("seen") in SEEN and isinstance(temporary.get("lanes", []), list)
            and set(temporary.get("lanes") or []) <= set(ids) and all(isinstance(temporary.get(k, ""), str) for k in ("effect", "evidence")),
            f"temporary_control must be {{seen: yes|no|uncertain, lanes: r# from {ids}, effect, evidence}}")
    details = value.get("detail_requests") or []
    require(isinstance(details, list) and len(details) <= (3 if allow_details else 0),
            "At most three detail requests" if allow_details else "No further detail requests: answer from what you have")
    for q in details:
        require(q.get("source") in sources, f"Unknown detail source {q.get('source')}; use one of {sorted(sources)}")
        box = q.get("bbox_xyxy")
        require(isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in box)
                and box[2] > box[0] and box[3] > box[1], "bbox_xyxy must be 0-1 fractions [x0, y0, x1, y1] with x1>x0 and y1>y0")
    return value


def detail_sheet(requests, sources, scale=3, max_side=900):
    cells = []
    for q in requests:
        image = sources[q["source"]]
        x0, y0, x1, y1 = q["bbox_xyxy"]
        piece = image.crop((round(x0 * image.width), round(y0 * image.height), round(x1 * image.width), round(y1 * image.height)))
        k = min(scale, max_side / max(piece.width, piece.height, 1))
        piece = piece.resize((max(1, round(piece.width * k)), max(1, round(piece.height * k))), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (max(piece.width, 300), piece.height + 30), PAPER)
        cell.paste(piece, (0, 30))
        ImageDraw.Draw(cell).text((6, 6), f"{q['source']} {q['bbox_xyxy']}", font=font(15), fill="white")
        cells.append(cell)
    return side_by_side(cells)


def detail_prompt(use_text, previous):
    import json
    return use_text + f"""

This is a second look. The image now shows the native-resolution crops you asked for, side by side, each titled with its source and box. Your earlier answer was:
{json.dumps(previous, ensure_ascii=False)}
Return the complete answer again in the same format, revised where the crops change it, with detail_requests empty."""


def turn_angle(approach_heading, exit_heading):
    """Signed turn from an approach's travel heading to an exit's (degrees, + = right, in (-180, 180])."""
    return (exit_heading - approach_heading + 180) % 360 - 180


def geometric_hint(angle):
    """Measurable fact about an exit relative to an approach; the model decides which turn it is."""
    a = abs(angle)
    if a <= 30:
        return "about straight ahead"
    if a >= 150:
        return "back the way the approach came"
    return ("to the right" if angle > 0 else "to the left") + (" at a sharp angle" if a > 120 else "")


def compass(heading):
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][round(heading / 45) % 8]
