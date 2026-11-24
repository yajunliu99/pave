"""The Read view's data, made from a run of read/scripts/pipeline/run_read.py and its imagery folder.

read_view(run, imagery, title, base) returns the two parts of the Read view in pave_template.html:
- the index (one entry per junction: name, position, approaches with their lanes, turns; the traced lanes in lon/lat
  for the map), and
- one JSON block per junction, parsed only when the junction is selected: the satellite image with every traced
  region, the street views, the checks, the figures, and the images themselves (AVIF, inline).
"""
import base64
import csv
import io
import json
import math
from pathlib import Path

from PIL import Image

TURN = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}
ORDER = ["NB", "SB", "EB", "WB", "NE", "SW", "NW", "SE"]
SAT = 1600            # satellite image width shown
VIEW_MAX = 480        # street views are stored at most this wide
QUALITY = {"sat": 50, "view": 45, "figure": 50}   # AVIF quality per kind of image
STEP_NAMES = {"trace": "Trace", "read": "Blind reading", "retrace": "Re-trace", "use": "Lane use", "detail": "Detail crops"}
R_EARTH = 6378137.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def avif(path, width, quality):
    """(data URI of an AVIF copy at most `width` wide, its size). A Pillow without AVIF writes WebP instead."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        try:
            image.save(buffer, "AVIF", quality=quality, speed=6)
            kind = "avif"
        except (KeyError, OSError):
            buffer = io.BytesIO()
            image.save(buffer, "WEBP", quality=quality + 25)
            kind = "webp"
        return f"data:image/{kind};base64," + base64.b64encode(buffer.getvalue()).decode(), list(image.size)


def mercator(lon, lat):
    return R_EARTH * math.radians(lon), R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def lonlat(x, y):
    return math.degrees(x / R_EARTH), math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2)


def turns_text(turns):
    return "".join(TURN[t] for t in ("left", "through", "right", "u_turn") if t in (turns or []))


def region_label(region, inbound):
    """(label, style) of a traced region: turns and lane number for a motor lane, else its facility type."""
    kind = region["surface_type"]
    if kind == "bicycle_lane":
        return "bike", "bike"
    if kind in ("median_or_buffer", "parking"):
        return "buffer" if kind == "median_or_buffer" else "parking", "other"
    if kind != "motor_vehicle_lane":
        return kind.replace("_", " "), "other"
    if region.get("existence") != "supported":
        return "lane?", "unsure"
    n = region.get("motor_lane_index") or ""
    if not inbound:
        return f"out{n}", "exit"
    use = region.get("lane_use_prediction") or {}
    letters = turns_text(use.get("allowed_turns"))
    return (letters if use.get("basis") == "observed" else letters.lower()) + str(n), "motor"


def node_position(base, site):
    node_csv = Path(base) / site["network"]["node_csv"]   # an absolute path stays as it is
    with open(node_csv, newline="", encoding="utf-8-sig") as stream:
        row = next(r for r in csv.DictReader(stream) if int(r["node_id"]) == int(site["node_id"]))
    return float(row["x_coord"]), float(row["y_coord"])


def read_view(run, imagery, title, base):
    """`base` is the folder the imagery's own paths (site.json, base_config.json) are relative to: read/."""
    run, imagery = Path(run), Path(imagery)
    site, config = load(imagery / "site.json"), load(imagery / "base_config.json")
    lanes, movements = load(run / "lanes.json"), load(run / "movements.json")
    traces, readings, summary = load(run / "traces.json"), load(run / "readings.json"), load(run / "summary.json")
    node = int(site["node_id"])
    img = {}

    # satellite: pixel <-> lon/lat through the viewport's Web Mercator bounds
    sat_source = config["sources"]["satellite"]
    viewport = sat_source["georeference"]["viewport"]
    x0, y0, x1, y1 = viewport["mercator_bounds"]
    to_lonlat = lambda px, py: lonlat(x0 + px / viewport["width"] * (x1 - x0), y1 - py / viewport["height"] * (y1 - y0))
    img["sat.avif"], sat_size = avif(imagery / "satellite.jpg", SAT, QUALITY["sat"])
    scale = sat_size[0] / viewport["width"]

    # traced regions
    regions, geo = [], []
    for section in lanes:
        inbound = section["kind"] == "inbound_stopbar"
        others = 0
        for r in section["regions"]:
            label, style = region_label(r, inbound)
            if r.get("region_id"):
                rid = r["region_id"]
            else:
                others += 1
                rid = f"x{others}"
            use = r.get("lane_use_prediction") or {}
            regions.append({"id": f"{section['section_id']}/{rid}", "section": section["section_id"], "region": rid,
                            "inbound": inbound, "label": label, "style": style, "type": r["surface_type"],
                            "existence": r.get("existence"), "lane": r.get("motor_lane_index"), "width": r.get("width_m"),
                            "turns": use.get("allowed_turns") or [], "basis": use.get("basis"),
                            "use_confidence": use.get("confidence"), "use_evidence": use.get("evidence", ""),
                            "trace_confidence": r.get("trace_confidence"), "evidence": r.get("trace_evidence", ""),
                            "poly": [[round(x * scale, 1), round(y * scale, 1)] for x, y in r["polygon_px"]]})
            ring = [[round(v, 7) for v in to_lonlat(x, y)] for x, y in r["polygon_px"]]
            geo.append([section["section_id"], label, style, ring + ring[:1]])

    # street views, placed on the satellite by their camera position
    views, view_dates = [], set()
    for sid, s in sorted(config["sources"].items()):
        if s.get("role") != "gsv":
            continue
        name = f"v{len(views)}.avif"
        img[name], size = avif(imagery / "street_views" / Path(s["path"]).name, VIEW_MAX, QUALITY["view"])
        mx, my = mercator(s["actual_lon"], s["actual_lat"])
        xy = [round((mx - x0) / (x1 - x0) * sat_size[0], 1), round((y1 - my) / (y1 - y0) * sat_size[1], 1)]
        view_dates.add(s.get("capture_date"))
        views.append({"id": sid, "src": name, "size": size, "group": f"Approach {s['direction']}",
                      "role": "looking back" if s.get("reverse_view") else "looking ahead",
                      "position": s.get("sampling_position") or "", "date": s.get("capture_date"),
                      "dist": round(s.get("distance_to_center_m") or 0, 1), "heading": round(s.get("compass_heading_deg") or 0),
                      "xy": xy})
    views.sort(key=lambda v: (v["group"], v["role"] != "looking back", v["dist"]))

    # approaches: the traced lanes with their turns, beside the blind reading
    roads = {sec["section_id"]: sec.get("road_name") for t in traces.values() for sec in t["sections"].values()}
    approaches = []
    for section in (s for s in lanes if s["kind"] == "inbound_stopbar"):
        d = section["direction"]
        motor = sorted((r for r in section["regions"] if r["surface_type"] == "motor_vehicle_lane" and r.get("existence") == "supported"),
                       key=lambda r: r["motor_lane_index"])
        vlm = []
        for r in motor:
            use = r.get("lane_use_prediction") or {}
            vlm.append({"n": r["motor_lane_index"], "t": turns_text(use.get("allowed_turns")),
                        "basis": use.get("basis") or "unknown", "region": r["region_id"]})
        reading = readings["readings"].get(d) or {}
        blind = [{"n": l["lane"], "t": turns_text(l.get("turns"))} for l in reading.get("lanes", [])]
        approaches.append({"dir": d, "key": d, "road": roads.get(section["section_id"]), "arm": section.get("arm"),
                           "link": section.get("link_id"), "vlm": vlm, "blind": blind,
                           "blind_settled": reading.get("count_settled"), "blind_confidence": reading.get("count_confidence"),
                           "bike": sum(r["surface_type"] == "bicycle_lane" for r in section["regions"])})
    approaches.sort(key=lambda a: ORDER.index(a["dir"]) if a["dir"] in ORDER else 99)

    # turning movements, grouped by approach
    rows = []
    for m in sorted(movements, key=lambda m: (ORDER.index(m["from_direction"]) if m["from_direction"] in ORDER else 99,
                                              "LTRU".index(TURN[m["turn"]]))):
        ins = sorted({p["ib_lane"] for p in m["lane_pairs"]})
        outs = sorted({p["ob_lane"] for p in m["lane_pairs"]})
        rows.append({"code": m["from_direction"] + TURN[m["turn"]], "group": m["from_direction"], "to": m.get("to_direction"),
                     "status": m["status"], "lanes": len(ins), "in": [ins[0], ins[-1]] if ins else None,
                     "out": [outs[0], outs[-1]] if outs else None, "reason": m.get("reason", "")})

    # checks: the blind reading against the trace, the further steps, temporary traffic control
    checks = {"requests": summary.get("answers_by_stage", {}), "approaches": {}}
    for a in approaches:
        d = a["dir"]
        reading = readings["readings"].get(d) or {}
        checks["approaches"][d] = {"reading": readings["checks"].get(d) or {}, "limitations": reading.get("limitations", ""),
                                   "widths": [[r["lane"], r["width"]] for r in regions if r["section"] == f"in_{d}" and r["style"] == "motor"],
                                   "addlane": summary.get("addlane", {}).get(d), "removal": summary.get("removals", {}).get(d),
                                   "leftlane": summary.get("leftlane", {}).get(d),
                                   "temporary": summary.get("temporary_control", {}).get(d)}

    # figures: the result, and every image sent to the model
    figures = {"result": None, "sheets": []}
    if (run / "result.jpg").exists():
        img["result.avif"], size = avif(run / "result.jpg", 1600, QUALITY["figure"])
        figures["result"] = {"src": "result.avif", "size": size}
    sheets = sorted(f for f in (run / "sheets").glob("*.jpg") if not f.name.startswith("arm_"))
    order = list(STEP_NAMES)
    for f in sorted(sheets, key=lambda f: (order.index(f.stem.split("_")[0]) if f.stem.split("_")[0] in order else 99, f.stem)):
        step, place = f.stem.split("_", 1)
        name = f"sheet_{f.stem}.avif"
        img[name], size = avif(f, 2000, QUALITY["figure"])
        figures["sheets"].append({"src": name, "size": size, "step": STEP_NAMES.get(step, step.title()), "place": place})

    lon, lat = node_position(base, site)
    dates = sorted(d for d in view_dates | {sat_source.get("capture_date")} if d)
    motor_lanes = sum(len(a["vlm"]) for a in approaches)
    entry = {"id": node, "name": site.get("name") or f"node {node}", "node": node, "lat": lat, "lon": lon, "legs": len(traces),
             "dates": [dates[0], dates[-1]] if dates else None, "approaches": approaches, "movements": rows,
             "satellite": sat_source.get("provider")}
    index = {"title": title, "sites": [entry], "geo": {node: geo},
             "metrics": [[str(motor_lanes), f"motor lanes on {len(approaches)} approaches"],
                         [str(sum(m["status"] == "candidate" for m in movements)), "turns proposed"],
                         [str(sum(checks["requests"].values())), "model requests"]]}
    data = {"id": node, "sat": {"src": "sat.avif", "size": sat_size}, "regions": regions, "views": views, "checks": checks,
            "figures": figures, "img": img}
    return index, {node: data}
