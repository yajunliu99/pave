"""Blind check of approaches whose VLM lane count differs from a reference (UTDF).

For each such approach the model sees the unmarked satellite crop and the mid and near forward street views,
and reads the lane layout at the stop bar on its own: it is told neither the VLM result nor the reference.
Code then compares its reading with both. The check is recorded beside the run; it never changes the lane
predictions, and the reference is never shown to a model.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from ..hybrid.common import require, resource_path
from ..hybrid.geometry import font

TURNS = ("left", "through", "right", "u_turn")
LETTER = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}
UTDF_TURN = {"left": "L", "thru": "T", "right": "R", "uturn": "U"}
CONFIDENCE = ("high", "medium", "low")


def reference_lanes(movement_csv):
    """Lane-by-lane turn letters of every reference approach: {incoming link: {approach name: letters}} (lane 1 = leftmost).

    A link can carry two reference approaches where the reference splits a junction differently from the network."""
    lanes = defaultdict(lambda: defaultdict(dict))
    with open(movement_csv, newline="", encoding="utf-8-sig") as stream:
        for r in csv.DictReader(stream):
            if not r.get("utdf_intid") or r.get("type") not in UTDF_TURN or not r.get("start_ib_lane") or not r.get("ib_link_id"):
                continue
            name = (r.get("utdf_mvmt") or r.get("mvmt_txt_id") or "")[:2]
            for lane in range(int(float(r["start_ib_lane"])), int(float(r["end_ib_lane"] or r["start_ib_lane"])) + 1):
                lanes[r["ib_link_id"]][name].setdefault(lane, set()).add(UTDF_TURN[r["type"]])
    return {link: {name: ["".join(t for t in "LTRU" if t in by[n]) for n in sorted(by)] for name, by in named.items()}
            for link, named in lanes.items()}


def reference_name(names, key):
    """Reference approach compared with network approach `key` on its link: the one of the same name, else the first."""
    names = sorted(names)
    return key if key in names else names[0] if names else None


def disagreements(config, lanes, reference):
    """Approaches whose VLM motor-lane count differs from the reference, with both layouts."""
    sections = {s["section_id"]: s for s in lanes}
    out = []
    for key, leg in config.get("legs", {}).items():
        named = reference.get(str(leg.get("in_link_id")), {})
        name = reference_name(named, key)
        section, ref = sections.get(f"in_{key}"), named.get(name)
        if section is None or not ref or section["count"]["model_motor_count"] == len(ref):
            continue
        out.append({"approach": key, "link": str(leg["in_link_id"]), "vlm_count": section["count"]["model_motor_count"],
                    "reference_name": name, "reference": ref, "reference_count": len(ref),
                    "reference_also_on_link": sorted(set(named) - {name})})
    return out


def street_views(config, approach, root):
    """Mid then near forward view of the approach: (image path, caption without any lane information)."""
    out = []
    for position in ("mid", "near"):
        view = next((v for v in config.get("context_views", []) if v["direction"] == approach and v.get("sampling_position") == position), None)
        if view:
            source = config["sources"][view["source_id"]]
            if source["kind"] != "image":
                continue  # panorama sources are rendered by the evidence stage; its rendered copy is used instead
            path = resource_path(root, source.get("path") or source["pano_path"])
            out.append((path, f"{position}: {source.get('distance_to_center_m')} m from the centre, {source.get('capture_date') or 'date unknown'}"))
    return out


def fit(image, height):
    return image.resize((round(image.width * height / image.height), height), Image.Resampling.LANCZOS)


def blind_sheet(satellite_crop, street_views, path):
    """Satellite crop (travel up) on the left, street views stacked on the right; captions carry no lane answer."""
    satellite = fit(Image.open(satellite_crop).convert("RGB"), 900)
    views = [(fit(Image.open(p).convert("RGB"), 445), caption) for p, caption in street_views]
    sheet = Image.new("RGB", (satellite.width + 10 + 445, 900 + 34), "#11151c")
    d = ImageDraw.Draw(sheet)
    d.text((8, 7), "satellite, traffic on this approach drives UP", font=font(18), fill="white")
    sheet.paste(satellite, (0, 34))
    for i, (view, caption) in enumerate(views):
        sheet.paste(view, (satellite.width + 10, 34 + i * 455))
        d.text((satellite.width + 18, 40 + i * 455), caption, font=font(16), fill="white", stroke_width=2, stroke_fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


def check_prompt(captions):
    return f"""This image shows one approach to a signalised intersection.
Left: a satellite crop rotated so that traffic on this approach drives UP toward the intersection; the stop bar and crosswalk are near the top.
Right: street views taken before the intersection, looking in the direction of travel ({'; '.join(captions) or 'none available'}). A street view may be old, taken from the opposite carriageway, or partly blocked; say so when it is.
Read the lane layout of THIS approach at the stop bar, counting from the driver's left.
- Count only lanes open to general motor traffic in the direction of travel. Turn pockets count; a lane shared with a streetcar counts.
- Do not count opposing lanes, medians, bicycle lanes, rail- or bus-only lanes, parking or shoulders; report those separately.
- For each counted lane give the movements its pavement arrows allow (left, through, right, u_turn), or an empty list if no arrow is legible.
- If shadow, occlusion, image age or camera position prevents a reliable count, set count_settled to false instead of guessing.
Return JSON only:
{{"lanes":[{{"lane":1,"turns":["left"],"evidence":"what shows this lane and its arrows"}}],"bicycle_lanes":0,"rail_or_bus_only_lanes":0,"parking_lanes":0,"count_confidence":"high|medium|low","count_settled":true,"limitations":"what limited the reading"}}"""


def validate_reading(value):
    require(isinstance(value, dict) and isinstance(value.get("lanes"), list), "lanes[] required")
    numbers = [l.get("lane") for l in value["lanes"]]
    require(numbers == list(range(1, len(numbers) + 1)), "Number the lanes 1, 2, ... from the driver's left")
    for lane in value["lanes"]:
        require(isinstance(lane.get("turns"), list) and set(lane["turns"]) <= set(TURNS), "turns must be left, through, right, u_turn")
        require(isinstance(lane.get("evidence"), str), "Each lane needs evidence text")
    for key in ("bicycle_lanes", "rail_or_bus_only_lanes", "parking_lanes"):
        require(isinstance(value.get(key, 0), int) and value.get(key, 0) >= 0, f"{key} must be a non-negative integer")
    require(value.get("count_confidence") in CONFIDENCE, "count_confidence must be high, medium or low")
    require(isinstance(value.get("count_settled"), bool), "count_settled must be true or false")
    return value


def layout(reading):
    lanes = " ".join("".join(LETTER[t] for t in TURNS if t in l["turns"]) or "?" for l in reading["lanes"])
    extras = [f"+{reading.get(k, 0)} {name}" for k, name in (("bicycle_lanes", "bike"), ("rail_or_bus_only_lanes", "rail/bus"), ("parking_lanes", "parking"))
              if reading.get(k, 0)]
    return " ".join([lanes] + extras)


def verdict(reading, vlm_count, reference):
    """Which of the VLM and the reference the blind reading supports, and the likely reason."""
    if not reading["count_settled"] or reading["count_confidence"] == "low":
        return "unclear", "imagery_not_settled"
    seen = len(reading["lanes"])
    if seen == len(reference) and seen != vlm_count:
        if seen > vlm_count:
            right = reading["lanes"][-1]["turns"] == ["right"] or reference[-1] == "R"
            return "utdf", "model_missed_right_lane" if right else "model_missed_lanes"
        return "utdf", "model_extra_regions"
    if seen == vlm_count and seen != len(reference):
        return "model", "utdf_differs_from_imagery"
    return "neither", "imagery_differs_from_both"
