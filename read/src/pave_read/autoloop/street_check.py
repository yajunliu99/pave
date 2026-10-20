"""Street-view lane counts held against the traced lanes, and the review issue that sends a mismatch back to tracing.

Classification reports, for each forward street view of an approach, how many motor lanes of that direction cross
the image at the camera and which lanes or regions do not line up (`street_view_counts`). Where the street views
settle on a count that differs from the motor lanes found among the traced regions, the section goes back to the
review loop with that finding as an issue; the reviewer sees the street view with the boundaries projected onto it
and decides whether a boundary is missing, spurious, or right after all (the camera may stand where a pocket has
not opened yet). Counts are model readings, so they prompt a review; they never overwrite the geometry.

An independent reading of the approach (scripts/pipeline/read_approaches.py: the unmarked satellite crop and street
views, read blind, normally by another model) is a second source of disagreement: where its settled count differs
from the classified motor lanes, its lane-by-lane layout goes to the reviewer the same way. No reference count (UTDF)
is used to choose or to word either issue.
"""
from __future__ import annotations

ORDER = ("near", "mid")


def street_count(use_audit, sources):
    """(count, basis, readings) for one approach, or (None, reason, readings).

    The near view's count when it is settled; otherwise the near and mid counts when they agree; otherwise the mid
    view's settled count. Far views are left out: lanes there often differ from the stop bar."""
    readings = {}
    for c in (use_audit or {}).get("street_view_counts", []):
        position = sources.get(c.get("source_id"), {}).get("sampling_position")
        if position in ORDER and c.get("motor_lanes") is not None:
            readings[position] = c
    near, mid = readings.get("near"), readings.get("mid")
    if near and near.get("settled", True):
        return near["motor_lanes"], "near view, settled", readings
    if near and mid and near["motor_lanes"] == mid["motor_lanes"]:
        return near["motor_lanes"], "near and mid views agree", readings
    if mid and mid.get("settled", True):
        return mid["motor_lanes"], "mid view, settled", readings
    return None, "no settled count", readings


def mismatches(lanes, uses, config):
    """Inbound sections whose street-view lane count differs from their classified motor lanes."""
    sources = config.get("sources", {})
    out = []
    for section in lanes:
        if section.get("kind") != "inbound_stopbar":
            continue
        direction = section["direction"]
        count, basis, readings = street_count(uses.get(direction), sources)
        traced = section["count"]["model_motor_count"]
        if count is None or count == traced:
            continue
        views = {p: {"motor_lanes": r["motor_lanes"], "settled": r.get("settled", True), "other_lanes": r.get("other_lanes", ""),
                     "unmatched": r.get("unmatched", ""), "distance_m": round(sources.get(r["source_id"], {}).get("distance_to_center_m") or 0),
                     "date": sources.get(r["source_id"], {}).get("capture_date")} for p, r in readings.items()}
        out.append({"section_id": section["section_id"], "direction": direction, "traced_motor_lanes": traced, "street_view_lanes": count,
                    "basis": basis, "views": views,
                    "regions": {r["region_id"]: f"{r['surface_type']} ({r['existence']})" for r in section["regions"]}})
    return out


def issue(m):
    """The review issue for one mismatch (the same shape as autoloop.checks issues)."""
    more = m["street_view_lanes"] > m["traced_motor_lanes"]
    hint = (f"The forward street views show {m['street_view_lanes']} motor lanes of this direction at the camera ({m['basis']}); "
            f"after classification the traced regions hold {m['traced_motor_lanes']}. "
            + ("Look for a lane with no strip of its own: a boundary missing inside a wide region, or a lane beyond the outermost boundary. "
               if more else "Look for a strip that is not a lane of this direction: opposing traffic, kerb, parking, or two strips over one lane. ")
            + "Compare the street view panel, where the boundaries are projected, with the crop. If the camera stands where the lane "
              "layout differs from this section (a pocket that opens nearer the junction, a lane that ends), dismiss and say so.")
    return {"key": f"street_view_count:{m['street_view_lanes']}:{m['traced_motor_lanes']}", "code": "street_view_count",
            "target": m["section_id"], "hint": hint,
            "detail": {"street_views": m["views"], "classified_regions": m["regions"]}}


def reading_mismatches(lanes, readings):
    """Inbound sections whose settled, not low-confidence independent reading differs from their classified motor lanes."""
    by_approach = {r["approach"]: r for r in readings if r.get("reading")}
    out = []
    for section in lanes:
        if section.get("kind") != "inbound_stopbar" or section["direction"] not in by_approach:
            continue
        r = by_approach[section["direction"]]
        reading, traced = r["reading"], section["count"]["model_motor_count"]
        if not reading["count_settled"] or reading["count_confidence"] == "low" or len(reading["lanes"]) == traced:
            continue
        out.append({"section_id": section["section_id"], "direction": section["direction"], "traced_motor_lanes": traced,
                    "reading_lanes": len(reading["lanes"]), "layout": r["layout_seen"], "confidence": reading["count_confidence"],
                    "lanes": [{"lane": l["lane"], "turns": l["turns"], "evidence": l["evidence"]} for l in reading["lanes"]],
                    "other": {k: reading.get(k, 0) for k in ("bicycle_lanes", "rail_or_bus_only_lanes", "parking_lanes")},
                    "limitations": reading.get("limitations", ""),
                    "regions": {x["region_id"]: f"{x['surface_type']} ({x['existence']})" for x in section["regions"]}})
    return out


def reading_issue(m):
    """The review issue for a disagreement with the independent reading (same shape as autoloop.checks issues)."""
    more = m["reading_lanes"] > m["traced_motor_lanes"]
    hint = (f"An independent reading of this approach at the stop bar (unmarked satellite crop and forward street views, read without "
            f"these regions) found {m['reading_lanes']} general motor lanes, from the driver's left: {m['layout']} "
            f"({m['confidence']} confidence); after classification the traced regions hold {m['traced_motor_lanes']}. "
            + ("Look for a lane with no strip of its own: a boundary missing inside a wide region, or a lane beyond the outermost boundary. "
               if more else "Look for a strip that is not a lane of its own: two strips over one lane (for example the edges of a rail slab "
                            "or a worn line taken for a lane line), a buffer, bicycle lane, parking or opposing traffic. ")
            + "The reading is another model's view, not ground truth: change a boundary only where you can point to the paint, kerb or "
              "surface edge that shows it is missing or spurious; if the regions are right, dismiss and say what the reading missed.")
    return {"key": f"independent_reading_count:{m['reading_lanes']}:{m['traced_motor_lanes']}", "code": "independent_reading_count",
            "target": m["section_id"], "hint": hint,
            "detail": {"reading_lanes": m["lanes"], "reading_other_facilities": m["other"], "reading_limitations": m["limitations"],
                       "classified_regions": m["regions"]}}

