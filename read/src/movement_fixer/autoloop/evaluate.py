"""Post-run comparison of traced regions with reference polygons.

Read only after inference has finished. The pilot's reference polygons were
themselves traced by an assistant and are partly provisional, so agreement
measures closeness to that tracing, not surveyed truth.
"""
from __future__ import annotations

from pathlib import Path

from ..hybrid.common import read_json
from .strips import Frame, region_polygon

SAMPLES = 5
# A region spanning two reference lanes overlaps each at 0.5; that must not count as a match.
MATCH_IOU = .6


def _interval(polygon, y):
    xs = []
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 <= y < y2) or (y2 <= y < y1):
            xs.append(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def _span(polygons):
    ys = [y for polygon in polygons for _, y in polygon]
    return min(ys), max(ys)


def _iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0., inter) / (max(a[1], b[1]) - min(a[0], b[0]))


def compare(strips, frame, reference_regions, mpp=None):
    """Match candidate strips to reference regions by lateral overlap in the section frame."""
    candidate = [region_polygon(strips, i) for i in range(len(strips["edges"]) - 1)]
    reference = [[frame.to_local(*p) for p in r["polygon"]] for r in reference_regions]
    low = max(_span(candidate)[0], _span(reference)[0])
    high = min(_span(candidate)[1], _span(reference)[1])
    result = {"candidate_regions": len(candidate), "reference_regions": len(reference),
              "count_difference": len(candidate) - len(reference)}
    if high <= low:
        return {**result, "comparable": False, "reason": "candidate and reference cover different road lengths"}
    ys = [low + (high - low) * (i + .5) / SAMPLES for i in range(SAMPLES)]
    scores = []
    for i, c in enumerate(candidate):
        for j, r in enumerate(reference):
            pairs = [(a, b) for y in ys if (a := _interval(c, y)) and (b := _interval(r, y))]
            if pairs:
                edge = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in pairs) / (2 * len(pairs))
                scores.append((sum(_iou(a, b) for a, b in pairs) / len(pairs), i, j, edge))
    matches, used_c, used_r = [], set(), set()
    for iou, i, j, edge in sorted(scores, reverse=True):
        if iou < MATCH_IOU or i in used_c or j in used_r:
            continue
        used_c.add(i)
        used_r.add(j)
        matches.append({"candidate": f"r{i + 1}", "reference": reference_regions[j]["id"], "iou": round(iou, 3),
                        "edge_error_px": round(edge, 1)})
    errors = [m["edge_error_px"] for m in matches]
    mean = round(sum(errors) / len(errors), 1) if errors else None
    return {**result, "comparable": True, "matched": len(matches),
            "matches": sorted(matches, key=lambda m: m["candidate"]),
            "unmatched_candidate": [f"r{i + 1}" for i in range(len(candidate)) if i not in used_c],
            "unmatched_reference": [r["id"] for j, r in enumerate(reference_regions) if j not in used_r],
            "mean_edge_error_px": mean, "mean_edge_error_m": round(mean * mpp, 2) if mean is not None and mpp else None}


def evaluate_run(run, reference_config, mpp=None):
    """Compare each section's draft and final geometry with the reference, to show what the loop changed."""
    history = read_json(Path(run) / "geometry" / "history.json")
    reference = {s["id"]: s for s in read_json(reference_config)["sections"]}
    rows = []
    for record in history:
        sid = record["section_id"]
        row = {"section_id": sid, "status": record["status"], "stop_reason": record["stop_reason"]}
        if sid not in reference or "final" not in record:
            rows.append({**row, "comparable": False, "reason": "no reference section or no geometry"})
            continue
        frame = Frame(**{**record["frame"], "center": tuple(record["frame"]["center"])})
        row["draft"] = compare(record["draft"], frame, reference[sid]["regions"], mpp)
        row["final"] = compare(record["final"], frame, reference[sid]["regions"], mpp)
        rows.append(row)

    def total(stage, key):
        return sum(r[stage][key] for r in rows if stage in r and r[stage].get("comparable"))
    both = [r for r in rows if "final" in r and r["final"].get("comparable") and r["draft"].get("comparable")]

    def score(result):  # more matched regions first, then fewer traced regions with no counterpart
        return result["matched"], -len(result["unmatched_candidate"])
    return {"sections": rows,
            "totals": {stage: {"reference_regions": total(stage, "reference_regions"), "matched": total(stage, "matched"),
                               "candidate_regions": total(stage, "candidate_regions"),
                               "traced_without_counterpart": total(stage, "candidate_regions") - total(stage, "matched")}
                       for stage in ("draft", "final")},
            "loop_effect": {"sections_improved": [r["section_id"] for r in both if score(r["final"]) > score(r["draft"])],
                            "sections_worsened": [r["section_id"] for r in both if score(r["final"]) < score(r["draft"])],
                            "sections_unchanged": sum(score(r["final"]) == score(r["draft"]) for r in both)},
            "claim_scope": "Closeness to one assistant-traced reference on an already inspected pilot; not accuracy against surveyed truth."}
