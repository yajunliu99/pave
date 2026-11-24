"""Coordinate-only review heuristics for lane strips.

Every finding is a prompt for the reviewer, not a verdict: a bicycle strip is
legitimately narrow and a turn bay legitimately tapers. Thresholds shared with
`hybrid.geometry_checks` keep the same names and defaults.
"""
from __future__ import annotations

import statistics

from ..hybrid.common import digest
from .strips import region_widths

DEFAULTS = {"merged_width_ratio": 1.65, "minimum_motor_width_ratio": .65, "taper_ratio": 1.6,
            "jagged_fraction": .35, "border_px": 6, "min_strip_width_m": 1.0, "max_lane_width_m": 5.5,
            "lane_width_prior_m": 3.6}

HINTS = {
    "possible_merged_lanes": "Much wider than its neighbours. Look for a separator inside it; split only if the pixels show one.",
    "narrow_region_check_facility": "Much narrower than its neighbours. It may be a bicycle strip or buffer, or a boundary may be misplaced.",
    "sliver_region": "Too narrow for any travel facility. Two boundaries may trace the same line.",
    "width_varies_along_section": "Its width changes a lot between stations. A real taper is possible; so is a boundary drifting off its line.",
    "jagged_edge": "This boundary steps sideways at one station instead of following a smooth line.",
    "touches_crop_border": "The outer boundary sits on the crop edge, so the carriageway may continue outside the crop.",
}


def check_strips(strips, width, policy=None, mpp=None, reference=None):
    """Return review issues; `key` changes whenever the flagged geometry changes."""
    p = {**DEFAULTS, **(policy or {})}
    stations, edges = strips["stations"], strips["edges"]
    widths = region_widths(strips)
    medians = [statistics.median(w) for w in widths]
    # A within-section reference needs enough strips to have a typical one.
    ref = reference or (statistics.median(medians) if len(medians) >= 3 else None)
    issues = {}

    def add(code, target, geometry, **detail):
        issue = issues.setdefault((target, code), {"key": f"{code}:{target}:{digest(geometry)[:8]}", "code": code,
                                                    "target": target, "severity": "review", "hint": HINTS[code], "detail": {}})
        issue["detail"].update(detail)

    for i, (samples, median) in enumerate(zip(widths, medians)):
        target = f"r{i + 1}"
        geometry = [[round(x) for x in edges[i]], [round(x) for x in edges[i + 1]]]
        if ref:
            ratio = median / ref
            if ratio > p["merged_width_ratio"]:
                add("possible_merged_lanes", target, geometry, relative_width=round(ratio, 2))
            if ratio < p["minimum_motor_width_ratio"]:
                add("narrow_region_check_facility", target, geometry, relative_width=round(ratio, 2))
        if mpp:
            metres = median * mpp
            if metres > p["max_lane_width_m"]:
                add("possible_merged_lanes", target, geometry, width_m=round(metres, 2))
            if metres < p["min_strip_width_m"]:
                add("sliver_region", target, geometry, width_m=round(metres, 2))
        if max(samples) / min(samples) > p["taper_ratio"]:
            add("width_varies_along_section", target, geometry, widths_px=[round(w, 1) for w in samples])

    scale = ref or (p["lane_width_prior_m"] / mpp if mpp else statistics.median(medians))
    for i, xs in enumerate(edges):
        for j in range(1, len(xs) - 1):
            span = (stations[j] - stations[j - 1]) / (stations[j + 1] - stations[j - 1])
            deviation = xs[j] - (xs[j - 1] + (xs[j + 1] - xs[j - 1]) * span)
            if abs(deviation) > p["jagged_fraction"] * scale:
                add("jagged_edge", f"e{i + 1}", [round(x) for x in xs], station=f"S{j + 1}", deviation_px=round(deviation, 1))
    if min(edges[0]) < p["border_px"]:
        add("touches_crop_border", "e1", [round(x) for x in edges[0]], side="left")
    if max(edges[-1]) > width - p["border_px"]:
        add("touches_crop_border", f"e{len(edges)}", [round(x) for x in edges[-1]], side="right")
    return [issues[k] for k in sorted(issues)]
