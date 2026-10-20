"""Lane labels shared by the figures: movement letters, lane number and facility type."""

MOTOR, SHARED, BIKE, RAIL, OTHER, REJECTED, UNSURE = "#16d5ec", "#ff9a1f", "#35c94a", "#c58bff", "#8fa3b8", "#8a8a8a", "#ffe14d"
TURN_COLOR = {"left": "#ff5cf4", "through": "#ffffff", "right": "#ffe14d", "u_turn": "#ff8a3d"}
LETTER = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}
PLAIN = {"bicycle_strip": ("BIKE", BIKE), "rail_area": ("RAIL", RAIL), "median_or_buffer": ("BUFFER", OTHER),
         "shoulder_or_parking": ("PARKING", OTHER), "non_road": ("NOT ROAD", REJECTED), "unknown": ("?", UNSURE)}
LEGEND = [(MOTOR, "motor lane: L left, T through, R right + lane number from driver-left; number alone = use not established"),
          (SHARED, "motor lane shared with rail"), (BIKE, "bicycle strip"), (RAIL, "rail area, no motor traffic"),
          (OTHER, "buffer or parking"), (REJECTED, "judged not a travel facility"), (UNSURE, "could not be resolved from the imagery")]


def label(region, inbound):
    """(lines, colour) for one region, from the run's fused surface and lane-use predictions."""
    kind, existence = region["surface_type"], region["existence"]
    if existence == "rejected":
        return [PLAIN.get(kind, ("REJECTED", REJECTED))[0] if kind in ("median_or_buffer", "shoulder_or_parking") else "NOT ROAD"], REJECTED
    if kind not in ("motor_vehicle_lane", "shared_rail_motor_lane"):
        text, color = PLAIN.get(kind, ("?", UNSURE))
        return [text + ("?" if existence == "uncertain" and text != "?" else "")], UNSURE if existence == "uncertain" else color
    if existence == "uncertain":
        return ["LANE?"], UNSURE
    number = region.get("motor_lane_index")
    turns = (region.get("lane_use_prediction") or {}).get("allowed_turns") or []
    letters = "".join(LETTER[t] for t in ("left", "through", "right", "u_turn") if t in turns)
    text = (letters if inbound else "OUT") + (str(number) if number else "")
    return ([text, "+RAIL"], SHARED) if kind == "shared_rail_motor_lane" else ([text], MOTOR)
