"""Approaches, receiving sections and turn candidates of one junction, from its sections' travel headings.

Nothing here assumes four legs or compass-aligned roads. Geometry settles only what it can measure: the
travel heading of every section and, when the network supplies it, the road arm a section lies on. Where
the heading change leaves the kind of turn open (a skewed leg, a Y), every plausible turn is offered and
the model chooses from the image.
"""
import math

from .common import require

# Travel heading implied by a compass section name; used only for configs that record no heading_deg.
COMPASS = {"NB": 0., "EB": 90., "SB": 180., "WB": 270.}
TURNS = ("left", "through", "right", "u_turn")
NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")
PREFIX = {"inbound_stopbar": "in_", "outbound_receiving": "out_"}
AMBIGUITY_DEG = 20.


def heading(section):
    """Travel heading, degrees clockwise from north (image up)."""
    value = section.get("heading_deg")
    if value is None:
        require(section["direction"] in COMPASS, f"{section['id']} records no heading_deg and is not named by compass")
        return COMPASS[section["direction"]]
    return float(value) % 360


def order(section):
    """Clockwise from north, so compass-aligned legs come out NB, EB, SB, WB."""
    return (heading(section) + 45) % 360


def turn_angle(h_in, h_out):
    """Signed heading change in degrees; positive is clockwise, i.e. a right turn on either driving side."""
    return (h_out - h_in + 180) % 360 - 180


def approaches(sections):
    return sorted((s for s in sections if s["kind"] == "inbound_stopbar"), key=order)


def candidates(delta, same_arm, margin=AMBIGUITY_DEG):
    """Turns a receiving section at heading change `delta` can receive.

    Through within 45 degrees, left/right to 135, U-turn beyond; within `margin` of a boundary both
    neighbours are offered. When arms are known (same_arm True/False) only the approach's own arm is a
    U-turn, however sharp a turn into another arm is.
    """
    if same_arm:
        return ["u_turn"]
    size, side = abs(delta), "right" if delta > 0 else "left"
    turns = []
    if size < 45 + margin:
        turns.append("through")
    if size > 45 - margin and (same_arm is False or size < 135 + margin):
        turns.append(side)
    if same_arm is None and size > 135 - margin:
        turns.append("u_turn")
    return turns


def receiving(sections, approach, margin=AMBIGUITY_DEG):
    """{receiving direction: [candidate turns]} for traffic entering at the inbound section `approach`."""
    arm = approach.get("arm")
    found = []
    for section in sections:
        if section["kind"] != "outbound_receiving":
            continue
        same = None if arm is None or section.get("arm") is None else section["arm"] == arm
        found.append((section, candidates(turn_angle(heading(approach), heading(section)), same, margin)))
    return {s["direction"]: turns for s, turns in sorted(found, key=lambda x: order(x[0]))}


def fixed_turns(options):
    """{turn: direction} when geometry pairs every turn with exactly one receiving section, else None."""
    if any(len(t) != 1 for t in options.values()):
        return None
    by_turn = {t[0]: d for d, t in options.items()}
    if len(by_turn) != len(options):
        return None
    return {t: by_turn[t] for t in TURNS if t in by_turn}


def describe(sections, approach, options):
    """What the model is told about receiving sections whose turn the geometry leaves open."""
    byid = {s["id"]: s for s in sections}
    rows = []
    for direction, turns in options.items():
        target = byid["out_" + direction]
        row = {"to_direction": direction, "heading_change_deg": round(turn_angle(heading(approach), heading(target))),
               "candidate_turns": turns}
        if approach.get("arm") is not None and target.get("arm") is not None:
            row["same_road_arm_as_approach"] = target["arm"] == approach["arm"]
        rows.append(row)
    return rows


def axis(section):
    """Unit travel vector in north-up image pixels (x right, y down)."""
    h = math.radians(heading(section))
    return math.sin(h), -math.cos(h)


def is_upstream(section, north_m, east_m):
    """True when a point (offsets from the junction centre) lies behind traffic of this approach."""
    h = math.radians(heading(section))
    return -(north_m * math.cos(h) + east_m * math.sin(h)) > 0


def opposing_approach(sections, outbound):
    """The inbound section on the receiving section's own road arm; its cameras look back over the receiving lanes."""
    inbound = [s for s in sections if s["kind"] == "inbound_stopbar"]
    if outbound.get("arm") is not None:
        return next((s for s in inbound if s.get("arm") == outbound["arm"]), None)
    best = max(inbound, key=lambda s: abs(turn_angle(heading(s), heading(outbound))), default=None)
    return best if best is not None and abs(turn_angle(heading(best), heading(outbound))) > 135 else None


def check_sections(sections, legs):
    """Every section is named by its kind and leg key, appears once, has a heading, and belongs to a network leg."""
    require(any(s["kind"] == "inbound_stopbar" for s in sections), "At least one inbound section is required")
    seen = set()
    for section in sections:
        require(section["kind"] in PREFIX, f"Unknown section kind {section['kind']}")
        require(section["id"] == PREFIX[section["kind"]] + section["direction"], f"{section['id']} must be named {PREFIX[section['kind']]}<direction>")
        require(section["id"] not in seen, f"Need one {section['direction']} {section['kind']} section")
        seen.add(section["id"])
        heading(section)
        key = "in_link_id" if section["kind"] == "inbound_stopbar" else "out_link_id"
        require(legs.get(section["direction"], {}).get(key) is not None, f"Section {section['id']} has no network link in legs")
    for direction, leg in legs.items():
        for key, prefix in (("in_link_id", "in_"), ("out_link_id", "out_")):
            require(leg.get(key) is None or prefix + direction in seen, f"Leg {direction} {key} has no {prefix}{direction} section")
