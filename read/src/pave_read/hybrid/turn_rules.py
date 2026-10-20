"""Traffic-law defaults applied to the model's movement decisions.

Kerbside turn: where traffic keeps right, the rightmost motor lane of an approach may turn right into a
receiving roadway on the right unless a marking or sign prohibits it (mirrored for left-hand traffic). The
movement step tends to leave such turns unresolved when no arrow is painted, because nothing in the image
"proves" them. This rule fills that gap and only that gap:

- it acts only where the junction geometry offers the kerbside turn toward exactly one receiving section, or
  the model itself named the receiving section for that turn;
- the kerbside lane is excluded when arrows read on it (satellite or street view) leave the turn out, e.g. a
  through-only arrow;
- the kerbside lane is paired with the receiving section's kerbside motor lane;
- the model's own decision is kept on record beside the default, and the movement is flagged.

Signs are not read here: a turn-restriction sign seen by the control audit is reported beside the movement for
review rather than applied, since most such signs (no turn on red, no U-turn) do not forbid the turn.
"""
from .legs import receiving

RULE = "kerbside_lane_turn_default"


def kerb_turn(side):
    return "right" if side == "right" else "left"


def motor_lanes(section):
    return sorted((r for r in section["regions"] if r.get("motor_lane_index")), key=lambda r: r["motor_lane_index"])


def kerb_lane(section, side):
    lanes = motor_lanes(section)
    return (lanes[-1] if side == "right" else lanes[0]) if lanes else None


def read_arrows(region):
    """Turns that arrows read on this lane allow, or [] when no arrow was read."""
    use = region.get("lane_use_prediction") or {}
    if use.get("basis") == "observed" and use.get("allowed_turns"):
        return list(use["allowed_turns"])
    return list(region.get("observed_arrows") or [])


def apply_kerb_turn_default(movements, lanes, config_sections, side="right"):
    """Movements with the kerbside-lane default added; also returns one note per approach it looked at."""
    turn = kerb_turn(side)
    byid = {s["section_id"]: s for s in lanes}
    out, notes = [dict(m) for m in movements], []
    for approach in (s for s in config_sections if s["kind"] == "inbound_stopbar"):
        direction = approach["direction"]
        inlet = byid.get(approach["id"])
        lane = kerb_lane(inlet, side) if inlet else None
        if lane is None:
            continue
        mine = [m for m in out if m["from_direction"] == direction]
        decided = next((m for m in mine if m["turn"] == turn), None)
        offered = [d for d, turns in receiving(config_sections, approach).items() if turn in turns]
        target = decided["to_direction"] if decided else (offered[0] if len(offered) == 1 else None)
        note = {"approach": direction, "kerb_lane": lane["motor_lane_index"], "turn": turn}
        if target is None:
            notes.append({**note, "applied": False, "why": "no single receiving section offered for the turn"})
            continue
        other = next((m for m in mine if m["to_direction"] == target and m["turn"] != turn), None)
        if other is not None:
            notes.append({**note, "applied": False, "why": f"the model judged the turn toward out_{target} as {other['turn']}"})
            continue
        exit_section = byid.get(f"out_{target}")
        dest = kerb_lane(exit_section, side) if exit_section else None
        arrows = read_arrows(lane)
        if dest is None:
            notes.append({**note, "applied": False, "why": f"out_{target} has no traced motor lane"})
            continue
        if arrows and turn not in arrows:
            notes.append({**note, "applied": False, "why": f"arrows read on the lane allow {'/'.join(arrows)} only"})
            continue
        pair = {"in_region_id": lane["region_id"], "out_region_id": dest["region_id"],
                "ib_lane": lane["motor_lane_index"], "ob_lane": dest["motor_lane_index"]}
        default = (f"Default rule: traffic keeps {side}, so the {side}most motor lane may turn {turn} into the receiving roadway "
                   f"unless a marking or sign prohibits it; no arrow read on this lane leaves the turn out.")
        if decided and decided["status"] == "candidate":
            if any(p["in_region_id"] == lane["region_id"] for p in decided["lane_pairs"]):
                notes.append({**note, "applied": False, "why": "the model already pairs the kerbside lane"})
                continue
            decided["lane_pairs"] = decided["lane_pairs"] + [pair]
            decided["reason"] = decided["reason"] + " [" + default + " The kerbside lane was added to the model's pairs.]"
        else:
            record = decided or {"turn": turn, "to_direction": target, "from_direction": direction,
                                 "in_section_id": approach["id"], "out_section_id": f"out_{target}", "evidence_refs": [],
                                 "assumptions": [], "validation_flags": [], "review_status": "needs_review", "auto_apply": False}
            if decided:
                record["model_decision"] = {k: decided.get(k) for k in ("status", "basis", "confidence", "reason")}
            record.update(status="candidate", lane_pairs=[pair], basis="inferred", confidence="medium",
                          reason=default + (" Model's own assessment: " + decided["reason"] if decided else ""))
            record["evidence_refs"] = sorted(set(record.get("evidence_refs", [])) | {f"{approach['id']}:{lane['region_id']}",
                                                                                       f"out_{target}:{dest['region_id']}"})
            if not decided:
                out.append(record)
            decided = record
        decided["default_rule"] = RULE
        decided["validation_flags"] = sorted(set(decided.get("validation_flags", [])) | {RULE})
        notes.append({**note, "applied": True, "to_direction": target, "pair": pair})
    return out, notes
