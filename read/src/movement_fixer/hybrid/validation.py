from __future__ import annotations

import json

from .common import require, ValidationError

DIRECTIONS = ("NB", "EB", "SB", "WB")
MOTOR_TYPES = {"motor_vehicle_lane", "shared_rail_motor_lane"}
SURFACE_TYPES = MOTOR_TYPES | {"bicycle_strip", "rail_area", "median_or_buffer", "shoulder_or_parking", "unknown"}
TURNS = {"left":-1, "through":0, "right":1, "u_turn":2}


def receiving_direction(incoming, turn):
    return DIRECTIONS[(DIRECTIONS.index(incoming)+TURNS[turn]) % 4]


def compass_receiving(incoming):
    """Receiving options of a four-way compass junction, in the {direction: [turn]} form of legs.receiving."""
    return {d:[t for t in TURNS if receiving_direction(incoming,t)==d] for d in DIRECTIONS}


def validate_lanes(value, section, geometry_audit=None):
    require(isinstance(value, dict) and isinstance(value.get("regions"), list), "Lane output requires regions[]")
    expected = [r["id"] for r in section["regions"]]
    actual = [r.get("region_id") for r in value["regions"]]
    require(set(actual) == set(expected) and len(actual) == len(expected), "Missing, duplicate or foreign region IDs")
    for r in value["regions"]:
        require(r.get("surface_type") in SURFACE_TYPES, "Unknown surface type")
        require(isinstance(r.get("observed_arrows"), list) and set(r["observed_arrows"]) <= set(TURNS), "Invalid observed arrows")
        require(r.get("confidence") in ("high","medium","low"), "Invalid confidence")
        require(r.get("basis") in ("observed","inferred","mixed"), "Invalid evidence basis")
        require(r.get("visibility") in ("clear","partial","occluded","not_visible","unclear"), "Invalid visibility")
        require(not r["observed_arrows"] or r["visibility"] in ("clear","partial"), "Observed arrows require clear/partial visibility")
        require(isinstance(r.get("evidence"), str), "Missing evidence text")
    byid = {r["region_id"]:r for r in value["regions"]}
    ordered = [byid[x] for x in expected]
    region_config={r["id"]:r for r in section["regions"]}
    lane_index = 0
    blocked=[]
    for r in ordered:
        metrics=(geometry_audit or {}).get("regions",{}).get(r["region_id"])
        if metrics: r["width_metrics"]=metrics
        annotation=region_config[r["region_id"]].get("review_annotation")
        if annotation:
            r["review_annotation"]=annotation
            r["reviewed_directions"]=annotation.get("intended_directions",[])
            r["review_classification_disagreement"]=annotation.get("surface_type")!=r["surface_type"]
        # User-confirmed exclusive turns take priority over geometric hypotheses.
        # Image observations retain their own provenance and cannot be fabricated.
        allowed=None; permission_source=None
        if annotation and annotation.get("exclusive"):
            allowed=annotation.get("intended_directions",[])
            require(allowed and set(allowed)<=set(TURNS),"Exclusive review requires valid intended directions")
            permission_source="user_review"
        elif r["surface_type"] in MOTOR_TYPES and r["observed_arrows"]:
            allowed=r["observed_arrows"]
            permission_source="observed_pavement_arrows"
        r["turn_constraint"]={"allowed_turns":allowed,"source":permission_source,
            "enforcement":"hard" if allowed else "unestablished"}
        review_class_ok=not annotation or annotation.get("surface_type",r["surface_type"])==r["surface_type"]
        r["endpoint_eligible"]=r["surface_type"] in MOTOR_TYPES and review_class_ok and (metrics is None or metrics["single_motor_lane_width_supported"])
        r["motor_lane_index"] = None
        if r["endpoint_eligible"]:
            lane_index += 1; r["motor_lane_index"] = lane_index
        elif r["surface_type"] in MOTOR_TYPES: blocked.append(r["region_id"])
    unknown = sum(r["surface_type"] == "unknown" for r in ordered)
    return {"section_id":section["id"], "kind":section["kind"], "direction":section["direction"],
            "source_id":section["source_id"], "regions":ordered,
            "count":{"model_motor_count":lane_index,"unclassified_regions":unknown,
                     "model_motor_region_count":sum(r["surface_type"] in MOTOR_TYPES for r in ordered),
                     "geometry_blocked_regions":blocked,
                     "model_count_range":[lane_index,None if blocked else lane_index+unknown],"human_verified":False,
                     "coverage":section.get("coverage","partial")},
            "notes":value.get("notes",[]),"excluded_regions":section.get("excluded_regions",[]),"reference_count":None}


def validate_context(value,view,source):
    require(isinstance(value,dict) and isinstance(value.get("observations"),list),"Context requires observations[]")
    ids=[x.get("id") for x in value["observations"]]
    require(len(ids)==len(set(ids)) and all(isinstance(x,str) and x for x in ids),"Invalid context observation IDs")
    for o in value["observations"]:
        require(isinstance(o.get("evidence"),str),"Context evidence required")
        require(isinstance(o.get("observed_arrows"),list) and set(o["observed_arrows"])<=set(TURNS),"Invalid context arrows")
        require(o.get("applies_to_incoming") in ("yes","no","uncertain"),"Invalid incoming applicability")
    return {"section_id":view["id"],"kind":"gsv_forward_context","direction":view["direction"],
            "source_id":view["source_id"],"reverse_view":False,"capture_date":source.get("capture_date"),
            "sampling_position":view.get("sampling_position"),"distance_to_center_m":source.get("distance_to_center_m"),
            "pano_id":source.get("pano_id"),
            "observations":value["observations"],"notes":value.get("notes",[])}


def validate_movement_output(value, direction, sections, geometry_quality=None, geometry_audit=None, enforce_turn_constraints=True, receiving=None):
    """receiving: {receiving direction: [candidate turns]} from legs.receiving; default is a four-way compass junction.
    One decision per receiving section; its turn must be one of that section's candidates."""
    receiving=compass_receiving(direction) if receiving is None else receiving
    require(isinstance(value,dict) and isinstance(value.get("movements"),list), "Movement output requires movements[]")
    alignment_notes=value.get("gsv_alignment_notes",[])
    require(isinstance(alignment_notes,list),"GSV alignment notes must be a list")
    # Free-text notes only: a structured note is kept as its JSON text rather than failing the decision.
    alignment_notes=[n if isinstance(n,str) else json.dumps(n,ensure_ascii=False) for n in alignment_notes]
    byid = {s["section_id"]:s for s in sections}
    context_dates={s["sampling_position"]:s.get("capture_date") for s in sections
        if s.get("kind")=="gsv_forward_context" and s.get("direction")==direction and s.get("sampling_position")}
    mixed_dates=bool(context_dates) and (len(set(context_dates.values()))>1 or None in context_dates.values())
    expected_in = f"in_{direction}"
    seen = set(); labels = set(); results = []; dropped = []
    for proposal in value["movements"]:
        turn = proposal.get("turn"); out_dir = proposal.get("to_direction")
        require(turn in TURNS, "One unique decision per turn is required")
        if out_dir not in receiving and not proposal.get("lane_pairs"):
            # A decision toward a receiving section that is not offered (e.g. one that could not be traced) is
            # kept on record but is not a decision of this run; one that pairs lanes with it is still an error.
            dropped.append({k: proposal.get(k) for k in ("turn", "to_direction", "status", "reason")}); continue
        require(out_dir in receiving and turn in receiving[out_dir], "Turn/receiving-direction mismatch")
        require(out_dir not in seen, "One decision per receiving section is required")
        seen.add(out_dir)
        require(proposal.get("status") in ("candidate","unresolved","not_proposed"), "Invalid proposal status")
        require(proposal.get("basis") in ("observed","inferred","mixed"), "Invalid movement basis")
        require(proposal.get("confidence") in ("high","medium","low"), "Invalid movement confidence")
        require(isinstance(proposal.get("reason"),str), "Movement reason required")
        pairs = proposal.get("lane_pairs")
        require(isinstance(pairs,list), "lane_pairs[] required")
        in_regions = {r["region_id"]:r for r in byid[expected_in]["regions"]}
        out_regions = {r["region_id"]:r for r in byid[f"out_{out_dir}"]["regions"]}
        converted = []; pairkeys=set(); flags=["gsv_capture_date_mismatch"] if mixed_dates else []
        if turn in labels: flags.append("turn_label_shared_by_two_receiving_sections")
        labels.add(turn)
        for pair in pairs:
            i,o = pair.get("in_region_id"), pair.get("out_region_id")
            require(i in in_regions and o in out_regions, "Movement references nonexistent section region")
            require(in_regions[i]["surface_type"] in MOTOR_TYPES and out_regions[o]["surface_type"] in MOTOR_TYPES,
                    "Movement cannot use a bicycle, rail-only or unclassified region")
            require(in_regions[i].get("endpoint_eligible",True) and out_regions[o].get("endpoint_eligible",True),
                    "Movement endpoint needs geometry split/type review; single-lane width is unsupported")
            require((i,o) not in pairkeys,"Duplicate lane pair"); pairkeys.add((i,o))
            arrows=in_regions[i]["observed_arrows"]
            constraint=in_regions[i].get("turn_constraint",{})
            allowed=constraint.get("allowed_turns")
            # Also protect cached/legacy records that lack normalized constraints.
            if not allowed and in_regions[i].get("review_annotation",{}).get("exclusive"):
                allowed=in_regions[i]["review_annotation"].get("intended_directions")
            if not allowed and arrows: allowed=arrows
            if enforce_turn_constraints:
                require(not allowed or turn in allowed,
                    f"Turn {turn} contradicts the lane-use constraint on {expected_in}:{i}; geometric alignment/assumptions cannot override exclusive turns or observed arrows")
            else:
                hypothesis=in_regions[i].get("lane_use_prediction",{}).get("allowed_turns",[])
                if hypothesis and turn not in hypothesis: flags.append(f"model_lane_use_disagreement:{i}:{turn}")
            if arrows and turn not in arrows: flags.append(f"arrow_disagreement:{i}:{turn}")
            reviewed=in_regions[i].get("reviewed_directions",[])
            if reviewed and turn not in reviewed: flags.append(f"reviewed_turn_disagreement:{i}:{turn}")
            if in_regions[i]["confidence"]=="low" or out_regions[o]["confidence"]=="low":
                flags.append("low_confidence_endpoint")
            if geometry_quality:
                for sid,rid in ((expected_in,i),(f"out_{out_dir}",o)):
                    quality=geometry_quality.get(sid,{}).get(rid,"")
                    if "provisional" in quality or "clipped" in quality:
                        flags.append("provisional_endpoint_geometry")
            pair_result={"in_region_id":i,"out_region_id":o,
                         "ib_lane":in_regions[i]["motor_lane_index"],"ob_lane":out_regions[o]["motor_lane_index"]}
            straight=(geometry_audit or {}).get("straight_pairs",{}).get(direction,{})
            if turn=="through" and straight.get("out_section_id",f"out_{direction}")==f"out_{out_dir}":
                check=next((x for x in straight.get("pairs",[])
                            if x["in_region_id"]==i and x["out_region_id"]==o),None)
                if check:
                    require(not check["hard_violation"],f"Through pair {i}->{o} has excessive cross-corridor displacement; choose an aligned pair or leave unresolved")
                    if check["review_required"]: flags.append("straight_alignment_large")
                    pair_result["alignment"]=check
            converted.append(pair_result)
        if proposal["status"]=="candidate": require(bool(converted),"Candidate needs at least one lane pair")
        else: require(not converted,"Unresolved/not_proposed decisions must not contain lane pairs")
        for a in converted:
            for b in converted:
                if a["ib_lane"]<b["ib_lane"] and a["ob_lane"]>b["ob_lane"]:
                    flags.append("crossing_lane_order")
        if len({p["ib_lane"] for p in converted})>len({p["ob_lane"] for p in converted}):
            flags.append("many_to_one_merge_requires_review")
        if turn=="u_turn" and converted: flags.append("u_turn_permission_unverified")
        refs=proposal.get("evidence_refs",[])
        require(isinstance(proposal.get("assumptions",[]),list) and all(isinstance(x,str) for x in proposal.get("assumptions",[])),"assumptions must be strings")
        require(isinstance(refs,list) and all(isinstance(x,str) for x in refs),"evidence_refs must be strings")
        listed=[];unlisted=[]
        for ref in refs:
            section_ref, sep, region_ref=ref.partition(":")
            if section_ref not in byid:
                # e.g. the overview image itself: kept and flagged, but it cannot be the only support.
                unlisted.append(ref);continue
            if sep:
                valid={r["region_id"] for r in byid[section_ref].get("regions",[])} | {o["id"] for o in byid[section_ref].get("observations",[])}
                require(region_ref in valid,f"Unknown evidence region/observation {ref}")
            listed.append(ref)
        if unlisted: flags.append("unlisted_evidence_reference")
        require(listed or proposal["status"]!="candidate","Candidates need traceable evidence references to the listed sections, regions or observations")
        proposal={**proposal,"evidence_refs":listed,**({"unlisted_evidence_refs":unlisted} if unlisted else {})}
        results.append({**proposal,"from_direction":direction,"in_section_id":expected_in,
                        "gsv_alignment_notes":alignment_notes,
                        "gsv_capture_dates":context_dates,
                        "out_section_id":f"out_{out_dir}","lane_pairs":converted,
                        "validation_flags":sorted(set(flags)),"review_status":"needs_review", "auto_apply":False})
    require(seen==set(receiving),"Every receiving section needs one decision, including an unresolved U-turn")
    for result in results if dropped else []:
        result["dropped_decisions"]=dropped
        result["validation_flags"]=sorted(set(result["validation_flags"])|{"decision_toward_unoffered_receiving_section_dropped"})
    return results
