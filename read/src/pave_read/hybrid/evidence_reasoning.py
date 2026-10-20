import json
from .common import require
from .validation import MOTOR_TYPES,SURFACE_TYPES,TURNS

EXISTENCE={"supported","rejected","uncertain"}
USES={"left_only","right_only","through_only","shared","unknown","not_applicable"}


KERB_LANE=("Near the stop bar the strip between the outermost lane line and the kerb is often a right-turn lane even without a readable arrow. "
           "A strip about 2.7 m or wider whose pavement runs to the kerb, with no bicycle symbol, green paint, parking stalls or parked cars, "
           "is a motor_vehicle_lane; a bicycle lane is usually 1.2-2 m wide and marked. Do not call a car-width kerb strip unknown or a bicycle "
           "lane only because no arrow can be read.")


def satellite_date(source):
    """The satellite image's capture date, with the provider's note on it when there is one, for the image list a prompt shows."""
    return {"capture_date":source.get("capture_date"),**({"date_note":source["date_note"]} if source.get("date_note") else {})}


def surface_prompt(section,geometry,images,mpp=None):
    geometry={rid:{**{k:r[k] for k in ("median_width_px","reference_width_px","relative_width")},
                   **({"median_width_m":round(r["median_width_px"]*mpp,2)} if mpp else {})}
              for rid,r in (geometry or {}).get("regions",{}).items()}
    return f"""Audit surface/roadway existence for {section['id']}, traffic direction {section['direction']}.
The satellite context has actual travel pointing UP. The two satellite panels are the SAME pixels: one unmarked, one with candidate IDs. Proposed polygons are hypotheses, not detected lane lines or a known road envelope. Every region may be valid, invalid, or unresolved. No target count or reference labels are provided.
Inspect the unmasked surroundings: curb alignment, sidewalk/pavers, vegetation, buildings, shadow boundaries, bike markings, lane paint and continuity. Determine the physical surface before counting lanes. {KERB_LANE if section["kind"]=="inbound_stopbar" else ""} Apparent car-like shapes in shadow and regular polygon widths do not establish roadway. A curb or other real edge can cross a proposed polygon; do not extend pavement just to fill its rectangle. Conversely, do not reject a region merely because markings are occluded.
If GSV is supplied, use its physical layout only after accounting for camera position and direction. For an OUTBOUND section, the supplemental GSV may be from the OPPOSING approach at the same side of the intersection: the target receiving carriageway can be across the center separation, not the camera's own lane. Do not match lane ordinals across views.
For each supplied ID report physical support for a motor/bicycle/rail travel facility, or rejected if it is actually non-road/sidewalk/landscape/buffer/parking, or uncertain if not resolvable. Exact subtype may remain unknown. Do not add human reference answers.
Return JSON {{"regions":[{{"region_id":"r#","surface_type":"motor_vehicle_lane|shared_rail_motor_lane|bicycle_strip|rail_area|median_or_buffer|shoulder_or_parking|non_road|unknown","existence":"supported|rejected|uncertain","observed_arrows":[],"visibility":"clear|partial|occluded|unclear","confidence":"high|medium|low","evidence":"physical landmarks and uncertainty","visual_refs":[{{"source_id":"one supplied image ID","bbox_xyxy":[0.0,0.0,1.0,1.0],"finding":"visible anchor supporting classification"}}]}}],"notes":[]}}.
Bboxes are normalized 0..1 in the named NATIVE IMAGE, not in the combined sheet. Use [] when no reliable anchor is visible. Supply each region exactly once; no invented arrow when unreadable.
Region IDs: {json.dumps([r['id'] for r in section['regions']])}
Image roles/dates: {json.dumps(images)}
Geometry hints (widths across the strip; metres from the image scale), not semantic answers: {json.dumps(geometry)}
"""


def classification_review_text(review):
    """Prompt text for a classification check (scripts/pipeline/classification_check.py); empty without one."""
    if not review:
        return ""
    return ("\nA later check of this approach disagreed with the classification: "+json.dumps(review,ensure_ascii=False)+"\n"
            "Re-examine the regions it concerns against the imagery. It is another model's reading, not ground truth and not a target: "
            "change a region's class or use only where you can point to the paint, arrows, hatching, kerb, parked cars or surface that show it, "
            "otherwise keep your class and say in notes what the check missed.\n")


def usage_prompt(direction,lane_evidence,source_metadata,contexts,review=None):
    return f"""Infer lane USE from visual evidence for the {direction} incoming approach.
The satellite panel shows candidate regions with travel pointing UP. GSV panels retain original camera orientation and each has its own source ID. No lane-use reference answers or reviewed turn constraints are provided. The supplied surface audit is another model's prediction, not truth.
Reassess the physical surface and facility existence for EVERY candidate using the joint evidence. A satellite-only parking/non-road interpretation can be wrong, especially with stopped cars or shadow; retain it, revise it, or leave uncertain based on explicit GSV-to-satellite landmarks. Do not mechanically propagate the preliminary class. Conversely, a GSV motor lane does not prove that a differently located satellite polygon is the same facility. Report the fused surface type, existence and reasoning separately from lane use.
{KERB_LANE}
For every forward street view, count the lanes of this direction that cross the image at the camera's position, from the centre line or median to the kerb, and say how they line up with the candidate regions: name a lane that has no region of its own, or a region that is not such a lane. Report this in street_view_counts; it is checked against the traced regions later. Keep the region classes consistent with it: a region you find to be opposing traffic, sidewalk, parking or another non-lane gets that class here, not motor_vehicle_lane.
Read arrows, ONLY signs and lane/bike/curb/rail ordering. Explicitly bind any marking to a satellite region using physical anchors. Vehicle heading, lane width and straight alignment alone do not establish permission. A right/left arrow plus an applicable ONLY marking supports dedicated use; unreadability is not absence. Do not assign a curbside sign to a neighboring lane merely because it is nearby in the picture.
Use all supplied distance bands. Trace any splits, turn pockets and bicycle crossings between the far/mid image and the stopbar; retain uncertainty when continuity is not visible. A date difference can be a temporal change, not a spatial transition. Same-panorama crops are not independent confirmations.
Do NOT make an answer match a preferred number of lanes or movements. For every region report your predicted use and the specific evidence binding. If a small sign/arrow could resolve uncertainty, request at most three native-image crops by source_id and normalized bbox. The code will crop those exact pixels once; do not guess text to avoid requesting a crop.
Return JSON {{"regions":[{{"region_id":"r#","surface_type":"motor_vehicle_lane|shared_rail_motor_lane|bicycle_strip|rail_area|median_or_buffer|shoulder_or_parking|non_road|unknown","existence":"supported|rejected|uncertain","surface_reason":"joint physical evidence for retaining or revising the initial audit","use":"left_only|right_only|through_only|shared|unknown|not_applicable","allowed_turns":["left|through|right|u_turn"],"basis":"observed|inferred|unknown","confidence":"high|medium|low","binding":"physical region correspondence, temporal caveats and uncertainty","visual_refs":[{{"source_id":"supplied ID","bbox_xyxy":[0.0,0.0,1.0,1.0],"finding":"visible arrow/sign/continuity anchor"}}]}}],"detail_requests":[{{"source_id":"supplied ID","bbox_xyxy":[0.0,0.0,1.0,1.0],"question":"specific visual ambiguity to inspect"}}],"street_view_counts":[{{"source_id":"forward street view ID","motor_lanes":0,"other_lanes":"bicycle, parking, rail or bus lanes seen, or none","unmatched":"lanes without a region or regions that are not lanes, or none","settled":true}}],"notes":[]}}.
Bboxes refer to the named native image, not the overall sheet. Unknown or not_applicable uses must have allowed_turns=[]. Labels ending _only must have only that turn. These are model predictions and will be evaluated without replacing them with a reviewed answer.
Surface audit: {json.dumps(lane_evidence,ensure_ascii=False)}
Sources: {json.dumps(source_metadata,ensure_ascii=False)}
Independent image observations: {json.dumps(contexts,ensure_ascii=False)}
{classification_review_text(review)}"""


def validate_refs(refs,sources):
    require(isinstance(refs,list),"Visual references must be a list")
    for r in refs:
        require(r.get("source_id") in sources,"Visual reference uses an unknown image")
        b=r.get("bbox_xyxy")
        require(isinstance(b,list) and len(b)==4 and all(isinstance(x,(int,float)) for x in b),"Visual bbox must have four normalized coordinates")
        require(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1,"Visual bbox outside native image")


def validate_surface(value,section,sources):
    require(isinstance(value,dict) and isinstance(value.get("regions"),list),"Surface output requires regions")
    expected=[r["id"] for r in section["regions"]];actual=[r.get("region_id") for r in value["regions"]]
    require(set(actual)==set(expected) and len(actual)==len(expected),"Surface audit needs each candidate ID once")
    ordered=[]
    for rid in expected:
        r=next(r for r in value["regions"] if r["region_id"]==rid)
        require(r.get("surface_type") in SURFACE_TYPES|{"non_road"},"Unknown surface class")
        require(r.get("existence") in EXISTENCE,"Invalid existence state")
        require(r.get("confidence") in ("high","medium","low"),"Invalid surface confidence")
        require(isinstance(r.get("evidence"),str),"Surface evidence text missing")
        require(isinstance(r.get("observed_arrows"),list) and set(r["observed_arrows"])<=set(TURNS),"Invalid arrows")
        validate_refs(r.get("visual_refs",[]),sources)
        r={**r,"basis":"observed" if r["observed_arrows"] else "inferred"}
        r["endpoint_eligible"]=r["surface_type"] in MOTOR_TYPES and r["existence"]=="supported"
        ordered.append(r)
    n=0
    for r in ordered:
        r["motor_lane_index"]=None
        if r["endpoint_eligible"]: n+=1;r["motor_lane_index"]=n
    uncertain=sum(r["existence"]=="uncertain" for r in ordered)
    return {"section_id":section["id"],"kind":section["kind"],"direction":section["direction"],"source_id":section["source_id"],
        "regions":ordered,"notes":value.get("notes",[]),"count":{"model_motor_count":n,"unclassified_regions":uncertain,
        "model_count_range":[n,n+uncertain],"human_verified":False,"coverage":"candidate_geometry_only"}}


def validate_usage(value,section,sources):
    require(isinstance(value,dict) and isinstance(value.get("regions"),list),"Usage output requires regions")
    expected={r["region_id"] for r in section["regions"]};actual=[r.get("region_id") for r in value["regions"]]
    require(set(actual)==expected and len(actual)==len(expected),"Usage audit needs each candidate ID once")
    for r in value["regions"]:
        require(r.get("surface_type") in SURFACE_TYPES|{"non_road"},"Fused surface class missing/invalid")
        require(r.get("existence") in EXISTENCE and isinstance(r.get("surface_reason"),str),"Fused existence/evidence missing")
        require(r.get("use") in USES,"Invalid lane-use prediction")
        turns=r.get("allowed_turns")
        require(isinstance(turns,list) and set(turns)<=set(TURNS),"Invalid predicted turn set")
        if r["use"].endswith("_only"): require(turns==[r["use"].replace("_only","")],"Use label and predicted turns inconsistent")
        if r["use"] in ("unknown","not_applicable"): require(not turns,"Unknown use cannot assert turns")
        require(isinstance(r.get("binding"),str),"Lane-use binding text missing")
        validate_refs(r.get("visual_refs",[]),sources)
        require(not turns or r.get("visual_refs"),"A lane-use prediction needs visual support")
    requests=value.get("detail_requests",[])
    require(isinstance(requests,list) and len(requests)<=3,"At most three model-requested details are allowed")
    validate_refs(requests,sources)
    counts=value.get("street_view_counts",[])
    require(isinstance(counts,list),"street_view_counts must be a list")
    for c in counts:
        require(isinstance(c,dict) and c.get("source_id") in sources,"street_view_counts need a supplied street-view source_id")
        require(c.get("motor_lanes") is None or (isinstance(c["motor_lanes"],int) and 0<=c["motor_lanes"]<=12),"motor_lanes must be a count or null")
        require(isinstance(c.get("settled",True),bool),"settled must be true or false")
    return value


def keep_clear_arrow(initial,u):
    """A clearly observed arrow is not overturned by a later view that only fails to read it.

    The use audit may revise a lane to unknown because a street-view crop is worn or occluded.
    That is absence of confirmation, not contrary evidence, so the satellite reading stands.
    A predicted use, a different turn set or a non-motor class from the use audit is left alone.
    """
    seen=[t for t in TURNS if t in initial.get("observed_arrows",[])]
    if not seen or initial.get("visibility")!="clear" or u["use"]!="unknown" or u["allowed_turns"]:
        return u
    if u["surface_type"] not in MOTOR_TYPES or u["existence"]!="supported":
        return u
    return {**u,"use":seen[0]+"_only" if len(seen)==1 else "shared","allowed_turns":seen,"basis":"observed","confidence":"medium",
        "binding":u["binding"]+" [Kept from the satellite surface audit, which saw the arrow clearly; this later view was inconclusive, not contrary.]",
        "fusion_rule":"clear_observation_not_overridden_by_inconclusive_view","use_audit_prediction":u["use"]}


def fuse_surface_usage(section,usage):
    n=0
    for r in section["regions"]:
        u=next(x for x in usage["regions"] if x["region_id"]==r["region_id"])
        r["initial_surface_audit"]={k:r[k] for k in ("surface_type","existence","evidence","observed_arrows","visibility")}
        u=keep_clear_arrow(r["initial_surface_audit"],u)
        r["surface_type"]=u["surface_type"];r["existence"]=u["existence"];r["evidence"]=u["surface_reason"]
        r["lane_use_prediction"]=u
        r["endpoint_eligible"]=r["surface_type"] in MOTOR_TYPES and r["existence"]=="supported"
        r["motor_lane_index"]=None
        if r["endpoint_eligible"]:n+=1;r["motor_lane_index"]=n
    unknown=sum(r["existence"]=="uncertain" for r in section["regions"])
    section["count"].update(model_motor_count=n,unclassified_regions=unknown,model_count_range=[n,n+unknown])
    return section


def evidence_movement_prompt(direction,lanes,usages,contexts,legs,receiving=None,geometry=None,untraced=()):
    """receiving/geometry: legs.receiving and legs.describe for this approach; default is a four-way compass junction.
    Where the geometry pairs every turn with one receiving section the model is given that pairing; otherwise it
    picks each receiving section's turn from the candidates after looking at how the roads meet."""
    from .legs import fixed_turns,NUMBERS
    from .validation import compass_receiving
    receiving=compass_receiving(direction) if receiving is None else receiving
    options=fixed_turns(receiving)
    if options:
        report=(f"Report all {NUMBERS[len(options)]} turns once." if len(options)>1 else "Report the one turn once.")+f" Output directions: {json.dumps(options)}."
    else:
        report=("Report one decision per receiving section. The roads do not meet as a plain cross, so the heading change alone does not name every turn: "
            "for each receiving section choose its turn from its candidate_turns by looking at how the roads meet in the image, and say why in reason. "
            f"Receiving sections: {json.dumps(geometry)}.")
    if untraced:
        report+=(f" Receiving sections {', '.join(untraced)} exist in the road network but could not be traced on the satellite image,"
                 " so they are not offered; do not report turns toward them.")
    return f"""Infer candidate lane-to-lane movements for {direction}. All lane existence and lane-use values below are MODEL PREDICTIONS from images. No reviewer labels or fixed turn answers are supplied.
Combine the full satellite context, source surface evidence and grounded GSV lane-use analysis. Explain conflicts, uncertain region correspondence, shadows and capture dates. A geometrically aligned path is not sufficient to conclude shared use; conversely, an unassigned distant sign should not silently become a restriction on every nearby lane. Decide which evidence applies to which lane and state why.
Use only region endpoints marked supported motor facilities by the surface audit. Do not treat rejected sidewalk/buffer/non-road polygons as travel lanes. IDs are local candidate-region identifiers, not equal lane numbers across views. Geometric width/order are supporting evidence, not proof of existence or permission.
For each turn give candidate with explicit pairs, unresolved if evidence is insufficient, or not_proposed if no supported hypothesis is proposed. Unknown U-turn permission stays unresolved or not_proposed. If your conclusion differs from the model's lane-use prediction, explain the visual reason explicitly; it will be flagged for review, not forcibly changed to a preset answer.
Return JSON {{"movements":[{{"turn":"left|through|right|u_turn","to_direction":"{'|'.join(receiving)}","status":"candidate|unresolved|not_proposed","lane_pairs":[{{"in_region_id":"r#","out_region_id":"r#"}}],"basis":"observed|inferred|mixed","confidence":"high|medium|low","evidence_refs":["section_id:region_id or context_id:observation_id"],"assumptions":[],"reason":"grounded explanation"}}],"gsv_alignment_notes":[]}}.
{report} Network link IDs only, not ground truth: {json.dumps(legs)}.
Surface predictions and model-derived lane-use values: {json.dumps(lanes,ensure_ascii=False)}
Grounded lane-use audit: {json.dumps(usages,ensure_ascii=False)}
GSV contexts: {json.dumps(contexts,ensure_ascii=False)}
"""
