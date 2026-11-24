import json

SYSTEM = """You are a transportation-image analyst. Distinguish visible evidence, geometric inference and unknowns.
Region IDs and proposed boundaries are annotation aids, not semantic labels or ground truth.
Return structured JSON only. Never invent visible arrows, image dates or confirmed legal permissions."""


def lane_prompt(section, source, yolo_hints, geometry_audit=None):
    role = section["kind"]
    base=f"""Inspect every surface-region panel for {section['id']} ({role}), travel direction {section['direction']}.
For satellite images, panels were rotated so actual traffic travel points UP. An outbound_receiving section is the outgoing carriageway after the crossing, not the opposing incoming carriageway.
For GSV reverse_view={source.get('reverse_view',False)}, the original perspective is retained: incoming vehicles move toward the camera in a reverse view, and driver-left generally appears image-right.
IDs are ordered driver-left to driver-right within the supplied set. They do not specify facility class or a complete lane count.
Gray padding is outside-region missing context. Original region pixels were preserved before display scaling.
Classify each region from its pavement, edges, symbols, color and occupancy. Rails may be shared with motor vehicles; occupancy alone does not establish legal use. Distinguish bicycle strips, shoulders/parking and median/buffer areas.
Do NOT interpret a green strip as a motor lane merely because it contains a straight arrow. Do NOT default an unmarked region to an observed through arrow.
YOLO hints are rectangle-union coverage estimates, not true segmentation or proof of visibility. Missed detections are possible. A low coverage value does not prove unobstructed pavement.
Return JSON:
{{"regions":[{{"region_id":"one supplied ID", "surface_type":"motor_vehicle_lane|shared_rail_motor_lane|bicycle_strip|rail_area|median_or_buffer|shoulder_or_parking|unknown",
"observed_arrows":["left|through|right|u_turn"],"visibility":"clear|partial|occluded|not_visible|unclear",
"basis":"observed|inferred|mixed","confidence":"high|medium|low","evidence":"visible support and uncertainty"}}],
"notes":["coverage, geometry or ambiguity notes"]}}
Supply each ID exactly once. Use empty observed_arrows when direction cannot be read; geometric direction hypotheses are a later stage.
REGION GEOMETRY QUALITY: {json.dumps([{'region_id':r['id'],'quality':r.get('quality')} for r in section['regions']])}
COVERAGE: {section.get('coverage')} / {section.get('notes','')}
SOURCE DATE: {source.get('capture_date') or 'unknown'}
YOLO AUXILIARY HINTS: {json.dumps(yolo_hints,ensure_ascii=False)}
"""
    if geometry_audit:
        base += "\nRELATIVE WIDTH AND COVERAGE CHECKS (same satellite pixels, not meter calibration): "+json.dumps(geometry_audit,ensure_ascii=False)+"\nA region roughly twice the clear single-lane width may merge two lanes. Do not call it one motor lane merely because cars are present. Narrow green/buffer regions need separate classification. Width alone does not establish legal use. Flag uncertain geometry rather than inventing missing boundary details.\n"
    reviews=[{"region_id":r["id"],"review_annotation":r["review_annotation"]} for r in section["regions"] if r.get("review_annotation")]
    if reviews:
        base += "\nUSER REVIEW INPUT (external evidence, not a model-observed marking): "+json.dumps(reviews,ensure_ascii=False)+"\nUse the explicit reviewed facility annotation as external support, and distinguish that basis in evidence. Do NOT copy its turn into observed_arrows unless a glyph is actually readable; movement inference can use the reviewed direction separately.\n"
    return base


def context_prompt(view,source,yolo):
    return f"""Inspect this FORWARD-facing GSV context for {view['direction']} approaching the intersection. It is a primary context view, not a backward/reverse view.
Extract directly visible pavement arrows, bicycle/rail facilities, lane/curb/median relationships, occlusion and readable restrictions. Near-intersection views may show crossing or outgoing pavement: state whether each observation applies to the target incoming approach.
Do not infer straight arrows or turn permission from moving cars, green signals or missing signs. Do not invent a full lane count or force an observation onto a numbered region.
YOLO detections are fallible object-location hints. Their absence does not prove clear pavement. Region classifications elsewhere may come from a reverse view of the same panorama, which is not independent evidence.
Return JSON: {{"observations":[{{"id":"o1, o2, ...", "kind":"arrow|facility|boundary|sign|occlusion|other", "observed_arrows":["left|through|right|u_turn"], "applies_to_incoming":"yes|no|uncertain", "evidence":"visible support and location"}}],"notes":["limitations"]}}
Source date: {source.get('capture_date')}; distance to center (not stopbar): {source.get('distance_to_center_m')} m.
Sampling position: {view.get('sampling_position','unspecified')}; nominal target: {source.get('target_distance_m')} m. Describe this position locally. Do not equate an upstream cross-section with the stopbar: turn pockets, lane splits and facility changes may occur between them. The supplied capture date limits temporal applicability; older imagery cannot prove the layout in newer images. Panorama: {source.get('pano_id')}.
YOLO object hints: {json.dumps(yolo.get('detections',[]),ensure_ascii=False)}
"""


def movement_prompt(direction, lanes, legs, source_dates, geometry_notes, geometry_audit=None):
    from .validation import receiving_direction
    options={turn:{"out_direction":receiving_direction(direction,turn),
                   "ob_link_id":legs[receiving_direction(direction,turn)]["out_link_id"]}
             for turn in ("left","through","right","u_turn")}
    # No benchmark values, analyst reference answers or baseline lane ranges.
    base=f"""Infer CANDIDATE lane-to-lane movements for incoming direction {direction} at this intersection.
The image is a north-up satellite overview with blue incoming and green receiving-section geometry. in_* sections are upstream of the stop line; out_* sections are downstream receiving carriageways. Labels are NOT proof of lane class or complete coverage.
Use the independently extracted satellite INBOUND and OUTBOUND regions and GSV evidence below. GSV region IDs are local subsets, not the same ordinal as satellite lane IDs. Match any GSV support using actual facility/curb/median/rail anchors; do not match numbers alone.
Evaluate evidence in this order: lane existence within the reviewed carriageway, lane-specific use from user review/visible arrows/signs, then geometric continuity and receiving capacity. Plausible width or straight alignment cannot establish that pavement is a real lane, and cannot override a turn restriction.
For an unmarked motor lane with no conflicting restriction evidence, geometric reasoning may support a through candidate. A missing arrow alone is not a prohibition, but an unresolved possible ONLY sign or dedicated-turn conflict must not be bypassed by a shared-use assumption. Omit the conflicting lane pair or leave that movement unresolved until applicability is established.
The supplied turn_constraint.allowed_turns is a HARD export constraint. User-reviewed exclusive turns and directly observed pavement-arrow sets cannot be overridden by geometry, a confidence score or assumptions. Nonmatching lane pairs are rejected. Do not infer U-turn permission from a left arrow or absent sign; without specific support report U-turn unresolved or not_proposed.
Pair only regions classified motor_vehicle_lane or shared_rail_motor_lane. Do not use bicycle, rail-only, shoulder, median or unknown regions as motor endpoints.
Excluded regions are a historical audit of rejected lane geometry, not available endpoints. Do not recreate them, include them in lane counts or use them to match receiving capacity.
Choose receiving lanes using road geometry and consistent driver-left-to-right ordering. Report exact region pairs rather than inventing a count. Flag merge/fan-out and ambiguous receiving alignment in assumptions. A lane pair is a candidate, not proven legal connectivity.
For all four turns, return one decision. Use candidate with nonempty lane_pairs when a defensible geometric hypothesis exists. Use unresolved/not_proposed with empty pairs when it does not; do not force a complete graph.
Return JSON:
{{"movements":[{{"turn":"left|through|right|u_turn","to_direction":"NB|EB|SB|WB",
"status":"candidate|unresolved|not_proposed","lane_pairs":[{{"in_region_id":"r# from in_{direction}","out_region_id":"r# from selected out section"}}],
"basis":"observed|inferred|mixed","confidence":"high|medium|low",
"evidence_refs":["section_id:region_id, e.g. in_NB:r1 or gsv_NB:r2"],
"assumptions":["explicit geometric assumptions, not invented observations"],
"reason":"justify source-lane choice, receiving-lane choice and uncertainty"}}],
"gsv_alignment_notes":["source correspondence and coverage limitations"]}}
Incoming link ID: {legs[direction]['in_link_id']}. Valid turn/outgoing options: {json.dumps(options)}
Available lane evidence (counts are candidate counts, not human truth): {json.dumps(lanes,ensure_ascii=False)}
Image dates/roles: {json.dumps(source_dates,ensure_ascii=False)}
Geometry notes: {json.dumps(geometry_notes,ensure_ascii=False)}
Satellite capture date is unknown in this pilot. Keep all outputs reviewable; do not claim contemporaneous layouts or confirmed movement legality.
"""
    base += "\nForward GSV contexts are primary for approach interpretation; reverse-view region evidence is auxiliary. Their source roles are explicit in the metadata. Do not count two views of one panorama as independent confirmations. Cite context observations as section_id:o# when used.\n"
    base += "\nReview ALL supplied distance bands: near (~25 m), mid (~50 m), and far (~100 m) when supplied. Compare observations from every band, explain conflicts or missing visibility in gsv_alignment_notes and assumptions, and use actual position/date/panorama metadata. Far and mid can reveal upstream lane development, but their lane count, ordering or use cannot be transferred to the stopbar without supported continuity; account for lane splits and turn pockets. Explicitly discuss differences in capture dates. Older imagery is historical/contextual support, not confirmation of newer lane permissions or continuity. Do not interpret temporal layout changes as spatial lane transitions. Distinct panoramas are distinct positions, not automatically independent legal confirmations.\n"
    base += "\nAny reviewed_directions/review_annotation field is separately supplied user review, not a visible arrow detected by the model. Preserve that distinction in the movement basis and assumptions.\n"
    if geometry_audit:
        base += "\nSTRAIGHT-CORRIDOR ALIGNMENT CHECKS: "+json.dumps(geometry_audit.get("straight_pairs",{}).get(direction,{}),ensure_ascii=False)
        base += "\nFor THROUGH connections prefer receiving lanes continuing the same corridor with small lateral displacement. For EB this normally means approximately horizontal connections in the original north-up image. Do not use ordinal equality alone. Large lateral shifts need explicit geometric justification; hard-violation pairs cannot be exported. These constraints do not require left/right turns to be straight.\n"
        base += "\nA motor region with endpoint_eligible=false is geometrically unresolved and must not be used as a lane endpoint.\n"
    return base
