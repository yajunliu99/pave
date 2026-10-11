"""Traffic-control observations and camera targeting, independent of lane truth."""
import math
from ..hybrid.common import require
from ..hybrid.evidence_reasoning import validate_refs

KINDS={"traffic_signal","pedestrian_signal","lane_use_sign","turn_restriction_sign","regulatory_sign","street_name_sign","other_sign"}


def detail_pose(view,bbox,min_fov=15.,max_fov=55.):
    """Re-aim a perspective ray at the selected native-image box; no depth claim."""
    x1,y1,x2,y2=bbox;w,h=view["image_size"]
    f=.5*w/math.tan(math.radians(view["hfov_deg"])/2)
    x=((x1+x2)/2-.5)*w;y=((y1+y2)/2-.5)*h;t=math.radians(view["pitch_deg"])
    forward=y*math.sin(t)+f*math.cos(t);up=f*math.sin(t)-y*math.cos(t)
    heading=(view["compass_heading_deg"]+math.degrees(math.atan2(x,forward)))%360
    pitch=math.degrees(math.atan2(up,math.hypot(x,forward)))
    angular=math.degrees(2*math.atan(max((x2-x1)*w,(y2-y1)*h)/(2*f)))
    return {"compass_heading_deg":round(heading,3),"pitch_deg":round(max(-75,min(75,pitch)),3),"hfov_deg":round(max(min_fov,min(max_fov,angular*1.7)),2)}


def validate_inventory(value,sources):
    require(isinstance(value,dict),"Inventory must be an object")
    requests=value.get("detail_requests",[])
    require(isinstance(requests,list) and len(requests)<=4,"At most four targeted details per approach")
    validate_refs(requests,sources)
    for r in requests:require(isinstance(r.get("question"),str),"Detail needs a concrete reading question")
    require(isinstance(value.get("observations"),list),"Inventory observations missing")
    return value


def validate_controls(value,region_ids,sources):
    require(isinstance(value,dict) and isinstance(value.get("controls"),list),"Controls must be a list")
    require(len(value["controls"])<=16,"Group repeated views of the same object; at most 16 hypotheses")
    ids=[]
    for c in value["controls"]:
        ids.append(c.get("id"));require(isinstance(c.get("id"),str),"Control ID missing")
        require(c.get("kind") in KINDS,"Invalid control kind")
        require(c.get("confidence") in {"high","medium","low"},"Invalid control confidence")
        require(c.get("readability") in {"readable","partial","unreadable","not_applicable"},"Invalid readability")
        require(c.get("face_status") in {"front","back","side","unclear"},"Invalid face status")
        require(c.get("binding_status") in {"candidate","unresolved","not_applicable"},"Invalid binding status")
        require(c.get("applies_to") in {"approach","lane_candidates","other_approach","pedestrians","unknown"},"Invalid governing scope")
        require(isinstance(c.get("lane_candidates"),list) and set(c["lane_candidates"])<=set(region_ids),"Unknown lane candidate")
        require(c["applies_to"]=="lane_candidates" or not c["lane_candidates"],"Non-lane scope must not create lane bindings")
        require(c["binding_status"]=="candidate" or not c["lane_candidates"],"Unresolved scope cannot assert lane candidates")
        for k in ("observed_text","symbol","description","binding_reason"):require(isinstance(c.get(k),str),"Missing control text field "+k)
        require(c.get("visual_refs"),"A control observation needs native-image evidence")
        validate_refs(c["visual_refs"],sources)
    require(len(ids)==len(set(ids)),"Duplicate control IDs")
    value["signal_phase_plan"]="not_observable_from_static_imagery"
    value["movement_constraints_applied"]=False
    value["human_verified"]=False
    requests=value.get('resampling_requests',[])
    require(isinstance(requests,list) and len(requests)<=2,'At most two resampling requests')
    validate_refs(requests,sources)
    for r in requests:require(r.get('problem') in ('small_target','occluded','oblique','unreadable_text','unresolved_arrow'),'Invalid resampling reason')
    return value


def needs_resampling(result):
    """Uncertain lane binding alone cannot trigger endless street-view requests."""
    if not result.get('controls'):return True  # Empty detection is not verified absence.
    if result.get('resampling_requests'):return True
    for c in result.get('controls',[]):
        if c['applies_to']=='other_approach' or c['kind'] in ('street_name_sign','other_sign'):continue
        if c['readability'] in ('partial','unreadable') or c['face_status'] in ('back','side'):return True
        if c['kind']=='traffic_signal' and any(t in c['symbol'].lower() for t in ('unknown','unresolved','unclear')):return True
    return False


def reading_signature(result):
    readings=set()
    for c in result.get('controls',[]):
        text=' '.join(c.get('observed_text','').upper().split())
        symbol=' '.join(c.get('symbol','').lower().split())
        if text and '[?]' not in text:readings.add((c['kind'],'text',text))
        if symbol and not any(t in symbol for t in ('unknown','unresolved','unclear')):readings.add((c['kind'],'symbol',symbol))
    return readings
