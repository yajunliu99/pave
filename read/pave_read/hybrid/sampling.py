"""Audit configured GSV distance bands from actual panorama positions."""
import math
from itertools import combinations

from .common import require
from .legs import approaches,is_upstream


def offsets(lat,lon,center):
    return ((lat-center[0])*111000,(lon-center[1])*111000*math.cos(math.radians(center[0])))


def audit_sampling(config):
    policy=config.get("gsv_sampling")
    if not policy:
        return {"checked":False,"reason":"No multi-position sampling policy configured"}
    positions=policy["positions"]; center=policy["center_lat_lon"]
    require({"near","mid"}<=set(positions)<= {"near","mid","far"},"GSV sampling requires near/mid and permits far")
    position_order=sorted(positions,key=lambda p:positions[p]["target_m"])
    rows=[]; seen=set(); sources=config["sources"]
    upstream={s["direction"]:s for s in approaches(config["sections"])}
    for view in config.get("context_views",[]):
        direction=view["direction"]; position=view.get("sampling_position")
        require(direction in upstream and position in positions,"Unknown GSV direction/position")
        key=(direction,position)
        require(key not in seen,"Duplicate GSV direction/position"); seen.add(key)
        source=sources[view["source_id"]]
        require(not source.get("reverse_view",False),"Required distance-band views must face forward")
        require(source.get("sampling_position")==position and source.get("direction")==direction,"GSV source/view position mismatch")
        n,e=offsets(source["actual_lat"],source["actual_lon"],center); distance=math.hypot(n,e)
        require(abs(distance-source["distance_to_center_m"])<.1,"GSV recorded distance does not match actual coordinates")
        low,high=positions[position]["allowed_distance_m"]
        require(low<=distance<=high,f"{direction} {position} outside configured sampling band")
        require(is_upstream(upstream[direction],n,e),"Required GSV point is not upstream of the approach")
        rows.append({"direction":direction,"position":position,"target_m":positions[position]["target_m"],
            "actual_distance_m":round(distance,2),"north_offset_m":round(n,2),"east_offset_m":round(e,2),
            "context_id":view["id"],"source_id":view["source_id"],"pano_id":source["pano_id"],
            "capture_date":source.get("capture_date"),"view_settings":source.get("view_settings"),
            "actual_lat":source["actual_lat"],"actual_lon":source["actual_lon"]})
    # A band acquisition could not fill is listed as missing, with its reason, rather than silently absent.
    missing={(m["direction"],m["position"]) for m in policy.get("missing",[])}
    require(not seen&missing,"A GSV band is both present and listed as missing")
    require(seen|missing=={(d,p) for d in upstream for p in positions},"Every approach requires all configured GSV distance bands")
    separations={}; date_comparisons={}
    for direction in upstream:
        selected=[next(r for r in rows if r["direction"]==direction and r["position"]==p) for p in position_order if (direction,p) in seen]
        pairs={}
        for a,b in combinations(selected,2):
            require(a["pano_id"]!=b["pano_id"],"Different distance bands must not reuse the same panorama")
            require(a["actual_distance_m"]<b["actual_distance_m"],"GSV distances do not follow configured band order")
            separation=math.hypot(a["north_offset_m"]-b["north_offset_m"],a["east_offset_m"]-b["east_offset_m"])
            require(separation>=policy.get("minimum_position_separation_m",10),"GSV distance-band positions are too close")
            pairs[a["position"]+"_"+b["position"]]=round(separation,2)
        separations[direction]=pairs
        dates={r["position"]:r["capture_date"] for r in selected}
        mismatch=len(set(dates.values()))>1 or None in dates.values() or not dates
        date_comparisons[direction]={"dates":dates,"mixed_or_unknown_dates":mismatch,
            "review_required":mismatch,"note":"Different image dates must not be interpreted as a purely spatial lane transition." if mismatch else "Same recorded month is not proof of simultaneous capture."}
    return {"checked":True,"complete":not missing,"missing":sorted(missing),"position_order":position_order,"positions":rows,"unique_panoramas":len({r["pano_id"] for r in rows}),
        "position_separation_m":separations,"date_comparisons":date_comparisons,"distance_reference":"intersection center, not stopbar",
        "note":"Coverage check only; comparable locations do not prove visible complete lane cross-sections."}
