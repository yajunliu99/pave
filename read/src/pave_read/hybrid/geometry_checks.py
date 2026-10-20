"""Relative width, independent envelope coverage, and through-alignment checks.

All pixel comparisons are restricted to the same satellite image. Thresholds
are review heuristics, not metric lane-width standards or calibrated geometry.
"""
from __future__ import annotations
import math
import statistics

from .common import require
from .legs import approaches,receiving,turn_angle,heading

# Exact axes of compass-named sections that record no heading.
AXES={"NB":(0.,-1.),"EB":(1.,0.),"SB":(0.,1.),"WB":(-1.,0.)}


def axis_for(direction, policy=None, heading_deg=None):
    """Travel axis in image pixels: a configured axis, else the section's heading, else its compass name."""
    axis=(policy or {}).get("axes",{}).get(direction)
    if axis is None:
        h=math.radians(heading_deg) if heading_deg is not None else None
        axis=AXES[direction] if h is None else (math.sin(h),-math.cos(h))
    length=math.hypot(*axis)
    require(length>0,"Zero corridor axis")
    return axis[0]/length,axis[1]/length


def project_polygon(polygon, axis):
    ux,uy=axis; nx,ny=-uy,ux
    return [(x*ux+y*uy,x*nx+y*ny) for x,y in polygon]


def lateral_interval(projected, station):
    crossings=[]
    for (s1,t1),(s2,t2) in zip(projected,projected[1:]+projected[:1]):
        if (s1<=station<s2) or (s2<=station<s1):
            crossings.append(t1+(t2-t1)*(station-s1)/(s2-s1))
    return (min(crossings),max(crossings)) if len(crossings)>=2 else None


def width_samples(polygon, axis):
    projected=project_polygon(polygon,axis)
    lo,hi=min(p[0] for p in projected),max(p[0] for p in projected)
    values=[]
    for fraction in (.2,.5,.8):
        interval=lateral_interval(projected,lo+(hi-lo)*fraction)
        if interval: values.append(interval[1]-interval[0])
    require(values and min(values)>0,"Could not sample a region's cross-sectional width")
    return values


def outside_envelope_fraction(polygon,envelope,axis):
    """Fraction of sampled transverse width outside an independently stored envelope."""
    region=project_polygon(polygon,axis); boundary=project_polygon(envelope,axis)
    low,high=min(p[0] for p in region),max(p[0] for p in region)
    fractions=[]
    for f in (.2,.5,.8):
        station=low+(high-low)*f; interval=lateral_interval(region,station)
        if interval:
            outer=lateral_interval(boundary,station)
            covered=0 if outer is None else max(0,min(interval[1],outer[1])-max(interval[0],outer[0]))
            fractions.append(1-covered/(interval[1]-interval[0]))
    return max(fractions,default=1.)


def uncovered_width(envelope, intervals):
    lo,hi=envelope
    clips=sorted((max(lo,a),min(hi,b)) for a,b in intervals if b>lo and a<hi)
    covered=0.; end=lo
    for a,b in clips:
        if b>max(a,end): covered+=b-max(a,end)
        end=max(end,b)
    return max(0.,hi-lo-covered)


def through_alignment(a,b,direction,reference_width,policy=None,heading_deg=None):
    ux,uy=axis_for(direction,policy,heading_deg); nx,ny=-uy,ux
    dx,dy=b[0]-a[0],b[1]-a[1]
    along=dx*ux+dy*uy; lateral=dx*nx+dy*ny
    ratio=abs(lateral)/reference_width
    angle=math.degrees(math.atan2(abs(lateral),max(along,1e-9)))
    policy=policy or {}
    hard_limit=policy.get("straight_hard_lane_widths_by_direction",{}).get(direction,policy.get("straight_hard_lane_widths",1.5))
    return {"along_px":round(along,3),"lateral_px":round(lateral,3),
        "lateral_in_lane_widths":round(ratio,3),"angle_deg":round(angle,3),
        "review_required":along<=0 or ratio>policy.get("straight_review_lane_widths",.65) or angle>policy.get("straight_review_angle_deg",8),
        "hard_limit_lane_widths":hard_limit,"hard_violation":along<=0 or ratio>hard_limit}


def audit_geometry(config):
    policy=config.get("geometry_checks",{})
    sections={s["id"]:s for s in config["sections"]}
    bysource={}
    for ref in policy.get("reference_regions",[]):
        sid,rid=ref.split(":",1); section=sections[sid]
        require(config["sources"][section["source_id"]]["role"]=="satellite","Width references must be satellite regions")
        region=next(r for r in section["regions"] if r["id"]==rid)
        width=statistics.median(width_samples(region["polygon"],axis_for(section["direction"],policy,section.get("heading_deg"))))
        bysource.setdefault(section["source_id"],[]).append({"ref":ref,"width_px":width})
    references={sid:{"width_px":round(statistics.median(x["width_px"] for x in vals),4),"samples":vals}
                for sid,vals in bysource.items()}
    # A rerun after a review revision keeps the first run's reference, so the width hints (and the cached
    # answers) of the sections that did not change stay the same.
    for sid,width in policy.get("frozen_reference_width_px",{}).items():
        if sid in references:
            references[sid].update({"median_of_current_samples_px":references[sid]["width_px"],"width_px":width,"frozen":True})
    audit={"coordinate_units":"pixels in the same satellite image; no meter calibration",
           "thresholds_are_heuristics":True,"references":references,"sections":{},"straight_pairs":{},"opposing_overlap":{"checked":False}}
    for section in config["sections"]:
        sid=section["source_id"]
        if sid not in references: continue
        reference=references[sid]["width_px"]; axis=axis_for(section["direction"],policy,section.get("heading_deg"))
        records={}
        for r in section["regions"]:
            samples=width_samples(r["polygon"],axis); width=statistics.median(samples); ratio=width/reference
            flags=[]
            if ratio>policy.get("merged_width_ratio",1.65): flags.append("possible_merged_lanes")
            if ratio<policy.get("minimum_motor_width_ratio",.65): flags.append("narrow_region_check_facility")
            outside=None
            if section.get("carriageway_envelope"):
                outside=outside_envelope_fraction(r["polygon"],section["carriageway_envelope"],axis)
                if outside>policy.get("outside_envelope_fraction",.1): flags.append("outside_reviewed_carriageway")
            records[r["id"]]={"width_samples_px":[round(x,3) for x in samples],"median_width_px":round(width,3),
                "reference_width_px":reference,"relative_width":round(ratio,3),"flags":flags,
                "single_motor_lane_width_supported":"outside_reviewed_carriageway" not in flags and (not flags or bool(r.get("width_exception_reason"))),
                "width_exception_reason":r.get("width_exception_reason")}
            if outside is not None: records[r["id"]]["outside_envelope_fraction"]=round(outside,4)
        coverage={"checked":False}
        if section.get("carriageway_envelope"):
            envelope=project_polygon(section["carriageway_envelope"],axis)
            low,high=min(x[0] for x in envelope),max(x[0] for x in envelope)
            gaps=[]
            for f in (.2,.5,.8):
                station=low+(high-low)*f; outer=lateral_interval(envelope,station)
                if outer:
                    intervals=[v for r in section["regions"] if (v:=lateral_interval(project_polygon(r["polygon"],axis),station))]
                    gaps.append(uncovered_width(outer,intervals))
            coverage={"checked":True,"source":"independently stored reviewed envelope",
                      "uncovered_width_samples_px":[round(g,3) for g in gaps],
                      "missing_region_review":bool(gaps and max(gaps)>reference*.5)}
        audit["sections"][section["id"]]={"source_id":sid,"regions":records,"envelope_coverage":coverage}
    for a in approaches(config["sections"]):
        direction=a["direction"]
        # The receiving section most nearly straight ahead, among those through traffic could reach.
        ahead=[sections["out_"+d] for d,turns in receiving(config["sections"],a).items() if "through" in turns]
        if not ahead:
            audit["straight_pairs"][direction]={"comparable":False,"reason":"no receiving section ahead"}; continue
        b=min(ahead,key=lambda s:abs(turn_angle(heading(a),heading(s))))
        if a["source_id"]!=b["source_id"] or a["source_id"] not in references:
            audit["straight_pairs"][direction]={"comparable":False}; continue
        ref=references[a["source_id"]]["width_px"]
        pairs=[]
        for i in a["regions"]:
            for o in b["regions"]:
                pairs.append({"in_region_id":i["id"],"out_region_id":o["id"],
                              **through_alignment(i["gate_point"],o["gate_point"],direction,ref,policy,a.get("heading_deg"))})
        audit["straight_pairs"][direction]={"comparable":True,"out_section_id":b["id"],"axis_xy":axis_for(direction,policy,a.get("heading_deg")),
            "reference_width_px":ref,"pairs":pairs}
    return audit
