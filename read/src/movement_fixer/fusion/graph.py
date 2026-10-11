"""Immutable observations, candidate lane segments and auditable graph revisions."""
import copy
import hashlib
import json
import cv2
import numpy as np


def stable_id(prefix,value):
    return prefix+hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]


def time_relation(capture_date,window):
    """How a capture month relates to the target window (start, end) of a run; without a window, only whether it is dated."""
    if not capture_date:return "unknown_date"
    if not window:return "no_target_window"
    if capture_date<window[0]:return "historical"
    if capture_date>window[1]:return "later"
    return "target_window"


def overlap_fraction(polygon,other):
    a=cv2.convexHull(np.asarray(polygon,np.float32));b=cv2.convexHull(np.asarray(other,np.float32))
    area=abs(cv2.contourArea(a))
    if area<1e-9:return 0.
    intersection,_=cv2.intersectConvexConvex(a,b)
    return max(0.,min(1.,float(intersection)/area))


class LaneGraph:
    def __init__(self,segments,observations=None,associations=None):
        self.segments=copy.deepcopy({s["id"]:s for s in segments})
        self.observations=copy.deepcopy(observations or [])
        self.associations=copy.deepcopy(associations or [])
        self.events=[]

    def replace_segments(self,operation,parent_ids,children,evidence_ids):
        if operation not in ("split","merge","revise","add","reject"):raise ValueError("Unknown graph operation")
        if not evidence_ids:raise ValueError("Graph revisions must cite evidence")
        known={o["id"] for o in self.observations}
        if any(x not in known for x in evidence_ids):raise ValueError("Graph revision references unknown evidence")
        if any(x not in self.segments or self.segments[x].get("superseded") for x in parent_ids):raise ValueError("Invalid graph parent")
        new_ids=[s["id"] for s in children]
        if len(set(new_ids))!=len(new_ids) or any(x in self.segments for x in new_ids):raise ValueError("Child ID already exists")
        if operation=="split" and (len(parent_ids)!=1 or len(children)<2):raise ValueError("Split cardinality")
        if operation=="merge" and (len(parent_ids)<2 or len(children)!=1):raise ValueError("Merge cardinality")
        event={"id":f"revision_{len(self.events)+1}","operation":operation,"parents":parent_ids,"children":new_ids,"evidence_ids":evidence_ids}
        for pid in parent_ids:self.segments[pid]["superseded"]=True
        for child in children:self.segments[child["id"]]={**copy.deepcopy(child),"parents":list(parent_ids)}
        self.events.append(event)
        return event

    def serialize(self):
        return {"segments":list(self.segments.values()),"observations":self.observations,"associations":self.associations,"revisions":self.events}
