"""Build inference inputs by allowlist, without reviewer answers or exclusions."""
import copy

from .common import require

FORBIDDEN={"review_annotation","reviewed_directions","turn_constraint","excluded_regions",
    "carriageway_envelope","envelope_review_source","reference_count","allowed_turns",
    "intended_directions","exclusive","semantic_revision"}


def select(value,keys):
    return {k:copy.deepcopy(value[k]) for k in keys if k in value}


def assert_no_answer_fields(value):
    if isinstance(value,dict):
        require(not (set(value)&FORBIDDEN),"Inference input contains reviewer/answer fields")
        for child in value.values(): assert_no_answer_fields(child)
    elif isinstance(value,list):
        for child in value: assert_no_answer_fields(child)
    elif isinstance(value,str):
        require("user correction" not in value.lower() and "user-reviewed" not in value.lower(),"Reviewer prose leaked into inference input")


def build_evidence_config(proposals):
    cfg=select(proposals,["schema_version","pilot_id","node_id","name","legs","network","vlm","yolo"])
    cfg["inference_mode"]="visual_evidence"
    cfg["sources"]={sid:select(s,["kind","path","pano_path","role","direction","capture_date","date_status",
        "reverse_view","camera_role","distance_to_center_m","pano_id","projection","sampling_position",
        "target_distance_m","actual_lat","actual_lon","selection_label","view_settings"])
        for sid,s in proposals["sources"].items()}
    for group in ("sections","gsv_views"):
        cfg[group]=[]
        for section in proposals[group]:
            s=select(section,["id","source_id","kind","direction","rotation_ccw","color"])
            s["regions"]=[select(r,["id","polygon","gate_point"]) for r in section["regions"]]
            s["coverage"]="candidate_regions_only"
            s["geometry_source"]="previously_drawn_candidate_geometry"
            s["notes"]="Candidate regions are unvalidated hypotheses and may contain non-road pavement or multiple facilities."
            cfg[group].append(s)
    cfg["context_views"]=[select(v,["id","source_id","kind","direction","sampling_position"]) for v in proposals["context_views"]]
    cfg["gsv_sampling"]=select(proposals["gsv_sampling"],["center_lat_lon","positions","minimum_position_separation_m"])
    cfg["geometry_checks"]=select(proposals["geometry_checks"],["reference_regions","axes","merged_width_ratio",
        "minimum_motor_width_ratio","straight_review_lane_widths","straight_hard_lane_widths","straight_review_angle_deg",
        "straight_hard_lane_widths_by_direction"])
    cfg["reference"]={"status":"withheld_from_inference","file":None}
    cfg["far_upstream"]={"status":"context_views_only_not_full_lane_crosssections"}
    assert_no_answer_fields(cfg)
    return cfg
