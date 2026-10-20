"""Classify graph/display roles from model outputs, never from region IDs."""
MOTOR={"motor_vehicle_lane","shared_rail_motor_lane"}
FACILITIES={"bicycle_strip","rail_area"}
NON_TRAVEL={"non_road","median_or_buffer","shoulder_or_parking"}


def region_role(surface,existence):
    if existence=="rejected":return "rejected_region"
    if existence!="supported" or surface=="unknown":return "unresolved_candidate"
    if surface in NON_TRAVEL:return "rejected_region"
    if surface in MOTOR:return "motor_lane_candidate"
    if surface in FACILITIES:return "non_motor_facility"
    return "unresolved_candidate"


def apply_presence(region):
    role=region_role(region["model_surface_type"],region["model_existence"])
    region["graph_role"]=role
    region["object_status"]={"rejected_region":"rejected_not_a_lane","unresolved_candidate":"unresolved_candidate",
        "motor_lane_candidate":"model_supported_motor_lane_not_human_verified","non_motor_facility":"model_supported_non_motor_facility"}[role]
    region["is_motor_lane"]=role=="motor_lane_candidate"
    region["movement_endpoint_eligible"]=region["is_motor_lane"]
    region["motor_use_applicable"]=False if role in ("rejected_region","non_motor_facility") else True if role=="motor_lane_candidate" else None
    return region
