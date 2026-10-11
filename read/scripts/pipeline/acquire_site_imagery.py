"""Satellite and street-view imagery for one network node, and the configs that start its pipeline.

Writes into --output: site.json (tracing), base_config.json (classification sources, street-view bands and
policies), exit_views.json (exit audit and atlas), acquisition.json (every request, snap and rejection) and
plan.json. --plan-only lists the requests and their budget without contacting any service; --cache-only
replays earlier requests. Satellite and Street View images are billed by their providers; metadata is not.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.control_acquisition import AcquisitionUnavailable, StreetViewClient
from movement_fixer.fusion.lane_acquisition import acquire, plan_site, satellite_client, site_files
from movement_fixer.hybrid.common import read_json, write_json


def plan_record(plan):
    legs = [{k: v for k, v in leg.items() if k != "line"} for leg in plan["legs"]]
    return {"node_id": plan["node_id"], "name": plan["name"], "center_lonlat": plan["center_lonlat"],
            "viewport": plan["viewport"].serialize(), "ground_mpp": plan["ground_mpp"], "legs": legs,
            "budget": plan["budget"], "policy": plan["policy"], "network": plan["network"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", type=int, required=True)
    p.add_argument("--node-csv", type=Path, required=True)
    p.add_argument("--link-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", type=Path, help="JSON overrides of lane_acquisition.POLICY")
    p.add_argument("--gsv-cache", type=Path, default=ROOT / "cache/gsv_lanes")
    p.add_argument("--satellite-cache", type=Path, default=ROOT / "cache/mapbox_sites")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = plan_site(a.node_csv.resolve(), a.link_csv.resolve(), a.node_id, read_json(a.policy) if a.policy else None)
    write_json(output / "plan.json", plan_record(plan))
    sections = [f"{leg['section_id']} (arm {leg['arm']}, link {leg['link_id']})" for leg in plan["legs"]]
    print(json.dumps({"node": plan["node_id"], "name": plan["name"], "sections": sections, "requests": plan["budget"]}, ensure_ascii=False, indent=2))
    if a.plan_only:
        return 0
    policy = plan["policy"]
    street = StreetViewClient(ROOT, a.gsv_cache, {"image_size": policy["image_size"], "max_metadata_requests": policy["max_metadata_requests"],
                                                  "max_image_requests": policy["max_image_requests"]}, cache_only=a.cache_only)
    satellite = satellite_client(policy, ROOT, a.satellite_cache, cache_only=a.cache_only)
    write_json(output / "status.json", {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        result = acquire(plan, street, satellite)
    except AcquisitionUnavailable as error:
        write_json(output / "status.json", {"state": "failed", "reason": str(error)})
        raise SystemExit(f"Acquisition stopped: {error}")
    site, base, manifest = site_files(plan, result, ROOT)
    write_json(output / "site.json", site)
    write_json(output / "base_config.json", base)
    write_json(output / "exit_views.json", manifest)
    write_json(output / "acquisition.json", {k: v for k, v in result.items() if k != "views"} | {"views": [
        {k: v for k, v in view.items() if k != "camera"} | {"section_id": view["camera"]["section_id"], "position": view["camera"]["position"]}
        for view in result["views"]]})
    state = "complete" if not result["stopped"] else "partial"
    write_json(output / "status.json", {"state": state, "stopped": result["stopped"], "missing": result["missing"],
                                        "counts": result["counts"], "finished_utc": datetime.now(timezone.utc).isoformat()})
    print({"state": state, "views": len(result["views"]), "missing": len(result["missing"]), "counts": result["counts"], "output": str(output)})
    return 0 if state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
