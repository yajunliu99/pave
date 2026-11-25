"""Read the traffic signals and signs of a junction whose lanes run_read.py has read. An optional step after it.

    python scripts/pipeline/run_signals.py --imagery <imagery folder> --read <run_read output> [--policy <policy.json>]

For every approach (each motor-vehicle link entering the node):
1. Cameras: three positions placed from the link geometry (upstream, near the junction, at it), each taken from the
   panorama the Street View metadata service returns there, checked for position, carriageway and date. Each view
   looks along the approach toward the junction, 100 degrees wide.
2. Detection: YOLO marks traffic lights and stop signs in these views, as proposals for the model to look at.
3. Inventory: the model sees the views, and the approach's traced lanes (r1, r2, ... from lanes.json, travel up), and
   asks for up to four close-ups: the same panorama re-aimed at a box, 15 to 55 degrees wide.
4. Reading: with the close-ups, the model lists every signal head and sign with its kind, literal text, symbol,
   facing, readability, what it applies to (the approach, lanes r#, another approach, pedestrians) and the image boxes
   that show it. Each answer is validated; a failed check goes back with the reason.
5. Another round (at most two in all) when the reading asks for new views or an important object stays unreadable:
   new panoramas near the existing cameras, then steps 3 and 4 again.

Writes into <read>/signals/ (or --output): signals.json (the readings, one entry per approach), summary.json (views,
requests, stop reasons), validation.json (structural checks), the profile and sampling plan, views.json (every view
with its pose, date and image), the lane scenes and sheets sent to the model, vlm/ (every prompt and answer), yolo/
(detections) and cache/ (every Street View response used, so the run replays with --cache-only). lanes.json and
movements.json are not changed. A lamp's colour in a photo says nothing about phases or timing, and a sign is bound
to lanes only where the model gives its reason.
"""
import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pave_read.fusion import yolo  # noqa: E402
from pave_read.fusion.control_acquisition import (AcquisitionUnavailable, StreetViewClient, acquire_context,  # noqa: E402
                                                  acquire_initial, alternative_requests)
from pave_read.fusion.control_prompts import final_prompt, inventory_prompt  # noqa: E402
from pave_read.fusion.control_sampling import from_network, initial_requests, prepare_profile, safe_id  # noqa: E402
from pave_read.fusion.controls import detail_pose, needs_resampling, reading_signature, validate_controls, validate_inventory  # noqa: E402
from pave_read.harness import figure  # noqa: E402
from pave_read.hybrid.common import digest, file_hash, read_json, require, resource_path, write_json  # noqa: E402
from pave_read.hybrid.evidence_visuals import sheet  # noqa: E402
from pave_read.hybrid.geometry import font  # noqa: E402
from pave_read.hybrid.inference import StageClient  # noqa: E402


def rel(path):
    """A path as stored in the run's files: relative to read/ when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def full(path):
    return resource_path(ROOT, path)


def up_angle(ux, uy):
    """Counter-clockwise turn, in degrees, that points the image direction (ux, uy) (y down) up."""
    return 90 - math.degrees(math.atan2(-uy, ux))


def lane_scene(image, section, path):
    """The approach's traced lanes on the satellite, rotated so that travel is up, motor lanes labelled r#."""
    motor = [r for r in section["regions"] if r.get("region_id") and r.get("existence") == "supported" and len(r["polygon_px"]) >= 3]
    if not motor:
        return None
    ux, uy = (sum(figure.axis(r["polygon_px"])[k] for r in motor) for k in (0, 1))
    phi = up_angle(ux, uy)
    points = [p for r in section["regions"] for p in r["polygon_px"]]
    cx, cy = (min(p[0] for p in points) + max(p[0] for p in points)) / 2, (min(p[1] for p in points) + max(p[1] for p in points)) / 2
    half = max(math.hypot(p[0] - cx, p[1] - cy) for p in points) + 60
    size = int(2 * half)
    rotated = image.crop((int(cx - half), int(cy - half), int(cx - half) + size, int(cy - half) + size)).rotate(phi, resample=Image.Resampling.BICUBIC)
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    turn = lambda p: ((p[0] - cx) * c + (p[1] - cy) * s + size / 2, -(p[0] - cx) * s + (p[1] - cy) * c + size / 2)
    d = ImageDraw.Draw(rotated)
    drawn = []
    for r in section["regions"]:
        polygon = [turn(p) for p in r["polygon_px"]]
        drawn.extend(polygon)
        d.line(polygon + [polygon[0]], fill=figure.MOTOR if r in motor else figure.OTHER, width=3)
    for r in motor:
        polygon = [turn(p) for p in r["polygon_px"]]
        x, y = sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon)
        d.text((x, y), r["region_id"], font=font(28), fill="white", anchor="mm", stroke_width=4, stroke_fill="black")
    box = (max(0, int(min(p[0] for p in drawn)) - 40), max(0, int(min(p[1] for p in drawn)) - 40),
           min(size, int(max(p[0] for p in drawn)) + 40), min(size, int(max(p[1] for p in drawn)) + 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    rotated.crop(box).save(path, quality=92)
    return path


def build_profile(site, lanes, policy, read_dir):
    """The sampling profile: the node's incoming links from the network, each labelled with its traced approach."""
    labels = {str(s["link_id"]): {"label": s["direction"], "section_id": s["section_id"],
                                  "region_ids": [r["region_id"] for r in s["regions"] if r.get("region_id") and r.get("existence") == "supported"]}
              for s in lanes if s["kind"] == "inbound_stopbar"}
    network = site["network"]
    profile = from_network(full(network["node_csv"]), full(network["link_csv"]), site["node_id"], policy, labels)
    profile["geometry_provenance"].update(node_csv=network["node_csv"], link_csv=network["link_csv"])
    profile["name"] = site.get("name") or profile["name"]
    profile["models"] = {"vlm": {"preset": "default", "max_tokens": 6144, "temperature": 0.0}, "yolo": dict(yolo.DEFAULTS)}
    profile["lanes"] = {"file": rel(read_dir / "lanes.json"), "sha256": file_hash(read_dir / "lanes.json")}
    return profile


def entries_for(views, scene_id=None, scene_path=None):
    entries = []
    if scene_id:
        entries.append((scene_id, "Traced lanes of this approach (r#), travel up; outlines from the lane trace", Image.open(scene_path).convert("RGB")))
    for v in views:
        path = full(v["path"])
        require(file_hash(path) == v["image_sha256"], "View hash mismatch")
        entries.append((v["id"], f"{v['view_role']} / {v['capture_date']} / {v['compass_heading_deg']:.1f} deg / station {v['signed_station_m']:.1f} m",
                        Image.open(path).convert("RGB")))
    return entries


def source_meta(views):
    keys = ["pano_id", "actual_lonlat", "capture_date", "capture_relation", "signed_station_m", "compass_heading_deg", "hfov_deg",
            "pitch_deg", "reference_kind", "parent_view_id"]
    return {v["id"]: {k: v.get(k) for k in keys} for v in views}


def save_sheet(entries, path):
    canvas = sheet(entries, path, cell=(900, 800), columns=3)
    canvas.save(path, quality=95, subsampling=0)
    return path


def stored(view):
    return {**view, "path": rel(view["path"])}


def audit(profile, manifest, output, client, vlm, scenes):
    """Steps 2 to 5 for every approach. Returns (readings, history, every view, acquisition attempts)."""
    policy, models = profile["policy"], profile["models"]
    allviews = [stored(v) for v in manifest["views"]]
    detections = yolo.detect(allviews, models["yolo"], {v["id"]: full(v["path"]) for v in allviews}, output) if allviews else {}
    readings, history, attempts = [], [], []
    for approach in profile["approaches"]:
        aid = approach["id"]
        folder = output / "approaches" / safe_id(aid)
        views = [v for v in allviews if v["approach_id"] == aid]
        scene_path = scenes.get(approach["section_id"])
        scene_id = "scene_" + approach["section_id"] if scene_path else None
        if not views:
            readings.append({"approach_id": aid, "direction": approach["label"], "section_id": approach["section_id"], "controls": [],
                             "unresolved": ["No usable Street View camera was returned"], "sampling_status": "no_usable_imagery",
                             "human_verified": False, "movement_constraints_applied": False})
            continue
        prior, result, stop, rounds = None, None, "round_limit", []
        for k in range(policy["max_rounds"]):
            rf = folder / f"round_{k + 1}"
            rf.mkdir(parents=True, exist_ok=True)
            if vlm.calls >= policy["max_vlm_calls"] - 1:
                stop = "vlm_budget_exhausted"
                break
            contexts = [v for v in views if v["view_role"] == "control_context"]
            meta = source_meta(views)
            proposals = {v["id"]: [{"class": d["class"], "confidence": d["confidence"],
                                    "bbox_xyxy": [c / v["image_size"][j % 2] for j, c in enumerate(d["bbox_xyxy"])]}
                                   for d in detections[v["id"]]["detections"]] for v in contexts}
            inventory = vlm.run(f"inventory_{safe_id(aid)}_r{k + 1}", save_sheet(entries_for(views, scene_id, scene_path), rf / "inventory.jpg"),
                                inventory_prompt(approach, meta, proposals, policy["max_details_per_round"]),
                                lambda x: validate_inventory(x, set(meta)))
            require(len(inventory["detail_requests"]) <= policy["max_details_per_round"], "Inventory exceeded policy detail limit")
            write_json(rf / "inventory.json", inventory)
            lookup, notes = {v["id"]: v for v in views}, []
            for r in inventory["detail_requests"]:
                parent = lookup[r["source_id"]]
                pose = detail_pose(parent, r["bbox_xyxy"], policy["detail_min_fov_deg"], policy["detail_max_fov_deg"])
                sid = "gsv_detail_" + safe_id(aid) + "_" + digest([parent["pano_id"], pose])[:12]
                if any(v["id"] == sid for v in views):
                    notes.append({"source": r["source_id"], "status": "duplicate_pose"})
                    continue
                view = {**parent, **pose, "id": sid, "view_role": "control_detail", "camera_role": "control_detail", "sampling_position": "detail",
                        "parent_view_id": parent["id"], "requested_bbox": r["bbox_xyxy"], "detail_question": r["question"], "adaptive_round": k + 1}
                try:
                    view = stored(client.image(view))
                except AcquisitionUnavailable as e:
                    notes.append({"source": r["source_id"], "status": str(e)})
                    continue
                views.append(view)
                allviews.append(view)
            meta = source_meta(views)
            result = vlm.run(f"final_{safe_id(aid)}_r{k + 1}", save_sheet(entries_for(views, scene_id, scene_path), rf / "final.jpg"),
                             final_prompt(approach, meta, scene_id, policy["preferred_date_window"]),
                             lambda x: validate_controls(x, approach["region_ids"], set(meta) | ({scene_id} if scene_id else set())))
            for r in result.get("resampling_requests", []):
                require(r["source_id"] in meta, "Resampling requires a Street View image")
            result.update(approach_id=aid, direction=approach["label"], section_id=approach["section_id"])
            seen = reading_signature(result)
            rounds.append({"round": k + 1, "views": len(views), "reading_count": len(seen), "new_readings": len(seen - (prior or set())),
                           "needs_resampling": needs_resampling(result), "detail_notes": notes})
            write_json(rf / "reading.json", result)
            if not needs_resampling(result):
                stop = "reading_sufficient"
                break
            if prior is not None and not seen - prior:
                stop = "no_new_readings"
                break
            prior = seen
            if k + 1 >= policy["max_rounds"]:
                stop = "round_limit"
                break
            slots = min(policy["max_adaptive_positions_per_round"], policy["max_context_views_per_approach"] - len(contexts))
            if slots <= 0:
                stop = "context_view_budget"
                break
            added, deferred = [], []
            for request in alternative_requests(profile, approach, contexts, k + 2):
                view, attempt = acquire_context(profile, approach, request, client, contexts + added, require_preferred_date=True)
                attempts.append(attempt)
                if attempt["status"] == "outside_preferred_date_window":
                    deferred.append(attempt)
                if view:
                    added.append(stored(view))
                if len(added) >= slots or attempt["status"] in ("metadata_budget_exhausted", "image_budget_exhausted"):
                    break
            if not added and policy["date_fallback"]:
                for old in sorted(deferred, key=lambda r: r["assessment"]["pose_score"]):
                    view, attempt = acquire_context(profile, approach, old["request"], client, contexts)
                    attempt["date_fallback"] = True
                    attempts.append(attempt)
                    if view:
                        added.append({**stored(view), "date_fallback_reason": "No distinct preferred-window panorama in bounded adaptive search"})
                        break
            if not added:
                stop = "no_distinct_usable_panorama"
                break
            views.extend(added)
            allviews.extend(added)
            detections.update(yolo.detect(added, models["yolo"], {v["id"]: full(v["path"]) for v in added}, output))
        if result is None:
            result = {"approach_id": aid, "direction": approach["label"], "section_id": approach["section_id"], "controls": [],
                      "unresolved": ["Inference budget exhausted before a result"], "movement_constraints_applied": False, "human_verified": False}
        result["sampling_status"] = stop
        readings.append(result)
        history.append({"approach_id": aid, "rounds": rounds, "stop_reason": stop})
    return readings, history, allviews, attempts


def validate(profile, views, readings, history, receipt):
    """Structural and provenance checks of the run, not an accuracy score."""
    policy, checks = profile["policy"], {}
    approaches = {a["id"]: a for a in profile["approaches"]}
    byid = {v["id"]: v for v in views}

    def check(name, ok):
        checks[name] = bool(ok)
        require(ok, "Run check failed: " + name)
    check("one_result_per_approach", {s["approach_id"] for s in readings} == set(approaches) and len(readings) == len(approaches))
    check("unique_view_ids", len(byid) == len(views))
    check("image_hashes", all(file_hash(full(v["path"])) == v["image_sha256"] for v in views))
    contexts = lambda aid: [v for v in views if v["approach_id"] == aid and v["view_role"] == "control_context"]
    check("distinct_context_panoramas", all(len({v["pano_id"] for v in contexts(aid)}) == len(contexts(aid)) for aid in approaches))
    details = [v for v in views if v["view_role"] == "control_detail"]
    check("detail_parents_same_panorama", all(v.get("parent_view_id") in byid and byid[v["parent_view_id"]]["pano_id"] == v["pano_id"] for v in details))
    check("bounded_detail_fov", all(policy["detail_min_fov_deg"] - .01 <= v["hfov_deg"] <= policy["detail_max_fov_deg"] + .01 for v in details))
    for s in readings:
        a = approaches[s["approach_id"]]
        if s.get("controls"):
            validate_controls(copy.deepcopy(s), a["region_ids"], set(byid) | {"scene_" + a["section_id"]})
    check("no_movement_constraints", all(s["movement_constraints_applied"] is False for s in readings))
    check("bounded_rounds", all(len(h["rounds"]) <= policy["max_rounds"] for h in history))
    check("bounded_context_views", all(len(contexts(aid)) <= policy["max_context_views_per_approach"] for aid in approaches))
    check("bounded_requests", receipt["model_requests"] <= policy["max_vlm_calls"] and receipt["street_view"]["metadata_requests"] <= policy["max_metadata_requests"]
          and receipt["street_view"]["image_requests"] <= policy["max_image_requests"])
    return {"status": "passed", "checks": checks, "scope": "Structural and provenance checks only; not an accuracy score or proof that every sign is visible."}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagery", type=Path, required=True, help="the imagery folder run_read.py read (site.json)")
    p.add_argument("--read", type=Path, required=True, help="the run_read.py output folder (lanes.json)")
    p.add_argument("--output", type=Path, help="default: <read>/signals")
    p.add_argument("--policy", type=Path, help="JSON overrides of the sampling policy (date window, pinned panoramas, budgets)")
    p.add_argument("--seed-cache", type=Path, action="append", default=[],
                   help="read-only Street View cache to take responses from before asking the service (repeatable)")
    p.add_argument("--plan-only", action="store_true", help="write the profile and camera plan; no request to any service")
    p.add_argument("--cache-only", action="store_true", help="no Street View or model request: replay the run's own cache")
    p.add_argument("--reuse-cache-from", type=Path, action="append", default=[], help="reuse model answers of an earlier run to identical requests")
    a = p.parse_args()
    imagery, read_dir = a.imagery.resolve(), a.read.resolve()
    out = (a.output or read_dir / "signals").resolve()
    out.mkdir(parents=True, exist_ok=True)
    site, lanes = read_json(imagery / "site.json"), read_json(read_dir / "lanes.json")
    profile = prepare_profile(build_profile(site, lanes, read_json(a.policy) if a.policy else {}, read_dir))
    signature = digest(profile)
    if (out / "input.json").exists():
        require(read_json(out / "input.json")["signature"] == signature, "The profile changed: choose a new output folder")
    write_json(out / "input.json", {"signature": signature, "imagery": rel(imagery), "read": rel(read_dir), "policy": rel(a.policy) if a.policy else None})
    write_json(out / "profile.json", profile)
    write_json(out / "sampling_plan.json", {"requests": [r for app in profile["approaches"] for r in initial_requests(profile, app)]})
    for app in profile["approaches"]:
        print(app["id"], app["label"], round(app["heading_deg"], 1), app["reference_kind"], round(app["reference_station_m"], 1), flush=True)
    if a.plan_only:
        print({"status": "planned", "approaches": len(profile["approaches"]), "output": rel(out)})
        return
    write_json(out / "status.json", {"state": "running"})
    policy = profile["policy"]
    client = StreetViewClient(ROOT, out / "cache", policy, cache_only=a.cache_only, seed=a.seed_cache)
    satellite = Image.open(full(site["satellite"]["path"])).convert("RGB")
    scenes = {}
    for s in lanes:
        if s["kind"] == "inbound_stopbar":
            path = lane_scene(satellite, s, out / "scenes" / f"{s['section_id']}.jpg")
            if path:
                scenes[s["section_id"]] = path
    manifest = acquire_initial(profile, client)
    initial = dict(client.counts)
    vlm = StageClient(ROOT, out, {**profile["models"]["vlm"], "max_api_calls": policy["max_vlm_calls"]},
                      cache_only=a.cache_only, reuse_cache_from=a.reuse_cache_from)
    try:
        readings, history, views, attempts = audit(profile, manifest, out, client, vlm, scenes)
    except Exception as e:
        write_json(out / "status.json", {"state": "failed", "error_type": type(e).__name__, "error": str(e)})
        raise
    receipt = {"model_requests": vlm.calls, "cache_hits": vlm.hits, "street_view": client.counts, "street_view_initial": initial}
    write_json(out / "views.json", {"views": views, "positions": manifest["positions"], "coverage": manifest["coverage"],
                                    "adaptive_attempts": attempts, "coordinate_note": manifest["coordinate_note"]})
    write_json(out / "signals.json", readings)
    write_json(out / "history.json", history)
    summary = {"site": profile["name"], "node_id": profile["site_id"], "model": f"{vlm.provider}/{vlm.model}", "approaches": len(readings),
               "controls": sum(len(s["controls"]) for s in readings),
               "context_views": sum(v["view_role"] == "control_context" for v in views),
               "detail_views": sum(v["view_role"] == "control_detail" for v in views), **receipt,
               "stop_reasons": {s["direction"]: s["sampling_status"] for s in readings}, "movement_constraints_applied": False}
    if not (out / "first_run.json").exists():
        write_json(out / "first_run.json", {**receipt, "finished_utc": datetime.now(timezone.utc).isoformat()})
    write_json(out / "summary.json", summary)
    write_json(out / "validation.json", validate(profile, views, readings, history, read_json(out / "first_run.json")))
    write_json(out / "status.json", {"state": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({k: summary[k] for k in ("site", "controls", "context_views", "detail_views", "model_requests", "street_view")}))


if __name__ == "__main__":
    main()
