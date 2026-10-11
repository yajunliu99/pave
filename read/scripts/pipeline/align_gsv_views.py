"""Align the projected satellite lanes with a junction's street views; nothing here is site-specific.

Road arms come from the traced regions, each camera is placed on an arm by its position, and views of
one arm captured on one date are treated as one drive that shares its position error.
1. For every view the model reads where the real line behind each projected boundary crosses a few
   image rows, and the view's pose (sideways offset, heading, camera height) is fitted.
2. The drive's views are fitted together: one offset and height, a heading per view.
3. Readings can lock onto lines a whole lane or two away. The model is shown the overlay shifted by
   -2..+2 lane widths and picks the panel whose strips sit on matching lanes; the drive's views vote.
   After a shift, every read line position is given to the nearest projected boundary again and the
   drive's pose is refitted, a few rounds, so the correction carries to views that could not choose.
4. The model checks the final overlay strip by strip. Only a consistent check counts as aligned.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop.gsv_projection import (HEIGHT, align_prompt, arm_of, boundaries, check_prompt, choose_rows, dense,
                                                    draw_regions, facing, fit_group, fit_pose, lane_width, left_to_right, leg_regions,
                                                    lane_offset, load_run, offset_prompt, offset_sheet, realign_group, render_for_model, road_arms,
                                                    validate_check, validate_offset, validate_readings, view_image, visible_ids,
                                                    visible_strips)
from movement_fixer.hybrid.common import ValidationError, write_json
from movement_fixer.hybrid.evidence_visuals import transport_jpeg
from movement_fixer.hybrid.inference import StageClient

EXTEND_M = 10.


def prepare(atlas, data, lanes, arms, view, folder):
    arm = arm_of(view, arms)
    regions = leg_regions(data, lanes, view, arms)
    edges, sides = boundaries([r for r, _, _ in regions])
    base = (0., 0., HEIGHT)
    # Lane lines continue past the traced section; extending them gives the fit the near ground too.
    samples, sides = left_to_right([dense(e, extend=EXTEND_M) for e in edges], sides, view, base)
    entry = {"view": view, "arm": arm["name"], "facing": facing(view, arm), "image": view_image(atlas, view), "regions": regions,
             "samples": samples, "sides": sides, "labels": [(" ".join(lines), color) for _, lines, color in regions],
             "step": lane_width(regions), "folder": folder}
    try:
        entry["rows"] = choose_rows(samples, view, base)
    except ValidationError as error:
        entry.update(rows=None, unreadable=str(error))  # too little of the traced lanes in view to read; no model call
    return entry


def ask(client, e, name, pose, prompt_for, validator_for):
    ids = visible_ids(e["samples"], e["view"], e["rows"], pose)
    canvas = render_for_model(e["image"], e["view"], e["samples"], e["sides"], e["labels"], e["rows"], pose)
    canvas.save(e["folder"] / f"{name}.png")
    width = e["view"]["image_size"][0]
    return client.run(f"{name}_{e['view']['id']}", transport_jpeg(canvas, e["folder"] / f"{name}.png"), prompt_for(ids),
                      lambda v: validator_for(v, ids, width))


def save_pair(e, pose, note):
    before, _ = draw_regions(e["view"], e["image"], e["regions"], (0., 0., HEIGHT), "Before: recorded camera pose, height 2.5 m assumed.")
    after, _ = draw_regions(e["view"], e["image"], e["regions"], pose, note)
    pair = Image.new("RGB", (before.width * 2, before.height))
    pair.paste(before, (0, 0))
    pair.paste(after, (before.width, 0))
    pair.save(e["folder"] / "before_after.jpg", quality=88)
    after.save(e["folder"] / "aligned.jpg", quality=90)


def as_pose(p):
    return {"lateral_shift_m": round(p[0], 3), "heading_shift_deg": round(p[1], 3), "camera_height_m": round(p[2], 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="evidence run supplying the lane labels")
    parser.add_argument("--views", nargs="*", help="view IDs; default every non-signal street view in the atlas")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-api-calls", type=int, default=30)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--reuse-cache-from", type=Path, action="append", default=[],
                        help="Reuse content-identical model stages from an earlier alignment run")
    args = parser.parse_args()
    atlas = args.atlas.resolve()
    output = (args.output or atlas.parent / "gsv_alignment").resolve()
    data, lanes = load_run(atlas, args.run.resolve())
    arms = road_arms(data["regions"])
    client = StageClient(ROOT, output, {"preset": "default", "max_tokens": 6144, "temperature": 0., "max_api_calls": args.max_api_calls},
                         cache_only=args.cache_only, reuse_cache_from=[p.resolve() for p in args.reuse_cache_from])
    views = [v for v in data["views"] if (v["id"] in args.views if args.views else v.get("sampling_domain") != "traffic_control")]
    groups = defaultdict(list)
    for view in views:
        groups[(arm_of(view, arms)["name"], view["capture_date"])].append(view)
    records, group_fits = {}, {}
    for key, members in groups.items():
        entries, unread = [], []
        for view in members:
            folder = output / view["id"]
            folder.mkdir(parents=True, exist_ok=True)
            record = records[view["id"]] = {"view_id": view["id"], "arm": key[0], "capture_date": key[1]}
            e = prepare(atlas, data, lanes, arms, view, folder)
            record["facing"] = e["facing"]
            if e["rows"] is None:
                record.update(status="unreadable", reason=e["unreadable"])
                unread.append(e)
                continue
            try:
                first = ask(client, e, "align", (0., 0., HEIGHT), lambda ids, e=e: align_prompt(e["view"], ids, e["rows"]),
                            lambda v, ids, w, e=e: validate_readings(v, ids, e["rows"], w))
                e["raw_readings"] = {k: v for k, v in first["readings"].items() if any(x is not None for x in v)}
                e["readings"] = dict(e["raw_readings"])
                e["fit"] = fit_pose(e["samples"], view, e["rows"], e["readings"])
                e["pose"] = tuple(e["fit"]["pose"].values())
                record.update(readings=first["readings"], features=first["features"], own_fit=e["fit"],
                              layout_mismatches=list(first["layout_mismatches"]))
                entries.append(e)
            except ValidationError as error:
                record.update(status="alignment_failed", reason=str(error))
        group = fit_group(entries) if len(entries) > 1 else {"inliers": [e["view"]["id"] for e in entries], "outliers": [], "shared": None}
        for e in entries:
            if group["shared"]:
                heading = group["views"][e["view"]["id"]]["heading_shift_deg"] if e["view"]["id"] in group["views"] else 0.
                e["pose"] = (e["facing"] * group["shared"]["leg_shift_m"], heading, group["shared"]["camera_height_m"])
            records[e["view"]["id"]]["pose_source"] = "group" if group["shared"] else "own"
        votes = {}
        for e in entries:
            sheet, mapping = offset_sheet(e["view"], e["image"], e["regions"], e["pose"], e["step"])
            sheet.save(e["folder"] / "offset.png")
            try:
                choice = client.run(f"offset_{e['view']['id']}", transport_jpeg(sheet, e["folder"] / "offset.png"), offset_prompt(e["view"]),
                                    validate_offset)
            except ValidationError as error:
                choice = None
                records[e["view"]["id"]]["offset_error"] = str(error)
            records[e["view"]["id"]]["offset_choice"] = choice
            if choice and choice["panel"] in mapping:
                votes[e["view"]["id"]] = (int(e["facing"] * mapping[choice["panel"]]), choice["confidence"])
        k, source = lane_offset(votes)
        if k:
            regrouped = realign_group(entries, k)
            group = regrouped or group
        group_fits["/".join(key)] = {**group, "lane_offset_votes": votes, "lane_offset": k, "lane_offset_source": source}
        for e in entries:
            view, record = e["view"], records[e["view"]["id"]]
            record.update(lane_offset=k, offset_source=source)
            try:
                strips = visible_strips(e["samples"], e["sides"], e["labels"], view, e["rows"], e["pose"])
                check = ask(client, e, "check", e["pose"], lambda ids, e=e, strips=strips: check_prompt(e["view"], ids, e["rows"], strips),
                            lambda v, ids, w, e=e, strips=strips: validate_check(v, ids, e["rows"], w, strips))
            except ValidationError as error:
                check = None
                record["check_error"] = str(error)
            verdicts = [c["verdict"] for c in (check or {}).get("strip_checks", {}).values()]
            if check:
                record.update(strip_checks=check["strip_checks"], layout_mismatches=record["layout_mismatches"] + check["layout_mismatches"])
            settled = check is not None and "consistent" in verdicts and "inconsistent" not in verdicts
            shared = record["pose_source"] == "group"
            if not settled:
                status = "alignment_doubtful"
            elif source == "single_vote":
                status = "aligned_single_vote"
            else:
                status = "aligned" if shared else "aligned_single_view"
            record.update(status=status, pose=as_pose(e["pose"]))
            save_pair(e, e["pose"], f"After (lane offset {k or 0:+d}, {source or 'no shift'}): shift {e['pose'][0]:+.2f} m, heading "
                                    f"{e['pose'][1]:+.1f} deg, height {e['pose'][2]:.2f} m; strip check {'/'.join(sorted(set(verdicts))) or 'none'}; {status}.")
        for e in unread:
            # Too far from the traced lanes to read, but the drive's offset still applies to this camera.
            if group.get("shared"):
                pose = (e["facing"] * group["shared"]["leg_shift_m"], 0., group["shared"]["camera_height_m"])
                records[e["view"]["id"]].update(status="group_pose_unchecked", pose_source="group_unread", pose=as_pose(pose))
                save_pair(e, pose, "Offset and height from the same drive; not read or checked in this view.")
        for view in members:
            write_json(output / view["id"] / "alignment.json", records[view["id"]])
    summary = {vid: {k: r.get(k) for k in ("arm", "capture_date", "status", "pose_source", "lane_offset", "offset_source", "pose",
                                           "strip_checks", "layout_mismatches", "reason")} for vid, r in records.items()}
    for vid, r in summary.items():
        print(vid, r["arm"], r["capture_date"], r["status"], "lane offset", r["lane_offset"], "| checks:",
              {k: v["verdict"] for k, v in (r["strip_checks"] or {}).items()}, flush=True)
    write_json(output / "summary.json", {"arms": arms, "views": summary, "groups": group_fits, "vlm_calls_this_run": client.calls,
                                         "vlm_cache_hits_this_run": client.hits, "model": f"{client.provider}/{client.model}",
                                         "scope": "Pose fitted to model readings of painted features; not a surveyed camera calibration."})
    print({"calls": client.calls, "hits": client.hits, "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
