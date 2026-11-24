"""Candidate-region generation with a review loop: prompts, validators, site runner.

Each review round shows the model something the draft did not have: its own
boundaries drawn back on the pixels, every region's pixels in isolation, and
the findings of the coordinate checks. No lane count, network attribute or
reference geometry is ever part of a prompt.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from ..hybrid.common import ValidationError, digest, file_hash, parallel_map, read_json, request_workers, require, resource_path, write_json
from ..hybrid.evidence_visuals import transport_jpeg
from ..hybrid.geometry import font
from . import anchor, render, street_panel
from .checks import check_strips
from .controller import run_review_loop
from .frames import DEFAULTS as FRAME_DEFAULTS, frames_from_network
from .strips import EDIT_OPS, MIN_GAP, apply_edits, change_px, edge_meta, is_number, section_regions, signature, validate_strips

VERSION = "autoloop-geometry-6"
DEFAULTS = {"max_rounds": 3, "converge_px": 2.0, "max_vlm_calls": 48, "display_scale": 2,
            "max_widen": 1, "widen_m": 6.0, "min_gap_px": MIN_GAP, "anchor": True, "street_view": True}
KINDS = ("centre_or_median_edge", "lane_line", "bicycle_line", "buffer_or_rail_edge", "kerb_or_pavement_edge", "unknown")
VISIBILITY = ("clear", "partial", "occluded")
WHERE = {"inbound_stopbar": "The junction lies beyond the TOP edge, so the stop bar is near the top.",
         "outbound_receiving": "The junction lies beyond the BOTTOM edge; vehicles here have just left it."}


def _names(spec):
    return [f"S{i + 1}" for i in range(len(spec["stations"]))]


def _shaded(spec):
    return (" Shadow covers all or part of the crop, so the lightness of the shaded parts has been contrast-stretched to show"
            " shaded paint; there brightness and colour are not true, and the stretch can exaggerate texture.") if spec.get("contrast_stretched") else ""


def _lifted(spec, review=False):
    if not spec.get("shade_panel"):
        return ""
    if review:
        return (" (3) the same boundaries on a copy of the crop whose shaded parts have been lightened (colours there are not true,"
                " and dark cars or new asphalt are lightened too); use it to see paint in shadow.")
    return (" A second panel shows the same crop with its shaded parts lightened, with the same rulers and stations; colours there"
            " are not true, and dark cars or new asphalt are lightened too. Use it to see paint in shadow; trace on either panel.")


def _street(spec, review=False):
    view = spec.get("street_view_used")
    if not view:
        return ""
    where = view["camera_note"]
    if review:
        return (f" (S) the street view taken from camera C ({where}), looking toward the junction, with the current boundaries e# projected onto"
                " the road from the camera's recorded position and heading (camera height 2.5 m assumed). The projection can sit up to about a lane"
                " to one side or skew with distance; use it to tell which painted line each boundary stands for. A lane in the street view with"
                " no strip of its own between two projected boundaries means a missing boundary; a projected strip over kerb, sidewalk or"
                " opposing lanes means a spurious one.")
    return (f" Panel S is a street view from camera C, marked in yellow on the crop with its field of view ({where}), looking toward the"
            f" junction{', captured ' + view['date'] if view.get('date') else ''}. Its position comes from the street-view metadata and can be off"
            " by a few metres. Use it to see lane lines, arrows and kerbs that shadow, vehicles or blur hide on the satellite crop: every lane"
            " of the target direction it shows at the camera's cross-section should lie between two of your boundaries.")


def _resampled(spec):
    return "" if spec["frame"].pixels_preserved() else " (the rotation resampled the pixels slightly)"


def draft_prompt(spec):
    names = _names(spec)
    return f"""Trace candidate surface boundaries for section {spec['id']} ({spec['kind']}) from what is visible.
The image is a satellite crop, rotated so that traffic on the target carriageway moves UP.{_resampled(spec)}{_shaded(spec)}{_lifted(spec)}{_street(spec)} {WHERE[spec['kind']]} Traffic keeps to the {spec.get('driving_side', 'right')}: the opposing carriageway, if it appears, is beside the target carriageway on the driver's left and must be left out.
The only things drawn on the pixels are dashed magenta station lines {names[0]}..{names[-1]}, top to bottom. Rulers along the top and bottom give x coordinates in crop units, valid on every station line.
List every longitudinal boundary that separates surface strips across the target carriageway, from its left limit (centre line, median edge or left kerb) to its right limit (kerb or pavement edge): lane lines, bicycle-lane lines, buffer edges, rail-zone edges and kerbs. The right limit is the kerb itself: near the stop bar a right-turn lane often lies between a bicycle lane and the kerb, and it belongs to the carriageway. Give each boundary's x coordinate on EVERY station line. Where vehicles, shadow or worn paint hide it, interpolate between its visible parts and name those stations in interpolated_stations.
The strips between neighbouring boundaries become neutral candidate regions for a later classification step. Do not classify them or decide which are motor-vehicle lanes. No expected number of strips is given; do not add or drop boundaries to make widths regular. Trace only strips present across the whole station range; describe a strip that starts or ends inside the crop in partial_strips instead.
Return JSON {{"boundaries":[{{"xs":[{len(names)} numbers, one per station],"kind":"{'|'.join(KINDS)}","visibility":"{'|'.join(VISIBILITY)}","interpolated_stations":["S#"],"evidence":"what is visible along this boundary"}}],"partial_strips":["description"],"cannot_determine":["uncertainty"]}}.
"""


PARTIAL_HINT = ("Each boundary needs one x per station. A line seen at only some stations is still a boundary: give null "
                "at stations where it cannot be placed (at least two stations with numbers); those stations are filled "
                "from the nearest placed ones and marked interpolated. A region needs a boundary on each side.")


def fill_partial(xs):
    """Stations given as null take the line through the nearest placed stations (constant beyond the ends)."""
    known = [(i, float(x)) for i, x in enumerate(xs) if is_number(x)]
    filled = []
    for i, x in enumerate(xs):
        if is_number(x):
            filled.append(float(x))
            continue
        left = [k for k in known if k[0] < i]
        right = [k for k in known if k[0] > i]
        if left and right:
            (i0, x0), (i1, x1) = left[-1], right[0]
            filled.append(round(x0 + (x1 - x0) * (i - i0) / (i1 - i0), 1))
        else:
            filled.append((left[-1] if left else right[0])[1])
    return filled


def validate_draft(value, spec, min_gap=MIN_GAP):
    require(isinstance(value, dict) and isinstance(value.get("boundaries"), list), "Draft requires boundaries[]")
    names, rows = set(_names(spec)), []
    order = _names(spec)
    for b in value["boundaries"]:
        require(isinstance(b, dict) and isinstance(b.get("xs"), list) and len(b["xs"]) == len(names)
                and all(x is None or is_number(x) for x in b["xs"]) and sum(is_number(x) for x in b["xs"]) >= 2, PARTIAL_HINT)
        gaps = [order[i] for i, x in enumerate(b["xs"]) if x is None]
        if gaps:
            b = {**b, "xs": fill_partial(b["xs"]), "interpolated_stations": sorted(set(b.get("interpolated_stations", [])) | set(gaps)),
                 "evidence": b.get("evidence", "") + f" [Not placed at {', '.join(gaps)}; filled from the nearest placed stations.]"}
        require(b.get("kind") in KINDS, "Unknown boundary kind")
        require(b.get("visibility") in VISIBILITY, "Invalid boundary visibility")
        marks = b.get("interpolated_stations", [])
        require(isinstance(marks, list) and set(marks) <= names, "interpolated_stations must name stations")
        require(isinstance(b.get("evidence"), str), "Boundary evidence text missing")
        rows.append(b)
    require(len(rows) >= 2, "At least two boundaries are needed to form a region. " + PARTIAL_HINT)
    rows.sort(key=lambda b: sum(b["xs"]))
    strips = {"stations": list(spec["stations"]), "edges": [[float(x) for x in b["xs"]] for b in rows],
              "meta": [{"kind": b["kind"], "visibility": b["visibility"], "evidence": b["evidence"], "origin": "draft",
                        "interpolated_stations": sorted(b.get("interpolated_stations", []))} for b in rows]}
    validate_strips(strips, spec["frame"].width, min_gap)
    return {"strips": strips, "partial_strips": value.get("partial_strips", []),
            "cannot_determine": value.get("cannot_determine", [])}


def review_prompt(spec, strips, issues, number, rounds):
    names = _names(spec)
    table = "\n".join(f"  e{i + 1} [{m.get('kind', 'unknown')}, {m.get('origin', 'draft')}]: " + ", ".join(f"{x:.0f}" for x in xs)
                      for i, (xs, m) in enumerate(zip(strips["edges"], edge_meta(strips))))
    checks = "\n".join(f"  - key {i['key']} | {i['target']} | {i['hint']} | {json.dumps(i['detail'])}" for i in issues) or "  none raised"
    earlier = "\n".join(f"  round {r['round']}: " + ("; ".join(e["effect"] for e in r.get("edits_applied", [])) or "no change")
                        for r in rounds) or "  none"
    count = len(strips["edges"])
    return f"""Review the candidate boundaries for section {spec['id']} ({spec['kind']}), round {number}.
Panels: (1) the satellite crop, travel UP,{_resampled(spec)}{_shaded(spec)} with the current boundaries e1..e{count} as dashed cyan lines (the pixels show through the gaps), neutral regions r1..r{count - 1} between them, dashed magenta station lines {names[0]}..{names[-1]} and x rulers in crop units. {WHERE[spec['kind']]} (2) each region's own pixels, unmarked and isolated at the same scale, driver-left to driver-right.{_lifted(spec, review=True)}{_street(spec, review=True)}
The boundaries came from a model and may be wrong: displaced from the real line, missing (one region holding two strips), spurious (a line through uniform pavement), or taking in the opposing carriageway or ground that is not road. Check the outermost strips too: pavement between the last boundary and the kerb (often a right-turn lane beyond a bicycle lane) is a missing region, to be added with add_region. Compare each with the pixels beside and beneath it.
Current boundaries, x on {names[0]}..{names[-1]}:
{table}
Geometry checks, computed from the coordinates alone; they do not know what the pixels show:
{checks}
Changes already made in earlier rounds, described by position because IDs are renumbered left to right every round (this round's e2 or r2 need not be what an earlier round called e2 or r2). Those changes are already in the geometry shown; do not repeat or undo one without new visible evidence:
{earlier}
Answer every check key once: "edit" and include the edit; "dismiss" when the geometry is right as drawn, saying what in the pixels shows it (a bicycle strip is legitimately narrow, a bay legitimately tapers); or "unresolved" when this image cannot settle it.
You may also correct boundaries no check mentioned. Change only what the pixels support. Shadow or vehicles hiding a boundary are not evidence that it is absent: where a boundary cannot be seen either way, answer "unresolved" and leave it rather than merging or dropping. Do not regularise widths and do not aim for a number of lanes; none is given. Edits use crop units, one x per station:
  {{"op":"move_edge","edge":"e2","xs":[...]}}
  {{"op":"split_region","region":"r3","xs":[...]}} for a boundary visible inside r3
  {{"op":"merge_regions","regions":["r3","r4"]}} when no real boundary separates them
  {{"op":"add_region","side":"left|right","xs":[...]}} when the carriageway continues beyond the outer boundary
  {{"op":"drop_region","region":"r1"}} when an outermost region is opposing carriageway or not road
IDs always refer to the geometry shown in this round, even when several edits are returned. Every edit needs "evidence".
Return JSON {{"issue_responses":[{{"issue":"key","decision":"edit|dismiss|unresolved","reason":"visible evidence"}}],"edits":[],"verdict":"accept|revise","notes":[]}}. Use verdict "accept" with no edits when the boundaries match the pixels.
"""


def validate_review(value, keys):
    require(isinstance(value, dict) and value.get("verdict") in ("accept", "revise"), "Review needs verdict accept or revise")
    edits, responses = value.get("edits"), value.get("issue_responses")
    require(isinstance(edits, list) and isinstance(responses, list), "Review needs edits[] and issue_responses[]")
    for edit in edits:
        require(isinstance(edit, dict) and edit.get("op") in EDIT_OPS, "Unknown edit operation")
        require(isinstance(edit.get("evidence"), str) and edit["evidence"].strip(), "Every edit must cite visible evidence")
    require(bool(edits) == (value["verdict"] == "revise"), "Verdict revise needs edits; verdict accept must have none")
    answered = [r.get("issue") if isinstance(r, dict) else None for r in responses]
    require(set(answered) == set(keys) and len(answered) == len(keys), "Answer each check key exactly once")
    for r in responses:
        require(r.get("decision") in ("edit", "dismiss", "unresolved"), "Invalid issue decision")
        require(isinstance(r.get("reason"), str) and r["reason"].strip(), "Each issue response needs a reason")
    require(edits or not any(r["decision"] == "edit" for r in responses), "An 'edit' decision needs an edit")
    return {"verdict": value["verdict"], "edits": edits, "issue_responses": responses, "notes": value.get("notes", [])}


def _sheet(views, path):
    """Panels side by side at their own size; returns the JPEG that is sent to the model."""
    gap, head = 14, 62
    canvas = Image.new("RGB", (sum(v[2].width for v in views) + gap * (len(views) + 1),
                               head + max(v[2].height for v in views) + gap), "#222831")
    d, x, manifest = ImageDraw.Draw(canvas), gap, []
    for key, title, image in views:
        canvas.paste(image, (x, head))
        d.text((x, 6), key, font=font(22), fill="white")
        d.text((x, 34), title, font=font(18), fill="#aebdcb")
        manifest.append({"image_id": key, "native_size": list(image.size),
                         "sheet_box_xyxy": [x, head, x + image.width, head + image.height]})
        x += image.width + gap
    canvas.save(path)
    write_json(Path(path).with_suffix(".panels.json"), manifest)
    return transport_jpeg(canvas, path)


def _inside(frame, size):
    return all(0 <= x <= size[0] and 0 <= y <= size[1] for x, y in frame.corners())


def run_section(image, spec, drafter, reviewer, output, policy, budget_left, street=None, start=None, extra_issues=()):
    """Draft one section's boundaries, then review them until a stop reason is reached.

    With `start` (earlier final strips) there is no draft: the review loop resumes from those strips, and
    `extra_issues` (raised by a later stage, e.g. a street-view lane count) are put to the first review round
    beside the geometry checks. Its files and model stages are kept apart under <section>/recheck."""
    p, sid, scale = policy, spec["id"], policy["display_scale"]
    folder = Path(output) / sid / ("recheck" if start else "")
    frame, stations, mpp = spec["frame"], spec["stations"], spec.get("ground_mpp")
    events, widened, latest = [], 0, {"round": 0}
    prefix = "recheck_" if start else ""

    def check(strips):
        issues = check_strips(strips, frame.width, p.get("checks"), mpp)
        return issues + list(extra_issues) if latest["round"] == 0 else issues

    def failed(reason, detail=None):
        return ({"id": sid, "source_id": "satellite", "kind": spec["kind"], "direction": spec["direction"],
                 "link_id": spec["link_id"], "heading_deg": frame.heading_deg, "arm": spec.get("arm"),
                 "regions": [], "geometry_source": "vlm_draft_review_loop",
                 "review": {"status": "unresolved", "stop_reason": reason, "rounds": 0, "open_issues": []}},
                {"section_id": sid, "frame": frame.serialize(), "stations": stations, "status": "unresolved",
                 "stop_reason": reason, "detail": detail, "events": events})

    while True:
        current = {**spec, "frame": frame}
        raw = render.crop(image, frame)
        stage = f"draft_{sid}" + (f"_w{widened}" if widened else "")
        d = folder / ("draft" + (f"_w{widened}" if widened else "")) if not start else folder
        d.mkdir(parents=True, exist_ok=True)
        raw.save(d / "raw.png")
        stretched, lifted = render.lit(raw), None
        if stretched is not None:
            current["contrast_stretched"] = True
            stretched.save(d / "contrast_stretched.png")
            raw = stretched  # what the model is shown, in the draft and in every review
        else:
            lifted = render.shade_lifted(raw)
            if lifted is not None:
                current["shade_panel"] = True
                lifted.save(d / "shade_lifted.png")
        ruled = render.ruled(raw, stations, scale)
        if street:
            at, relative, camera_px = street_panel.camera_on_crop(street["view"], frame, street["to_pixel"])
            behind = (at[1] - frame.height) * mpp if at[1] > frame.height else 0
            note = (f"{behind:.0f} m below the crop's bottom edge" if behind > 1 else
                    f"{(-at[1]) * mpp:.0f} m above its top edge" if at[1] < 0 else "inside the crop")
            if 0 <= at[1] <= frame.height:
                ruled = street_panel.mark_camera(ruled, at, relative, street["view"].get("view_settings", {}).get("fov", 90), scale)
            current["street_view_used"] = {"id": street["view"]["id"], "camera_note": f"camera {note}", "date": street["view"].get("capture_date")}
            street["camera_px"] = camera_px
        views = [("ruled", "Travel UP; station lines S#, x ruler in crop units" + ("; C = street-view camera" if street else ""), ruled)]
        if lifted is not None:
            views.append(("lightened", "Same crop, shaded parts lightened", render.ruled(lifted, stations, scale)))
        if street:
            views.append(("S", f"Street view from C toward the junction, {street['view'].get('capture_date') or 'date unknown'}", street["image"]))
        if start:
            draft = {"strips": start, "partial_strips": [], "cannot_determine": []}
            break
        if not budget_left():
            return failed("budget_exhausted")
        try:
            draft = drafter.run(stage, _sheet(views, d / "input.png"), draft_prompt(current),
                                lambda v: validate_draft(v, current, p["min_gap_px"]))
        except ValidationError as error:
            return failed("draft_failed", str(error))
        write_json(d / "draft.json", draft)
        border = [i for i in check(draft["strips"]) if i["code"] == "touches_crop_border"]
        wider = frame.expanded(round(p["widen_m"] / mpp)) if mpp else None
        if not border or widened >= p["max_widen"] or wider is None:
            break
        if not _inside(wider, image.size):
            events.append({"event": "widen_blocked_by_image_edge", "sides": [i["detail"]["side"] for i in border]})
            break
        # The carriageway may continue outside the crop: redraft on a wider one rather than extrapolate.
        widened += 1
        frame = wider
        events.append({"event": "crop_widened", "attempt": widened, "width": frame.width,
                       "sides": [i["detail"]["side"] for i in border]})

    def review(strips, issues, number, rounds):
        r = folder / f"round_{number}"
        r.mkdir(parents=True, exist_ok=True)
        write_json(r / "strips.json", strips)
        write_json(r / "issues.json", issues)
        views = [("boundaries", "Travel UP; current boundaries e#, regions r#", render.overlay(raw, strips, scale)),
                 ("regions", "Each region's own pixels, unmarked", render.panels(raw, strips, scale))]
        if lifted is not None:
            views.append(("lightened", "Same boundaries, shaded parts lightened", render.overlay(lifted, strips, scale)))
        if street:
            views.append(("S", "Street view from C with the boundaries e# projected",
                          street_panel.projected_edges(street["image"], street["view"], frame, strips, street["camera_px"], mpp)))
        keys = [i["key"] for i in issues]
        value = reviewer.run(f"{prefix}review_{sid}_r{number}", _sheet(views, r / "input.png"),
                             review_prompt(current, strips, issues, number, rounds), lambda v: validate_review(v, keys))
        write_json(r / "review.json", value)
        latest["round"] = number
        return value

    def apply(strips, edits):
        return apply_edits(strips, edits, frame.width, p["min_gap_px"], origin=f"review_r{latest['round']}")

    result = run_review_loop(draft["strips"], check=check, review=review, apply=apply, signature=signature,
                             magnitude=change_px, policy=p, budget_left=budget_left)
    final = result["state"]
    heading = frame.heading_deg
    section = {"id": sid, "source_id": "satellite", "kind": spec["kind"], "direction": spec["direction"],
               "link_id": spec["link_id"], "heading_deg": heading, "arm": spec.get("arm"),
               **({"contrast_stretched": True} if current.get("contrast_stretched") else {}),
               **({"shade_panel": True} if current.get("shade_panel") else {}),
               **({"street_view": current["street_view_used"]} if current.get("street_view_used") else {}),
               "rotation_ccw": int(heading) % 360 if frame.pixels_preserved() else None,
               **({"window_anchor": spec["anchor"]} if spec.get("anchor") else {}),
               "regions": section_regions(final, frame, spec["kind"]), "coverage": "candidate_regions_only",
               "geometry_source": "vlm_draft_review_loop",
               "review": {"status": result["status"], "stop_reason": result["stop_reason"], "rounds": len(result["rounds"]),
                          "open_issues": result["open_issues"]},
               "notes": "Model-traced candidate regions; unvalidated hypotheses that may contain non-road pavement or several facilities."}
    record = {"section_id": sid, "frame": frame.serialize(), "stations": stations, "status": result["status"],
              "stop_reason": result["stop_reason"], "revised": result["revised"], "draft": draft["strips"], "final": final,
              "rounds": result["rounds"], "open_issues": result["open_issues"], "dismissed": result["dismissed"],
              "partial_strips": draft["partial_strips"], "cannot_determine": draft["cannot_determine"], "events": events}
    write_json(folder / "result.json", record)
    return section, record


def read_anchor(image, spec, client, output, policy, budget_left):
    """One model reading of a long probe strip around the section (see anchor.py): (probe, reading or None, record)."""
    sid = spec["id"]
    folder = Path(output) / sid / "anchor"
    folder.mkdir(parents=True, exist_ok=True)
    probe = anchor.probe_frame(spec, policy.get("anchor_probe"))
    record = {"section_id": sid, "probe": probe.serialize(), "network_frame": spec["frame"].serialize()}
    raw = render.crop(image, probe)
    stretched = render.lit(raw)
    sheet = anchor.gridded(stretched or raw, probe, spec)
    if not budget_left():
        return probe, None, {**record, "outcome": "budget_exhausted"}
    try:
        value = drafter_run(client, f"anchor_{sid}", sheet, folder / "input.png", anchor.anchor_prompt(spec, probe, stretched is not None),
                            lambda v: anchor.validate_anchor(v, spec, probe, policy.get("anchor_probe")))
    except ValidationError as error:
        return probe, None, {**record, "outcome": "reading_failed", "error": str(error)}
    write_json(folder / "reading.json", value)
    return probe, value, {**record, "reading": value}


def place_sections(image, readings, policy):
    """Section windows from all readings of a site: the start of each is agreed per road arm (anchor.harmonise)."""
    frame_policy = {**FRAME_DEFAULTS, **(policy.get("frames") or {})}
    starts = anchor.harmonise([(spec, probe, value) for spec, probe, value, _ in readings], (policy.get("anchor_probe") or {}).get("arm_tolerance_m"))
    out = []
    for spec, probe, value, record in readings:
        if value is None:
            out.append((spec, record))
            continue
        start_m, how = starts[spec["id"]]
        frame, info = anchor.anchored_frame(spec, probe, value, frame_policy, start_m, how)
        if not _inside(frame, image.size):
            out.append((spec, {**record, "outcome": "kept_network_window", "reason": "anchored_window_outside_image"}))
            continue
        summary = {"outcome": info.pop("placed"), "line_kind": value["stop_line"]["kind"], "confidence": value["confidence"],
                   "moved": anchor.compare(spec["frame"], frame, spec), **info}
        out.append(({**spec, "frame": frame, "anchor": summary}, {**record, **summary, "frame": frame.serialize()}))
    return out


def drafter_run(client, stage, sheet, path, prompt, validator):
    sheet.save(path)
    return client.run(stage, transport_jpeg(sheet, path), prompt, validator)


def load_georeference(spec, root):
    """Return (lon/lat -> source pixel, ground metres per pixel) for the satellite image."""
    from ..fusion.coordinates import Viewport, affine_points, inverse_affine
    if spec["kind"] == "viewport":
        view = Viewport(**spec["viewport"])
        return view.to_pixel, spec["ground_mpp"]
    require(spec["kind"] == "atlas_affine", "Unknown georeference kind")
    data = read_json(resource_path(root, spec["file"]))
    view = Viewport(**data["atlas_viewport"])
    inverse = inverse_affine(data["primary_to_atlas"])

    def to_pixel(lon, lat):
        x, y = affine_points([view.to_pixel(lon, lat)], inverse)[0]
        return float(x), float(y)
    return to_pixel, data["primary_effective_ground_mpp"]


def plan_site(config, root):
    """Section crop windows and the call budget, without contacting any service."""
    policy = {**DEFAULTS, **config.get("policy", {})}
    image = Image.open(resource_path(root, config["satellite"]["path"])).convert("RGB")
    to_pixel, mpp = load_georeference(config["satellite"]["georeference"], root)
    with open(resource_path(root, config["network"]["link_csv"]), newline="", encoding="utf-8-sig") as stream:
        links = list(csv.DictReader(stream))
    with open(resource_path(root, config["network"]["node_csv"]), newline="", encoding="utf-8-sig") as stream:
        node = next((r for r in csv.DictReader(stream) if int(float(r["node_id"])) == int(config["node_id"])), None)
    require(node is not None, "Site node is not in the node file")
    specs = frames_from_network(int(config["node_id"]), (float(node["x_coord"]), float(node["y_coord"])), links,
                                to_pixel, mpp, image.size, policy.get("frames"))
    for spec in specs:
        spec["driving_side"] = (policy.get("frames") or {}).get("driving_side", "right")
    usable = sum(s["usable"] for s in specs)
    stages = usable * (1 + policy["max_widen"] + policy["max_rounds"])
    anchoring = usable if policy["anchor"] else 0
    budget = {"sections": usable, "anchoring_calls": anchoring,
              "typical_calls": [usable * 2 + anchoring, usable * (1 + policy["max_rounds"]) + anchoring],
              "stages_upper_bound": stages + anchoring, "with_one_repair_per_stage": (stages + anchoring) * 2,
              "hard_cap": policy["max_vlm_calls"] + anchoring}
    return image, specs, policy, budget


def serialize_spec(spec):
    return {**{k: v for k, v in spec.items() if k != "frame"}, "frame": spec["frame"].serialize(),
            "corners": [[round(v, 1) for v in c] for c in spec["frame"].corners()]}


def draw_plan(image, specs, path, network=None):
    """Crop windows on the satellite image; with `network`, the windows the network alone gave are drawn thin."""
    canvas = image.copy()
    d = ImageDraw.Draw(canvas)
    for spec in network or []:
        corners = [tuple(c) for c in spec["frame"].corners()]
        render._dashed(d, corners + [corners[0]], "#c8ccd4", on=14, off=10, width=2)
    for spec in specs:
        if spec.get("anchor", {}).get("start_px"):
            x, y = spec["anchor"]["start_px"]
            d.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#ff3df2", width=4)
        corners = [tuple(c) for c in spec["frame"].corners()]
        color = "#16d5ec" if spec["kind"] == "inbound_stopbar" else "#ffbd59"
        d.line(corners + [corners[0]], fill=color if spec["usable"] else "#ff5555", width=4)
        # A tick on the top side marks the direction of travel.
        top = ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2)
        d.line((spec["frame"].center, top), fill=color, width=3)
        d.text(spec["frame"].center, spec["id"], font=font(30), fill="white", stroke_width=3, stroke_fill="black", anchor="mm")
    canvas.save(path)


def run_site(config_path, root, output, *, plan_only=False, cache_only=False, clients=None, reuse_cache_from=(),
             max_vlm_calls=None):
    """Plan, then draft and review every usable section of one junction."""
    root, output = Path(root), Path(output)
    config = read_json(config_path)
    require(config.get("schema") == "autoloop-site-1", "Unsupported site config schema")
    if max_vlm_calls is not None:  # recorded with the config in the run manifest
        config["policy"] = {**config.get("policy", {}), "max_vlm_calls": max_vlm_calls}
    image, specs, policy, budget = plan_site(config, root)
    to_pixel, _ = load_georeference(config["satellite"]["georeference"], root)
    # forward street views of the approaches (acquisition writes them beside the site config); shown while tracing
    base = Path(config_path).with_name("base_config.json")
    street_views = read_json(base) if policy["street_view"] and base.exists() else None
    geometry = output / "geometry"
    geometry.mkdir(parents=True, exist_ok=True)
    hashes = {"satellite": file_hash(resource_path(root, config["satellite"]["path"])),
              "link_csv": file_hash(resource_path(root, config["network"]["link_csv"])),
              "node_csv": file_hash(resource_path(root, config["network"]["node_csv"]))}
    manifest = {"version": VERSION, "config": config, "input_sha256": hashes}
    mark = digest(manifest)
    path = output / "input_manifest.json"
    if path.exists():
        require(read_json(path)["signature"] == mark, "Inputs changed: choose a new output directory")
    write_json(path, {"signature": mark, **manifest, "reference_used_in_inference": False})
    write_json(geometry / "plan.json", {"budget": budget, "sections": [serialize_spec(s) for s in specs]})
    draw_plan(image, specs, geometry / "plan.png")
    if plan_only:
        write_json(output / "status.json", {"state": "planned"})
        return {"state": "planned", "budget": budget, "sections": [s["id"] for s in specs if s["usable"]]}
    # One anchoring reading per usable section comes on top of the tracing budget.
    cap = policy["max_vlm_calls"] + (budget["sections"] if policy["anchor"] else 0)
    if clients is None:
        from ..hybrid.inference import StageClient
        models = config.get("models", {})
        drafter = StageClient(root, output, {**models.get("draft", {}), "max_api_calls": cap}, cache_only, reuse_cache_from)
        same = models.get("review", models.get("draft", {})) == models.get("draft", {})
        reviewer = drafter if same else StageClient(root, output, {**models["review"], "max_api_calls": cap}, cache_only, reuse_cache_from)
    else:
        drafter, reviewer = clients
    used = {id(c): c for c in (drafter, reviewer)}.values()

    def budget_left():
        return sum(c.calls for c in used) < cap

    write_json(output / "status.json", {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    sections, history, anchors, traced = [], [], [], []
    try:
        usable = [s for s in specs if s["usable"]]
        for spec in specs:
            if not spec["usable"]:
                history.append({"section_id": spec["id"], "status": "unresolved", "stop_reason": spec["unusable_reason"]})
        workers = request_workers()
        own = (lambda: image.copy()) if workers > 1 else (lambda: image)  # each thread renders from its own copy
        if policy["anchor"]:
            # every section is read first, so that the windows of one road arm can be placed consistently
            readings = parallel_map(lambda spec: (spec, *read_anchor(own(), spec, drafter, geometry, policy, budget_left)), usable, workers)
            placed = place_sections(image, readings, policy)
            usable = [spec for spec, _ in placed]
            anchors = [record for _, record in placed]
            write_json(geometry / "anchors.json", anchors)
        traced = list(usable)
        draw_plan(image, traced, geometry / "plan.png", network=specs if policy["anchor"] else None)

        def trace(spec):
            street = None
            if street_views and spec["kind"] == "inbound_stopbar":
                view = street_panel.pick_view(street_views, spec["direction"])
                if view:
                    street = {"view": view, "image": street_panel.load(view, root), "to_pixel": to_pixel}
            return run_section(own(), spec, drafter, reviewer, geometry, policy, budget_left, street)

        # Sections are independent once placed; with several workers their review loops run side by side.
        for section, record in parallel_map(trace, usable, workers):
            sections.append(section)
            history.append(record)
        write_json(geometry / "sections.partial.json", sections)
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error_type": type(error).__name__})
        raise
    write_json(geometry / "sections.json", sections)
    write_json(geometry / "history.json", history)
    statuses = [h["status"] for h in history]
    summary = {"state": "complete", "version": VERSION, "site_id": config["site_id"], "sections": len(history),
               "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
               "stop_reasons": {h["section_id"]: h["stop_reason"] for h in history},
               "review_rounds": {h["section_id"]: len(h.get("rounds", [])) for h in history},
               "vlm_calls_this_run": sum(c.calls for c in used), "vlm_cache_hits_this_run": sum(c.hits for c in used),
               "actual_models": sorted({f"{getattr(c, 'provider', '?')}/{getattr(c, 'model', '?')}" for c in used}),
               "budget": budget, "reference_used_in_inference": False,
               "anchoring": {a["section_id"]: a["outcome"] for a in anchors},
               "scope": "Loop termination and provenance only; not an accuracy score for the traced boundaries."}
    write_json(geometry / "summary.json", summary)
    write_json(output / "status.json", {"state": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    return summary
