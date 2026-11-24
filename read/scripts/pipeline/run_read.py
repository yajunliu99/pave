"""PAVE Read for one site: the lanes, lane use and turning movements of every approach of a junction, read by a
vision-language model from a satellite crop and street-level views (about 12-20 model requests at a four-arm junction).

1. trace    one request per road arm: stop line, exit start and every longitudinal boundary of both carriageways, and
            what lies between each pair (inbound, outbound or neither; motor lane, bicycle lane, parking, median ...).
            The trace is told that rails laid in a lane are not lane lines, and always to give a stop line.
2. read     one request per approach, independent of the trace: the satellite crop and the approach's street views
            (near look-back, mid and far forward), read blind for the lane layout at the stop bar.
3. retrace  once per arm, only where a settled reading sees more motor lanes on the approach than the trace, or a
            traced motor lane is measurably wider or narrower than one lane (--lane-width-m) or marked low-confidence, or
            a bicycle strip as wide as a lane lies between two lanes: the trace is shown back with the measured issues
            (and the reading, when it saw more lanes), and the model corrects or keeps it. A reading that sees fewer
            lanes is not a reason to remove one: the lane-use step judges each traced lane from the images.
4. use      one request per approach: whether each traced lane exists, its allowed turns and, per turn, the exit and
            its lane pairs; the lanes are projected into the street views, and the prompt gives the traced cross-section
            (bicycle lanes, parking and medians between the lanes). Lanes and turns are the permanent layout; temporary
            traffic control (cones, barrels, portable signs, work zones) is reported apart. A lane judged not to exist
            is not counted only when the independent reading agrees (7). Notes are added only where their condition
            holds: a lane between a bicycle lane and the curb that measures at least a lane's width is normally where
            right-turning traffic waits; a bicycle strip as wide as a lane between two lanes should be checked for a lane.
   4b.      a lane given left and through by inference (no arrow) while other lanes carry through is asked about once
            more, with a note that a left-turn lane may lie outside the outlines.
The trace sheets mark the street-view cameras on the crop, and a re-trace projects the traced boundaries into them.
5. detail   only for an approach whose `use` answer asked for native-resolution crops.
6. addlane  once per approach whose final `use` answer reports a motor lane at the stop bar with no outline of its own:
            the arm is re-traced with that report, its inbound lanes are taken from the new trace (its exits are kept,
            so no other approach's answer changes), and `use` (and `detail`) run again on the new lanes.
7. keepuse  where a `use` answer judges traced lanes not to exist, they are dropped only when the blind reading (2), which
            never sees the trace, is settled and counts no more lanes than would remain. Otherwise the lanes stay and
            `use` (and `detail`) run again, told so; that answer may drop no lane and must give the kept lanes' turns.
   7b.      a lane as wide as a motor lane is not dropped on the reading's agreement alone: `use` is asked once more,
            told its width.
8. leftlane once per approach where an exit lies to the left (by link headings) but no lane of the settled `use` answer
            allows a left turn: the arm is re-traced and asked to look for a left-turn lane on the road-centre side. As
            in 6, the new inbound lanes are kept only if there are more of them, then `use` runs again (and 7 on it).
   8b.      where that re-trace finds no lane, `use` is asked once more whether the leftmost lane allows left turns.
So a lane is added only when a second trace finds it, and removed only when the blind reading agrees; `use` alone can
only ask. --stop-after retrace runs steps 1-3 only and writes trace_check.json (a cheap trace test).

Window size and placement come from link geometry only; no lane count from the network and no reference data is used.
Writes lanes.json, movements.json, traces.json, readings.json and uses.json, every sheet and prompt sent (sheets/ and
vlm/), result.jpg (every traced region on the satellite image, labelled, with an arrow for each proposed turn) and summary.json with the requests made.
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pave_read.autoloop import render, street_panel
from pave_read.autoloop.geometry_stage import load_georeference
from pave_read.fusion.lane_check import validate_reading
from pave_read.hybrid.common import ValidationError, file_hash, parallel_map, read_json, request_workers, resource_path, write_json
from pave_read.hybrid.inference import StageClient
from pave_read.harness import figure, geometry, stages
from pave_read.harness import views as street

CODE = ["scripts/pipeline/run_read.py", "pave_read/harness/geometry.py", "pave_read/harness/stages.py",
        "pave_read/harness/views.py", "pave_read/fusion/lane_check.py", "pave_read/autoloop/street_panel.py",
        "pave_read/autoloop/render.py", "pave_read/autoloop/frames.py", "pave_read/autoloop/strips.py",
        "pave_read/hybrid/inference.py", "pave_read/hybrid/prompts.py"]


def code_hashes():
    """sha256 of every source file whose code shapes a request or its reading, recorded with each run."""
    return {path: file_hash(ROOT / path) for path in CODE}


def save(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagery", type=Path, required=True, help="site imagery folder (site.json, base_config.json)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scale", type=float, default=1.5, help="display scale of the trace sheets")
    p.add_argument("--max-calls", type=int, default=40)
    p.add_argument("--lane-width-m", type=float, nargs=2, default=(2.5, 5.0), help="a motor lane outside this width range is re-checked")
    p.add_argument("--reuse-cache-from", type=Path, action="append", default=[], help="earlier run folders whose identical requests are reused")
    p.add_argument("--prepare-only", action="store_true", help="build the trace and reading sheets and prompts, make no request")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--stop-after", choices=("retrace",), default=None,
                   help="stop after the trace, blind reading and re-trace, and write trace_check.json (no lane-use request)")
    a = p.parse_args()
    out = a.output.resolve()
    site, base = read_json(a.imagery / "site.json"), read_json(a.imagery / "base_config.json")
    image = Image.open(resource_path(ROOT, site["satellite"]["path"])).convert("RGB")
    to_pixel, mpp = load_georeference(site["satellite"]["georeference"], ROOT)
    links = list(csv.DictReader(open(resource_path(ROOT, site["network"]["link_csv"]), encoding="utf-8-sig")))
    node = next(r for r in csv.DictReader(open(resource_path(ROOT, site["network"]["node_csv"]), encoding="utf-8-sig"))
                if int(float(r["node_id"])) == int(site["node_id"]))
    windows = geometry.arm_windows(int(site["node_id"]), (float(node["x_coord"]), float(node["y_coord"])), links, to_pixel, mpp, image.size)
    name = site["name"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                                     "code_sha256": code_hashes()})
    options = {"preset": "default", "max_tokens": 8192, "temperature": 0.0, "max_api_calls": a.max_calls,
               "stage_max_tokens": {"trace": 12288, "retrace": 12288, "use": 16384, "detail": 16384,
                                    "keepuse": 16384, "keepdetail": 16384, "leftlane": 12288, "leftuse": 16384, "leftask": 16384, "leftaskdetail": 16384,
                                    "leftaskkeep": 16384, "leftaskkeepdetail": 16384, "wide": 16384, "sharedask": 16384,
                                    "leftdetail": 16384, "leftkeepuse": 16384, "leftkeepdetail": 16384}}
    client = StageClient(ROOT, out, options, cache_only=a.cache_only, reuse_cache_from=[x.resolve() for x in a.reuse_cache_from])

    # --- 1 and 2: traces of every arm and blind readings of every approach, independent of each other
    jobs = []
    for w in windows:
        raw = render.crop(image, w["frame"])
        raw_path = save(raw, out / "sheets" / f"arm_{w['arm']}_raw.jpg")
        views = street.approach_views(base, w["sections"]["inbound"]["direction"], ROOT) if "inbound" in w["sections"] else []
        captions = [street.caption(v) for v in views]
        photos = [(street_panel.load(v, ROOT), c) for v, c in zip(views, captions)]
        sheet, shaded = stages.trace_sheet(raw, w, a.scale, cameras=(views, to_pixel), photos=photos)
        w.update(raw=raw, raw_path=raw_path, shaded=shaded, views=views, captions=captions,
                 trace_sheet=save(sheet, out / "sheets" / f"trace_{w['arm']}.jpg"))
        w["trace_prompt"] = stages.trace_prompt(w, shaded, name, captions)
        (out / "sheets" / f"trace_{w['arm']}.prompt.txt").write_text(w["trace_prompt"], encoding="utf-8")
        jobs.append(("trace", w))
        if "inbound" in w["sections"]:
            direction = w["sections"]["inbound"]["direction"]
            (out / "sheets").mkdir(parents=True, exist_ok=True)
            w["read_sheet"] = street.reading_sheet(raw, views, ROOT, out / "sheets" / f"read_{direction}.jpg")
            w["read_prompt"] = street.reading_prompt(captions)
            jobs.append(("read", w))
    print({"site": name, "arms": [w["arm"] for w in windows], "first_requests": len(jobs)}, flush=True)
    if a.prepare_only:
        write_json(out / "status.json", {"state": "prepared"})
        return 0

    def first(job):
        kind, w = job
        if kind == "trace":
            w["trace"] = client.run(f"trace_{w['arm']}", w["trace_sheet"], w["trace_prompt"],
                                    lambda v, w=w: stages.validate_trace(v, w))
            w["found"] = geometry.regions(w["trace"], w)
        else:
            direction = w["sections"]["inbound"]["direction"]
            try:
                w["reading"] = client.run(f"read_{direction}", w["read_sheet"], w["read_prompt"], validate_reading)
            except ValidationError as error:
                w["reading"], w["reading_error"] = None, str(error)
    parallel_map(first, jobs, request_workers())

    # --- 3: re-trace an arm once, where a settled reading sees more inbound lanes than the trace or a lane is
    #        measurably too wide or too narrow (or low-confidence). A reading that sees fewer lanes is recorded but not
    #        shown: such re-traces removed a correct lane more often than a wrong one.
    def retrace(w):
        reading = w.get("reading")
        traced = len(geometry.motor_regions(w["found"], "inbound"))
        settled = bool(reading and reading["count_settled"] and reading["count_confidence"] != "low")
        disagrees = settled and len(reading["lanes"]) != traced
        more = settled and len(reading["lanes"]) > traced
        issues = geometry.trace_issues(w["found"], w["sections"], *a.lane_width_m)
        w["check"] = {"traced": traced, "seen": len(reading["lanes"]) if reading else None, "settled": settled,
                      "disagrees": disagrees, "reading_sees_more": more, "issues": issues}
        if not more and not issues:
            w["check"]["action"] = ("kept_reading_fewer" if disagrees else "agree") if settled else "no_issue"
            return
        photos = [(street.with_boundaries(v, ROOT, w, w["trace"], to_pixel, mpp), c) for v, c in zip(w["views"], w["captions"])]
        sheet, _ = stages.trace_sheet(w["raw"], w, a.scale, overlay=w["trace"], cameras=(w["views"], to_pixel), photos=photos)
        path = save(sheet, out / "sheets" / f"retrace_{w['arm']}.jpg")
        prompt = stages.retrace_prompt(w, w["shaded"], name, w["trace"], w["found"], reading if more else None, issues, more, w["captions"])
        w["first_trace"], w["first_found"] = w["trace"], w["found"]
        w["trace"] = client.run(f"retrace_{w['arm']}", path, prompt, lambda v, w=w: stages.validate_trace(v, w))
        w["found"] = geometry.regions(w["trace"], w)
        w["check"].update(action="retraced", retraced=len(geometry.motor_regions(w["found"], "inbound")),
                          issues_after=geometry.trace_issues(w["found"], w["sections"], *a.lane_width_m))
    parallel_map(retrace, windows, request_workers())
    if a.stop_after == "retrace":
        write_json(out / "trace_check.json", {
            "code_sha256": code_hashes(), "site": name, "node_id": site["node_id"],
            "model": f"{client.provider}/{client.model}", "requests_this_run": client.calls, "cache_hits": client.hits,
            "checks": {w["sections"]["inbound"]["direction"]: w.get("check") for w in windows if "inbound" in w["sections"]},
            "traces": {w["arm"]: {"trace": w["trace"], **({"first_trace": w["first_trace"]} if "first_trace" in w else {})} for w in windows}})
        print({"site": name, "stopped_after": "retrace", "requests": client.calls, "cache_hits": client.hits}, flush=True)
        return

    # --- 4 and 5: lane use and movements per approach
    exits = []
    for w in windows:
        if "outbound" in w["sections"]:
            s = w["sections"]["outbound"]
            exits.append({**s, "regions": [r["region_id"] for r in geometry.motor_regions(w["found"], "outbound")],
                          "polygons": [(f"{s['section_id']}:{r['region_id']}", r["polygon_px"]) for r in geometry.motor_regions(w["found"], "outbound")]})
    overview = [poly for e in exits for poly in e["polygons"]]

    def ask_use(w, use_stage, detail_stage, kept=(), kept_why="", left_note=(), wide_note=(), shared_note=()):
        """One `use` request on the arm's current inbound lanes (and its `detail` follow-up): (answer, first answer
        when a detail request revised it, exit candidates). kept: region IDs that stay lanes (step 7); with them the
        answer may drop no lane."""
        s = w["sections"]["inbound"]
        lanes = geometry.motor_regions(w["found"], "inbound")
        candidates = [{**e, "compass": stages.compass(e["travel_heading_deg"]),
                       "turn_angle": stages.turn_angle(s["travel_heading_deg"], e["travel_heading_deg"]),
                       "geometric_hint": stages.geometric_hint(stages.turn_angle(s["travel_heading_deg"], e["travel_heading_deg"]))}
                      for e in exits]
        outlines = [(r["region_id"], r["polygon_px"]) for r in lanes]
        photos = [(street.with_regions(v, ROOT, outlines, to_pixel, mpp), c + f" ({v['id']})") for v, c in zip(w["views"], w["captions"])]
        sheet, extra = stages.use_sheet(image, w, w["found"], overview, w["node_px"], mpp, photos)
        path = save(sheet, out / "sheets" / f"{use_stage}_{s['direction']}.jpg")
        sources = {"approach": extra["crop"], "overview": extra["overview"], **{v["id"]: street_panel.load(v, ROOT) for v in w["views"]}}
        prompt = stages.use_prompt(s["direction"], name, lanes, candidates, w.get("reading"), [c + f" ({v['id']})" for v, c in zip(w["views"], w["captions"])],
                                   extra["lifted"], found=w["found"], kept=kept, kept_why=kept_why, left_note=left_note,
                                   low_m=a.lane_width_m[0], wide_note=wide_note, shared_note=shared_note)
        keep = list(kept)
        answer = client.run(f"{use_stage}_{s['direction']}", path, prompt,
                            lambda v: stages.validate_use(v, lanes, candidates, sources, must_exist=keep, keep_all=bool(keep)))
        first = None
        if answer.get("detail_requests"):
            dpath = save(stages.detail_sheet(answer["detail_requests"], sources), out / "sheets" / f"{detail_stage}_{s['direction']}.jpg")
            first = answer
            answer = client.run(f"{detail_stage}_{s['direction']}", dpath, stages.detail_prompt(prompt, answer),
                                lambda v: stages.validate_use(v, lanes, candidates, sources, allow_details=False, must_exist=keep, keep_all=bool(keep)))
        return answer, first, candidates

    def settle(w, answer, first, candidates, stage, detail):
        """7: lanes `answer` judges not to exist are dropped only when the blind reading is settled and counts no more
        lanes than would remain; otherwise `use` is asked again (stage, detail) and must keep every lane."""
        lanes = geometry.motor_regions(w["found"], "inbound")
        rejected = [l for l in answer["lanes"] if l["exists"] == "no"]
        if not rejected:
            return answer, first, candidates
        reading = w.get("reading")
        settled = bool(reading and reading["count_settled"] and reading["count_confidence"] != "low")
        seen, remain = (len(reading["lanes"]) if reading else None), len(lanes) - len(rejected)
        agreed = settled and seen <= remain
        ids = [l["region_id"] for l in rejected]
        w.setdefault("removals", []).append({"stage": stage, "rejected": [{"region_id": l["region_id"], "evidence": l["evidence"]} for l in rejected],
                                             "lanes": len(lanes), "seen": seen, "settled": settled, "dropped": agreed})
        width = {r["region_id"]: r.get("width_m") or 0 for r in lanes}
        wide = [(l["region_id"], width.get(l["region_id"], 0)) for l in rejected if width.get(l["region_id"], 0) >= a.lane_width_m[0]]
        if agreed and wide and not stage.startswith("wide"):
            # 7b: a lane as wide as a motor lane is not dropped on the reading's agreement alone: ask once more, stating its width
            w["removals"][-1]["wide_reask"] = [r for r, _ in wide]
            w[f"before_wide{stage}"] = {"use": answer, **({"first_use": first} if first else {})}
            return ask_use(w, f"wide{stage}", f"wide{stage}detail", wide_note=wide)
        if agreed:
            return answer, first, candidates
        why = (f"The separate reading above counted {seen} general motor lane(s) at the stop bar, more than the {remain} that would remain"
               f" without {', '.join(ids)}." if settled else
               "The separate reading above could not settle its count." if reading else "No separate street-view reading is available.")
        w[f"before_{stage}"] = {"use": answer, **({"first_use": first} if first else {})}
        return ask_use(w, stage, detail, kept=ids, kept_why=why)

    def retrace_with(w, stage, **note):
        """A re-trace of the arm for step 6 or 8 (note: missing= or no_left=), drawn over its latest adopted trace.
        Returns the new trace, or None after a failed request (recorded in w[stage]["error"])."""
        current = w.get("adopted_trace") or w["trace"]
        photos = [(street.with_boundaries(v, ROOT, w, current, to_pixel, mpp), c) for v, c in zip(w["views"], w["captions"])]
        sheet, _ = stages.trace_sheet(w["raw"], w, a.scale, overlay=current, cameras=(w["views"], to_pixel), photos=photos)
        path = save(sheet, out / "sheets" / f"{stage}_{w['arm']}.jpg")
        prompt = stages.retrace_prompt(w, w["shaded"], name, current, w["found"], None, (), False, w["captions"], **note)
        try:
            return client.run(f"{stage}_{w['arm']}", path, prompt, lambda v, w=w: stages.validate_trace(v, w))
        except ValidationError as error:
            w[stage]["error"] = str(error)
            return None

    def adopt(w, stage, trace, answer, first):
        """Keep a re-trace's inbound lanes (and the exits as they were) when it has more of them. True when kept."""
        before = w[stage]["lanes_before"]
        found = geometry.regions(trace, w)
        after = len(geometry.motor_regions(found, "inbound"))
        w[stage].update(lanes_after_trace=after, kept=after > before)
        w[f"{stage}_trace"] = trace
        if after > before:
            w[f"before_{stage}"] = {"use": answer, **({"first_use": first} if first else {})}
            w["found"] = {"inbound": found["inbound"], "outbound": w["found"]["outbound"]}
            w["adopted_trace"] = trace
        return after > before

    def use(w):
        if not geometry.motor_regions(w["found"], "inbound"):
            w["use"], w["use_error"] = None, "no inbound motor lanes traced"
            return
        answer, first, candidates = ask_use(w, "use", "detail")
        shared = stages.shared_inferred(answer)
        if shared:
            # 4b: a lane given left and through by inference while other lanes carry through: ask once more
            w["sharedask"] = {"lanes": shared}
            w["before_sharedask"] = {"use": answer, **({"first_use": first} if first else {})}
            answer, first, candidates = ask_use(w, "sharedask", "sharedaskdetail", shared_note=shared)
        missing = [(m["where"], m["evidence"]) for m in answer.get("missing_lanes") or []]
        if missing:
            # 6: re-trace the arm with the lane `use` saw but had no outline for; keep the new inbound lanes only if
            #    there are more of them (this step is asked to add a lane), and keep the exits as they were.
            w["addlane"] = {"missing": [{"where": a_, "evidence": b_} for a_, b_ in missing],
                            "lanes_before": len(geometry.motor_regions(w["found"], "inbound"))}
            trace = retrace_with(w, "addlane", missing=missing)
            if trace is not None and adopt(w, "addlane", trace, answer, first):
                answer, first, candidates = ask_use(w, "reuse", "redetail")
        answer, first, candidates = settle(w, answer, first, candidates, "keepuse", "keepdetail")
        lefts = [e for e in candidates if e["geometric_hint"].startswith("to the left")]
        if lefts and not any("left" in l["turns"] for l in answer["lanes"]):
            # 8: an exit lies to the left but no lane allows a left turn: look again for a left-turn lane
            w["leftlane"] = {"left_exits": [e["section_id"] for e in lefts],
                             "lanes_before": len(geometry.motor_regions(w["found"], "inbound"))}
            trace = retrace_with(w, "leftlane", no_left=lefts)
            if trace is not None and adopt(w, "leftlane", trace, answer, first):
                answer, first, candidates = ask_use(w, "leftuse", "leftdetail")
                answer, first, candidates = settle(w, answer, first, candidates, "leftkeepuse", "leftkeepdetail")
            else:
                # 8b: no left-turn lane found: ask once more whether the leftmost lane allows left turns
                w["before_leftask"] = {"use": answer, **({"first_use": first} if first else {})}
                w["leftlane"]["asked_again"] = True
                answer, first, candidates = ask_use(w, "leftask", "leftaskdetail", left_note=lefts)
                answer, first, candidates = settle(w, answer, first, candidates, "leftaskkeep", "leftaskkeepdetail")
        if first:
            w["first_use"] = first
        w["use"], w["candidates"] = answer, candidates
    parallel_map(use, [w for w in windows if "inbound" in w["sections"]], request_workers())

    # --- outputs
    sections, movements = [], []
    index = {}
    for w in windows:
        for side, key in (("inbound", "inbound_stopbar"), ("outbound", "outbound_receiving")):
            if side not in w["sections"]:
                continue
            s = w["sections"][side]
            uses = {l["region_id"]: l for l in ((w.get("use") or {}).get("lanes") or [])} if side == "inbound" else {}
            regs, kept = [], 0
            for r in w["found"][side]:
                u = uses.get(r.get("region_id"))
                dropped = bool(u and u.get("exists") == "no")
                if r.get("region_id") and not dropped:
                    kept += 1
                    index[(s["section_id"], r["region_id"])] = kept
                existence = ("rejected_by_lane_use" if dropped else "supported") if r.get("region_id") else \
                    ("not_at_gate" if r["surface_type"] in geometry.MOTOR else "not_motor")
                regs.append({"region_id": r.get("region_id"), "surface_type": r["surface_type"], "existence": existence,
                             "motor_lane_index": index.get((s["section_id"], r.get("region_id"))) if not dropped else None,
                             "width_m": r["width_m"], "polygon_px": r["polygon_px"],
                             "trace_confidence": r["confidence"], "trace_evidence": r["evidence"],
                             **({"lane_use_prediction": {"exists": u.get("exists"), "allowed_turns": u["turns"], "basis": u["basis"],
                                                         "confidence": u["confidence"], "evidence": u["evidence"]}} if u else {})})
            sections.append({"section_id": s["section_id"], "kind": key, "direction": s["direction"], "arm": w["arm"],
                             "link_id": s["link_id"], "regions": regs, "count": {"model_motor_count": kept}})
    for w in windows:
        if "inbound" not in w["sections"] or not w.get("use"):
            continue
        s = w["sections"]["inbound"]
        for m in w["use"]["movements"]:
            target = next((e for e in w["candidates"] if e["section_id"] == m.get("to")), None)
            movements.append({"from_direction": s["direction"], "turn": m["turn"], "status": m["status"],
                              "in_section_id": s["section_id"], "out_section_id": m.get("to") if target else None,
                              "to_direction": target["direction"] if target else None, "reason": m.get("reason", ""),
                              "lane_pairs": [{"in_region_id": a_, "out_region_id": b_, "ib_lane": index.get((s["section_id"], a_)),
                                              "ob_lane": index.get((m["to"], b_))} for a_, b_ in (m.get("lane_pairs") or [])]})
    write_json(out / "lanes.json", sections)
    write_json(out / "movements.json", movements)
    checks = {w["sections"]["inbound"]["direction"]: w.get("check") for w in windows if "inbound" in w["sections"]}
    readings = {w["sections"]["inbound"]["direction"]: w.get("reading") for w in windows if "inbound" in w["sections"]}
    traces = {w["arm"]: {"trace": w["trace"], **({"first_trace": w["first_trace"]} if "first_trace" in w else {}),
                         **({"addlane_trace": w["addlane_trace"]} if "addlane_trace" in w else {}),
                         **({"leftlane_trace": w["leftlane_trace"]} if "leftlane_trace" in w else {}),
                         "frame": w["frame"].serialize(), "stations": w["stations"], "stations_m": w["stations_m"], "sections": w["sections"]}
              for w in windows}
    write_json(out / "traces.json", traces)
    write_json(out / "readings.json", {"readings": readings, "checks": checks})
    write_json(out / "uses.json", {w["sections"]["inbound"]["direction"]: {"use": w.get("use"), **({"first_use": w["first_use"]} if "first_use" in w else {}),
                                                                           **({"addlane": w["addlane"]} if "addlane" in w else {}),
                                                                           **({"before_addlane": w["before_addlane"]} if "before_addlane" in w else {}),
                                                                           **{k: w[k] for k in ("removals", "before_keepuse", "leftlane", "before_leftlane",
                                                                                                "before_leftkeepuse", "before_leftask", "before_leftaskkeep") if k in w},
                                                                           **{k: w[k] for k in w if k.startswith("before_wide")},
                                                                           **{k: w[k] for k in ("sharedask", "before_sharedask") if k in w}}
                                   for w in windows if "inbound" in w["sections"]})
    # result figure: every traced region on the satellite, labelled, with an arrow for each proposed turn
    save(figure.overview(image, sections, movements), out / "result.jpg")
    calls = {}
    for folder in (out / "vlm").glob("*"):
        stage = folder.name.split("_")[0]
        calls[stage] = calls.get(stage, 0) + len(list(folder.glob("*.raw.json")))
    write_json(out / "summary.json", {"code_sha256": code_hashes(), "site": name, "node_id": site["node_id"], "model": f"{client.provider}/{client.model}",
                                      "requests_this_run": client.calls, "cache_hits": client.hits, "answers_by_stage": calls,
                                      "checks": checks,
                                      "addlane": {w["sections"]["inbound"]["direction"]: w["addlane"] for w in windows if "addlane" in w},
                                      "removals": {w["sections"]["inbound"]["direction"]: w["removals"] for w in windows if "removals" in w},
                                      "leftlane": {w["sections"]["inbound"]["direction"]: w["leftlane"] for w in windows if "leftlane" in w},
                                      "temporary_control": {w["sections"]["inbound"]["direction"]: (w.get("use") or {}).get("temporary_control")
                                                            for w in windows if "inbound" in w["sections"]},
                                      "counts": {s["section_id"]: s["count"]["model_motor_count"] for s in sections}})
    write_json(out / "status.json", {"state": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                                     "requests": client.calls})
    print({"site": name, "requests": client.calls, "counts": {s["section_id"]: s["count"]["model_motor_count"] for s in sections}, "checks": checks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
