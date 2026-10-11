"""One dashboard for the VLM-analysed intersections, with UTDF and OSM alongside.

Output: a single self-contained HTML file (default dashboard/intersection_explorer.html). Every intersection's data and
images (AVIF) and the map library are inside it; an intersection's data is parsed only when it is selected. Only
the background map tiles need a network connection. Per-intersection parts are cached outside the synced folder
(<runs root>/dashboard_build) so that updating one intersection does not reprocess the others.

Per intersection it shows: lanes traced and classified by the VLM on the satellite image (lane number, type,
movement), each approach's lane layout from the VLM, UTDF and OSM side by side with the visual check of
disagreements, the movement comparison, every street view with the model's observation boxes, the signal and
sign audit, and the labelled figures. UTDF and OSM come from ../net2cell_utdf (read only); VLM results come from
every run exported to the results folder (run_site_pipeline.py updates its own site here as its last step) and the
run folders. Model output is a prediction, not reference data.

Runs with no UTDF intersection, for example on another city's network, are added with --extra-runs. They are shown
with the VLM and street-view readings only. Their position comes from their own network, and their keys sit above
EXTRA_ID so that they never collide with a UTDF intersection. Without ../net2cell_utdf, or with --no-utdf-sites, the
page holds the extra runs alone.
"""
import argparse
import base64
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.downstream import with_exit_audits
from movement_fixer.fusion.lane_check import reference_name
from movement_fixer.paths import results_root, runs_root
from movement_fixer.utdf_matches import node_utdf as utdf_by_node

TURN = {"left": "L", "through": "T", "right": "R", "u_turn": "U"}
UTDF_TURN = {"left": "L", "thru": "T", "right": "R", "uturn": "U"}
SAT = 1600            # satellite image width shown in the dashboard
VIEW_MAX = 480        # street views are stored at most this wide
QUALITY = {"sat": 50, "view": 45, "figure": 50, "check": 45}   # AVIF quality per kind of image
ORDER = ["NB", "SB", "EB", "WB", "NE", "SW", "NW", "SE"]
NODES_XY = {}         # node -> (lon, lat); the map marks the analysed node, which a corrected UTDF match may move
EXTRA_ID = 1_000_000  # page keys of runs without a UTDF intersection start here


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_image(source, target, width=None, quality=50):
    """AVIF copy of an image, at most `width` wide; returns its size."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if width and image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        image.save(target, "AVIF", quality=quality, speed=6)
        return list(image.size)


def utdf_movements(movement_csv):
    """incoming link -> UTDF approach prefixes on it (as UTDF names them); some links carry two UTDF approaches."""
    prefix = defaultdict(set)
    with open(movement_csv, newline="", encoding="utf-8-sig") as stream:
        for r in csv.DictReader(stream):
            if r.get("utdf_intid") and r.get("utdf_mvmt") and r.get("type") in UTDF_TURN and r.get("ib_link_id"):
                prefix[r["ib_link_id"]].add(r["utdf_mvmt"][:2])
    return prefix


def utdf_match(prefixes, key, mcmp, claimed):
    """UTDF approach shown for one network approach, and how it was matched.

    By incoming link, as the lane check does; where a link carries several UTDF approaches, the one named like the
    network approach. Without a link match the UTDF approach of the same name is shown, flagged, unless UTDF's own
    link match already puts that name on another leg."""
    if prefixes:
        pick = reference_name(prefixes, key)
        return pick, {"by": "link", "also": sorted(prefixes - {pick})}
    if key not in claimed and any(code[:2] == key for code in mcmp):
        return key, {"by": "name", "also": []}
    return key, None


def turn_of(code):
    t = code[2:]
    return "L" if t.startswith("L") else "T" if t == "T" else "R" if t.startswith("R") else "U"


def lanes_by_source(mcmp, prefix, source):
    """Lane-by-lane turn letters of one approach in one source (lane 1 = leftmost)."""
    lanes = defaultdict(set)
    for code, entry in mcmp.items():
        e = entry.get(source)
        if code[:2] != prefix or not e or not e.get("sib"):
            continue
        for lane in range(int(e["sib"]), int(e.get("eib") or e["sib"]) + 1):
            lanes[lane].add(turn_of(code))
    return [{"n": n, "t": "".join(t for t in "LTRU" if t in lanes[n])} for n in sorted(lanes)]


def vlm_lanes(section, movements):
    """VLM lanes of an approach, left to right: turns read from markings, else turns the movement step paired."""
    paired = defaultdict(set)
    for m in movements:
        if m["in_section_id"] == section["section_id"] and m["status"] == "candidate":
            for p in m["lane_pairs"]:
                paired[p["in_region_id"]].add(TURN[m["turn"]])
    out = []
    for r in sorted((r for r in section["regions"] if r.get("existence") == "supported" and r.get("motor_lane_index")),
                    key=lambda r: r["motor_lane_index"]):
        observed = "".join(TURN[t] for t in ("left", "through", "right", "u_turn") if t in (r.get("lane_use_prediction", {}).get("allowed_turns") or []))
        inferred = "".join(t for t in "LTRU" if t in paired.get(r["region_id"], ()))
        out.append({"n": r["motor_lane_index"], "t": observed or inferred, "basis": "marking" if observed else "movement" if inferred else "unknown",
                    "rail": "rail" in r["surface_type"], "region": r["region_id"]})
    return out


def region_label(region, lane, inbound, paired):
    """Short label in the style of the labelled figures: turns + lane number, or the facility type."""
    kind, existence = lane["surface_type"], lane["existence"]
    if existence == "rejected":
        return "not road", "rejected"
    if kind == "bicycle_strip":
        return "bike", "bike"
    if kind == "rail_area":
        return "rail", "rail"
    if kind in ("median_or_buffer", "shoulder_or_parking"):
        return "buffer" if kind == "median_or_buffer" else "parking", "other"
    if kind not in ("motor_vehicle_lane", "shared_rail_motor_lane"):
        return "?", "unsure"
    if existence == "uncertain":
        return "lane?", "unsure"
    n = lane.get("motor_lane_index") or ""
    if not inbound:
        return f"out{n}", "exit"
    observed = "".join(TURN[t] for t in ("left", "through", "right", "u_turn") if t in ((lane.get("lane_use_prediction") or {}).get("allowed_turns") or []))
    letters = observed or "".join(t for t in "LTRU" if t in paired.get((region["section_id"], region["region_id"]), ())).lower()
    return f"{letters}{n}" + (" rail" if kind == "shared_rail_motor_lane" else ""), "motor"


def lane_geo(atlas_dir, regions):
    """Traced regions in lon/lat for the map: [section, label, style, ring]; rejected areas are left out."""
    path = atlas_dir / "active_lanes.geojson"
    if not path.exists():
        return []
    shown = {(r["section"], r["region"]): r for r in regions if r["style"] != "rejected"}
    out = []
    for f in read(path).get("features", []):
        p, geometry = f["properties"], f["geometry"]
        r = shown.get((p.get("section_id"), p.get("region_id")))
        if r is None or geometry["type"] != "Polygon":
            continue
        out.append([r["section"], r["label"], r["style"], [[round(x, 6), round(y, 6)] for x, y in geometry["coordinates"][0]]])
    return out


def build_site(site, run, utdf_id, runs, results, out, link_prefix, reviews):
    atlas_dir = next((d for d in (runs / f"control_generic_{run}/atlas", runs / f"geospatial_evidence_{run}") if (d / "atlas_data.json").exists()), None)
    res = results / run
    config, lanes = read(res / "classification/inference_config.json"), read(res / "classification/lanes.json")
    if atlas_dir is None or not config or lanes is None:
        return None
    lanes = with_exit_audits(lanes, read(res / "exits/predictions.json"))  # exits as re-audited, as in the atlas
    atlas = read(atlas_dir / "atlas_data.json")
    movements = read(res / "classification/movements.json") or []
    img = out / "img" / str(utdf_id)
    if img.exists():
        shutil.rmtree(img)
    scale = SAT / atlas["atlas"]["size"][0]
    sat_size = save_image(atlas_dir / atlas["atlas"]["asset"], img / "sat.avif", SAT, QUALITY["sat"])
    by_region = {(s["section_id"], r["region_id"]): (s, r) for s in lanes for r in s["regions"]}
    paired = defaultdict(set)
    for m in movements:
        if m["status"] == "candidate":
            for p in m["lane_pairs"]:
                paired[(m["in_section_id"], p["in_region_id"])].add(TURN[m["turn"]])
    assoc = defaultdict(set)
    for a in atlas.get("associations", []):
        assoc[a["observation_id"]].add(a["region_id"])
    regions = []
    for r in atlas["regions"]:
        section, lane = by_region.get((r["section_id"], r["region_id"]), (None, None))
        if lane is None:
            continue
        inbound = section["kind"] == "inbound_stopbar"
        text, style = region_label(r, lane, inbound, paired)
        use = (lane.get("lane_use_prediction") or {})
        regions.append({"id": r["id"], "section": r["section_id"], "region": r["region_id"], "inbound": inbound, "label": text, "style": style,
                        "type": lane["surface_type"], "existence": lane["existence"], "lane": lane.get("motor_lane_index"),
                        "use": use.get("use"), "turns": use.get("allowed_turns") or [], "confidence": use.get("confidence") or lane.get("confidence"),
                        "evidence": lane.get("evidence", ""), "binding": use.get("binding", ""),
                        "paired": sorted(paired.get((r["section_id"], r["region_id"]), ()), key="LTRU".index),
                        "poly": [[round(x * scale, 1), round(y * scale, 1)] for x, y in r["polygon_atlas_px"]]})
    control_refs = {ref["source_id"] for c in atlas.get("controls", []) for ref in c.get("visual_refs", [])}
    views, index = [], {}
    for v in atlas["views"]:
        domain = v.get("sampling_domain") or "approach"
        if domain == "traffic_control" and v["id"] not in control_refs:
            continue
        name = f"v{len(views)}.avif"
        size = save_image(atlas_dir / v["asset"], img / name, VIEW_MAX, QUALITY["view"])
        if domain == "traffic_control":
            group, role = f"Signals {v.get('direction', '')}", "signal close-up" if v.get("view_role") == "control_detail" else "signal overview"
        elif domain == "outbound":
            group, role = f"Exit {v.get('target_section', '').replace('out_', '')}", "looking along exit" if v.get("view_role") == "away" else "looking back"
        else:
            group = f"Approach {v.get('direction', '')}"
            role = "looking back" if v.get("reverse_view") else "looking ahead"
        index[v["id"]] = len(views)
        views.append({"id": v["id"], "src": name, "size": size, "group": group, "role": role,
                      "position": v.get("sampling_position") or "", "date": v.get("capture_date"), "relation": v.get("time_relation"),
                      "dist": round(v.get("distance_to_center_m") or 0, 1), "heading": round(v.get("compass_heading_deg") or 0),
                      "xy": [round(c * scale, 1) for c in v["atlas_xy"]] if v.get("atlas_xy") else None})
    observations = []
    for o in atlas.get("observations", []):
        if o.get("source_id") in index and o.get("pixel_bbox_normalized"):
            observations.append({"view": index[o["source_id"]], "box": o["pixel_bbox_normalized"], "finding": o.get("finding", ""),
                                 "stage": o.get("stage"), "regions": sorted(assoc.get(o["id"], ()))})
    controls = []
    for c in atlas.get("controls", []):
        controls.append({"kind": c.get("kind"), "dir": c.get("direction"), "symbol": c.get("symbol"), "text": c.get("observed_text"),
                         "description": c.get("description", ""), "confidence": c.get("confidence"), "applies": c.get("applies_to"),
                         "binding": c.get("binding_status"),
                         "refs": [{"view": index[ref["source_id"]], "box": ref.get("bbox_xyxy"), "finding": ref.get("finding", "")}
                                  for ref in c.get("visual_refs", []) if ref["source_id"] in index]})
    figures = {}
    for name, width in (("overview", 1500), ("sections", 1900)):
        source = res / "annotated/labels" / f"{name}.jpg"
        if source.exists():
            figures[name] = {"src": f"{name}.avif", "size": save_image(source, img / f"{name}.avif", width, QUALITY["figure"])}
    data = {"id": utdf_id, "sat": {"src": "sat.avif", "size": sat_size}, "regions": regions, "views": views, "geo": lane_geo(atlas_dir, regions),
            "observations": observations, "controls": controls, "figures": figures,
            "movements": [{"from": m["from_direction"], "turn": TURN[m["turn"]], "to": m["to_direction"], "status": m["status"],
                           "confidence": m.get("confidence"), "pairs": [[p["ib_lane"], p["ob_lane"]] for p in m["lane_pairs"]],
                           "reason": m.get("reason", "")} for m in movements]}
    (out / "sites").mkdir(parents=True, exist_ok=True)
    (out / "sites" / f"{utdf_id}.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return config, lanes, movements


def summary_entry(site, run, utdf_int, mcmp, config, lanes, movements, link_prefix, reviews, results, ctrl_names, out, lonlat=None):
    sections = {s["section_id"]: s for s in lanes}
    checks = {c["approach"]: c for c in (read(results / run / "lane_check/results.json") or [])}
    uses = read(results / run / "classification/lane_use_audits.json") or {}
    sources = config.get("sources", {})
    approaches = []
    claimed = set().union(*[link_prefix.get(str(leg.get("in_link_id")), set()) for leg in config.get("legs", {}).values()])
    for key, leg in config.get("legs", {}).items():
        section = sections.get(f"in_{key}")
        if section is None:
            continue
        prefix, match = utdf_match(link_prefix.get(str(leg.get("in_link_id")), set()), key, mcmp, claimed)
        vl = vlm_lanes(section, movements)
        utdf, osm = (lanes_by_source(mcmp, prefix, "utdf"), lanes_by_source(mcmp, prefix, "osm")) if match else ([], [])
        review = reviews.get(f"{site['node_id']}_{key}")
        check = checks.get(key)
        auto = None
        if check:
            auto = {k: check.get(k) for k in ("verdict", "cause", "layout_seen", "confidence", "note")}
            sheet = results / run / "lane_check" / (check.get("sheet") or "")
            if check.get("sheet") and sheet.exists():
                target = out / "img" / str(utdf_int["id"]) / f"check_{key}.avif"
                if not target.exists() or target.stat().st_mtime < sheet.stat().st_mtime:
                    save_image(sheet, target, 1100, QUALITY["check"])
                auto["sheet"] = target.name
        bikes = sum(r.get("surface_type") == "bicycle_strip" and r.get("existence") == "supported" for r in section["regions"])
        gsv = []
        for c in (uses.get(key) or {}).get("street_view_counts", []):
            src = sources.get(c.get("source_id"), {})
            gsv.append({"pos": src.get("sampling_position") or "", "dist": round(src.get("distance_to_center_m") or 0),
                        "date": src.get("capture_date"), "lanes": c.get("motor_lanes"), "other": c.get("other_lanes", ""),
                        "unmatched": c.get("unmatched", ""), "settled": c.get("settled", True)})
        gsv.sort(key=lambda g: g["dist"])
        approaches.append({"dir": prefix, "key": key, "vlm": vl, "bike": bikes, "utdf": utdf, "osm": osm, "gsv": gsv,
                           "agree": bool(utdf) and len(vl) == len(utdf), "review": review, "auto": auto, "match": match})
    shown = {f"check_{a['key']}.avif" for a in approaches if a["auto"] and a["auto"].get("sheet")}
    for stale in (out / "img" / str(utdf_int["id"])).glob("check_*.avif"):
        if stale.name not in shown:
            stale.unlink()
    approaches.sort(key=lambda a: ORDER.index(a["dir"]) if a["dir"] in ORDER else 99)
    pick = lambda e: {"lanes": e.get("lanes"), "in": [e.get("sib"), e.get("eib")] if e.get("sib") else None,
                      "out": [e.get("sob"), e.get("eob")] if e.get("sob") else None} if e else None

    def row(code, entry, m, group):
        vl = None
        if m:
            ins, outs = sorted({p["ib_lane"] for p in m["lane_pairs"]}), sorted({p["ob_lane"] for p in m["lane_pairs"]})
            vl = {"status": m["status"], "lanes": len(ins), "in": [ins[0], ins[-1]] if ins else None, "out": [outs[0], outs[-1]] if outs else None,
                  "confidence": m.get("confidence"), "reason": m.get("reason", "")[:600]}
        return {"code": code, "group": group, "utdf": pick(entry.get("utdf")), "osm": pick(entry.get("osm")), "vlm": vl,
                "vol": (entry.get("utdf") or {}).get("vol")}

    # Rows grouped by network approach; UTDF/OSM rows join the approach they were matched to, the rest are listed alone.
    rows, used = [], set()
    for a in approaches:
        prefix = a["dir"] if a["match"] else None
        mine = [m for m in movements if m["from_direction"] == a["key"]]
        group = a["dir"] if a["dir"] == a["key"] else f"{a['dir']} (network {a['key']})"
        used.add(prefix)
        for code in sorted({c for c in mcmp if c[:2] == prefix} | {a["dir"] + TURN[m["turn"]] for m in mine}):
            m = next((m for m in mine if TURN[m["turn"]] == turn_of(code) and len(code) == 3), None)
            rows.append(row(code, mcmp.get(code, {}) if prefix else {}, m, group))
    for code in sorted(c for c in mcmp if c[:2] not in used):
        rows.append(row(code, mcmp[code], None, f"{code[:2]} (UTDF/OSM only)"))
    imagery = read(results / run / "imagery/status.json") or {}
    dates = sorted({v.get("capture_date") for v in read(results / run / "imagery/base_config.json").get("sources", {}).values() if v.get("capture_date")}) \
        if (results / run / "imagery/base_config.json").exists() else []
    compared = [a for a in approaches if a["utdf"]]
    if lonlat is None:
        lonlat = (float(NODES_XY[site["node_id"]][0]), float(NODES_XY[site["node_id"]][1])) if site["node_id"] in NODES_XY \
            else (utdf_int.get("lon"), utdf_int.get("lat"))
    return {"id": utdf_int["id"], "utdf": site.get("utdf_intid"), "name": utdf_int["nm"], "node": site["node_id"], "match_note": site.get("match_note"),
            "lat": lonlat[1], "lon": lonlat[0],
            "legs": utdf_int.get("nap"), "control": ctrl_names.get(str(utdf_int.get("ct")), ""), "cycle": utdf_int.get("cyc"),
            "volume": utdf_int.get("vol"), "approaches": approaches, "movements": rows,
            "compared": len(compared), "agree": sum(a["agree"] for a in compared),
            "untraced": [u["id"] for u in config.get("untraced_sections", [])],
            "missing_views": [f"{m['section_id']} {m['position']}" for m in imagery.get("missing", [])],
            "dates": [dates[0], dates[-1]] if dates else None}


def discover(results, node_csv, movement_csv, ints):
    """Exported runs with a finished classification on the network UTDF is matched to, one per UTDF intersection.

    A run on another network is skipped (its node numbers mean other nodes); of two runs of one node the newer is used."""
    node_utdf = utdf_by_node(movement_csv)  # net2cell_utdf's matches with configs/utdf_node_overrides.json applied
    found = {}
    for res in sorted(d for d in results.iterdir() if d.is_dir()):
        site, config = read(res / "imagery/site.json"), res / "classification/inference_config.json"
        if not site or not config.exists():
            continue
        if Path(site.get("network", {}).get("node_csv", "")).resolve() != node_csv.resolve():
            print(f"skip {res.name}: not on {node_csv}", flush=True)
            continue
        utdf_id, note, _ = node_utdf.get(site["node_id"], (None, None, False))
        if utdf_id not in ints:
            print(f"skip {res.name}: node {site['node_id']} has no UTDF intersection", flush=True)
            continue
        stamp = config.stat().st_mtime
        if utdf_id not in found or stamp > found[utdf_id]["stamp"]:
            found[utdf_id] = {"run": res.name, "node_id": site["node_id"], "utdf_intid": utdf_id, "utdf_name": ints[utdf_id]["nm"],
                              "match_note": note, "stamp": stamp}
    return sorted(found.values(), key=lambda s: s["utdf_intid"])


def extra_sites(results, runs, first_id):
    """Exported runs shown without a reference: their node, name, position on their own network and arm count."""
    out = []
    for i, run in enumerate(runs):
        site = read(results / run / "imagery/site.json")
        if not site:
            print(f"skip {run}: no exported imagery/site.json", flush=True)
            continue
        node_csv = Path(site["network"]["node_csv"])
        node_csv = node_csv if node_csv.is_absolute() else ROOT / node_csv
        with open(node_csv, newline="", encoding="utf-8-sig") as stream:
            row = next(r for r in csv.DictReader(stream) if int(r["node_id"]) == int(site["node_id"]))
        sections = read(results / run / "geometry/sections.json") or []
        arms = {s.get("arm") for s in sections if s.get("arm")}
        out.append({"run": run, "node_id": site["node_id"], "utdf_intid": None, "match_note": None, "id": first_id + i,
                    "name": site.get("name") or run, "lonlat": (float(row["x_coord"]), float(row["y_coord"])), "legs": len(arms) or None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--utdf-root", type=Path, default=ROOT.parent / "net2cell_utdf")
    p.add_argument("--sites", type=Path, help="batch list to restrict the page to (and take its title from); default: every exported run")
    p.add_argument("--extra-runs", nargs="*", default=[], help="exported runs without a UTDF intersection, shown with VLM readings only")
    p.add_argument("--no-utdf-sites", action="store_true", help="leave out the UTDF-matched runs (a page of --extra-runs alone)")
    p.add_argument("--title", help="page title (default: the batch list's title)")
    p.add_argument("--review", type=Path, default=results_root() / "reviews/lane_count_review.json")
    p.add_argument("--template", type=Path, default=ROOT / "dashboard/intersection_explorer_template.html")
    p.add_argument("--vendor", type=Path, default=ROOT / "dashboard/vendor", help="folder holding leaflet.js and leaflet.css")
    p.add_argument("--output", type=Path, default=ROOT / "dashboard/intersection_explorer.html")
    p.add_argument("--cache", type=Path, default=runs_root() / "dashboard_build", help="per-intersection parts (outside the synced folder)")
    p.add_argument("--only", nargs="*", help="UTDF intersection IDs or run names to (re)build; the others are taken from the cache. "
                                              "The page always lists every site with data; --only 0 rebuilds the page alone")
    a = p.parse_args()
    utdf_file = a.utdf_root / "build/dashboard/dashboard_data.json"
    with_utdf = utdf_file.exists() and not a.no_utdf_sites
    utdf = json.loads(utdf_file.read_text(encoding="utf-8")) if with_utdf else {"ints": [], "mcmp": {}, "ctrl": {}}
    ints = {i["id"]: i for i in utdf["ints"]}
    link_prefix, reviews, sites = {}, {}, []
    batch = json.loads(a.sites.read_text()) if a.sites else {}
    if with_utdf:
        movement_csv = a.utdf_root / "build/movements/movement_utdf.csv"
        with open(a.utdf_root / "build/network/macronet/node.csv", newline="", encoding="utf-8-sig") as stream:
            NODES_XY.update({int(r["node_id"]): (r["x_coord"], r["y_coord"]) for r in csv.DictReader(stream)})
        link_prefix = utdf_movements(movement_csv)
        reviews = (read(a.review) or {}).get("reviews", {})
        wanted = {s["node_id"] for s in batch.get("sites", [])}
        sites = [s for s in discover(results_root(), a.utdf_root / "build/network/macronet/node.csv", movement_csv, ints)
                 if not wanted or s["node_id"] in wanted]
    extras = extra_sites(results_root(), a.extra_runs, EXTRA_ID)
    cache = a.cache
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for site in sites + extras:
        run = site["run"]
        key = site["utdf_intid"] if site["utdf_intid"] is not None else site["id"]
        site_reviews = reviews if site["utdf_intid"] is not None else {}  # reviews are keyed by Tempe node numbers
        if a.only is not None and not {str(key), run} & set(a.only) and (cache / "sites" / f"{key}.json").exists():
            built = (read(results_root() / run / "classification/inference_config.json"),
                     with_exit_audits(read(results_root() / run / "classification/lanes.json") or [], read(results_root() / run / "exits/predictions.json")),
                     read(results_root() / run / "classification/movements.json") or [])
        else:
            built = build_site(site, run, key, runs_root(), results_root(), cache, link_prefix, site_reviews)
        if not built or not built[0]:
            continue
        if site["utdf_intid"] is not None:
            entries.append(summary_entry(site, run, ints[key], utdf["mcmp"].get(str(key), {}), *built, link_prefix, site_reviews,
                                         results_root(), utdf.get("ctrl", {}), cache))
            print(key, site["utdf_name"], run, flush=True)
        else:
            own = {"id": key, "nm": site["name"], "nap": site["legs"]}
            entries.append(summary_entry(site, run, own, {}, *built, {}, {}, results_root(), {}, cache, lonlat=site["lonlat"]))
            print(key, site["name"], run, "(no reference)", flush=True)
    verdicts, auto = defaultdict(int), defaultdict(int)
    for e in entries:
        for a_ in e["approaches"]:
            if a_["utdf"] and not a_["agree"]:
                for counts, check in ((verdicts, a_.get("review")), (auto, a_.get("auto"))):
                    if check:
                        counts[check["verdict"]] += 1
    # Each intersection becomes one JSON block with its images inline; its lanes in lon/lat go to the index for the map.
    geo, blocks = {}, []
    for e in entries:
        data = json.loads((cache / "sites" / f"{e['id']}.json").read_text(encoding="utf-8"))
        geo[e["id"]] = data.pop("geo", [])
        data["img"] = {f.name: "data:image/avif;base64," + base64.b64encode(f.read_bytes()).decode()
                       for f in sorted((cache / "img" / str(e["id"])).glob("*.avif"))}
        blocks.append(f'<script type="application/json" id="site-{e["id"]}">{inline_json(data)}</script>')
    index = {"title": a.title or batch.get("title") or "VLM-analysed intersections", "network": batch.get("network"), "area": batch.get("bbox_source"),
             "sites": entries, "reviews": dict(verdicts), "auto_checks": dict(auto), "geo": geo,
             "approaches": sum(e["compared"] for e in entries), "agree": sum(e["agree"] for e in entries)}
    page = a.template.read_text(encoding="utf-8")
    parts = {"/*__INDEX__*/": inline_json(index), "<!--__SITES__-->": "\n".join(blocks),
             "/*__LEAFLET_CSS__*/": (a.vendor / "leaflet.css").read_text(encoding="utf-8"),
             "/*__LEAFLET_JS__*/": (a.vendor / "leaflet.js").read_text(encoding="utf-8").replace("</script", "<\\/script")}
    for key, value in parts.items():
        if page.count(key) != 1:
            raise SystemExit(f"template must contain one {key} placeholder")
        page = page.replace(key, value)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(page, encoding="utf-8")
    print({"intersections": len(entries), "output": str(a.output), "mb": round(a.output.stat().st_size / 1e6, 1)})


def inline_json(value):
    """JSON safe to place inside a <script> element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


if __name__ == "__main__":
    main()
