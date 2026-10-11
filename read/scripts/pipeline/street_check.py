"""Send approaches whose lane count disagrees with other evidence back to the tracing review loop (disagreement review).

Reads a finished classification run (lanes.json, lane_use_audits.json with street_view_counts) and the tracing
run it was built on. Two kinds of disagreement (autoloop/street_check.py) are found: the street views' lane count,
and, with --readings, an independent reading of the approach (read_approaches.py). Each section with either goes
through its review loop once more, resumed from its final strips, with every disagreement as an issue and the street
view shown with the boundaries projected onto it. No reference count (UTDF) is used. Revised sections replace the old
ones in the tracing run (the old strips are kept in its history); geometry/street_check.json records the outcome and
whether classification has to run again. Runs once per tracing run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop import street_panel
from movement_fixer.autoloop.geometry_stage import load_georeference, plan_site, run_section
from movement_fixer.autoloop.street_check import issue, mismatches, reading_issue, reading_mismatches
from movement_fixer.autoloop.strips import Frame
from movement_fixer.hybrid.common import read_json, write_json
from movement_fixer.hybrid.inference import StageClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", type=Path, required=True, help="tracing site config (site.json)")
    p.add_argument("--geometry", type=Path, required=True, help="tracing run")
    p.add_argument("--evidence", type=Path, required=True, help="classification run built on it")
    p.add_argument("--readings", type=Path, help="results.json of read_approaches.py for this classification run")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--max-api-calls", type=int, default=16)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--reuse-cache-from", type=Path, action="append", default=[])
    a = p.parse_args()
    geometry, evidence = a.geometry.resolve(), a.evidence.resolve()
    g = geometry / "geometry"
    config = read_json(a.site)
    image, _, policy, _ = plan_site(config, ROOT)
    policy = {**policy, "max_rounds": a.max_rounds}
    to_pixel, _ = load_georeference(config["satellite"]["georeference"], ROOT)
    base = a.site.with_name("base_config.json")
    street_views = read_json(base) if base.exists() else None
    lanes = read_json(evidence / "lanes.json")
    found = {}  # section -> disagreements, street view first
    for m in mismatches(lanes, read_json(evidence / "lane_use_audits.json"), read_json(evidence / "inference_config.json")):
        found.setdefault(m["section_id"], []).append({"source": "street_view", **m, "issue": issue(m)})
    for m in reading_mismatches(lanes, read_json(a.readings) if a.readings else []):
        found.setdefault(m["section_id"], []).append({"source": "independent_reading", **m, "issue": reading_issue(m)})
    history = read_json(g / "history.json")
    sections = read_json(g / "sections.json")
    plan = {s["id"]: s for s in read_json(g / "plan.json")["sections"]}
    by_history = {h["section_id"]: i for i, h in enumerate(history)}
    by_section = {s["id"]: i for i, s in enumerate(sections)}
    client = StageClient(ROOT, geometry, {**config.get("models", {}).get("review", config.get("models", {}).get("draft", {})),
                                          "max_api_calls": a.max_api_calls}, a.cache_only, [x.resolve() for x in a.reuse_cache_from])
    results, changed = [], False
    for sid, disagreements in found.items():
        m, issues = disagreements[0], [d.pop("issue") for d in disagreements]
        base_record = {"section_id": sid, "direction": m["direction"], "traced_motor_lanes": m["traced_motor_lanes"],
                       "street_view_lanes": next((d["street_view_lanes"] for d in disagreements if d["source"] == "street_view"), None),
                       "reading_lanes": next((d["reading_lanes"] for d in disagreements if d["source"] == "independent_reading"), None),
                       "disagreements": disagreements}
        record = history[by_history[sid]] if sid in by_history else None
        if not record or not record.get("final") or sid not in by_section:
            results.append({**base_record, "outcome": "no traced strips to review"})
            continue
        spec = {**plan[sid], "frame": Frame(tuple(record["frame"]["center"]), record["frame"]["heading_deg"], record["frame"]["width"],
                                            record["frame"]["height"]), "stations": record["stations"]}
        view = street_panel.pick_view(street_views, m["direction"]) if street_views else None
        street = {"view": view, "image": street_panel.load(view, ROOT), "to_pixel": to_pixel} if view else None
        section, rerun = run_section(image, spec, client, client, g, policy, lambda: client.calls < a.max_api_calls, street,
                                     start=record["final"], extra_issues=issues)
        outcome = {**base_record, "outcome": rerun["status"], "stop_reason": rerun["stop_reason"], "revised": rerun["revised"],
                   "rounds": rerun["rounds"], "regions_before": len(record["final"]["edges"]) - 1, "regions_after": len(rerun["final"]["edges"]) - 1}
        results.append(outcome)
        if rerun["revised"]:
            changed = True
            old = sections[by_section[sid]]
            section.update({k: old[k] for k in ("window_anchor", "street_view", "contrast_stretched", "shade_panel") if k in old})
            sections[by_section[sid]] = section
            history[by_history[sid]] = {**record, "final": rerun["final"], "status": rerun["status"], "street_check": {
                "issues": issues, "rounds": rerun["rounds"], "strips_before": record["final"]}}
    if changed:
        write_json(g / "sections.json", sections)
        write_json(g / "history.json", history)
    summary = {"flagged": len(found), "revised": sum(bool(r.get("revised")) for r in results), "classification_outdated": changed,
               "vlm_calls_this_run": client.calls, "vlm_cache_hits_this_run": client.hits, "results": results,
               "readings": str(a.readings) if a.readings else None,
               "scope": "Street-view counts and independent readings are model readings; a disagreement prompts a review, it never "
                        "overwrites the geometry. No reference count is used."}
    write_json(g / "street_check.json", summary)
    print(json.dumps({k: summary[k] for k in ("flagged", "revised", "classification_outdated", "vlm_calls_this_run")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
