"""One entry point for a junction: imagery -> trace regions -> classify -> exit re-audit -> optional lane-count
check -> atlas -> controls -> labels -> optional street-view lane alignment -> export -> combined dashboard.

The stages are the ones behind dashboard/dashboard.html, preceded by acquisition and automatic tracing. The
results are exported to the results folder and the site is added to (or refreshed in) dashboard/intersection_explorer.html.
Start from a network node (--node-id with --node-csv/--link-csv), or from existing inputs (--site,
--base-config, --exit-views). A stage whose run directory is already complete is skipped, so the command
can be repeated after an interruption without repeating requests or model calls.

Billed requests need explicit flags: --acquire for the satellite and street-view images of a node, and
--audit-controls for the signal and sign audit. --plan-only lists the stages and the acquisition budget
without contacting any service.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.hybrid.common import read_json, resource_path
from movement_fixer.paths import results_root, runs_root

DASHBOARD = ROOT / "dashboard" / "intersection_explorer.html"

CALLS = {"lane_check": "1 per approach whose lane count differs from the reference",
         "street_check": "1-2 review rounds per approach whose street-view lane count differs from the traced lanes; then classification again",
         "imagery": "1 satellite image + 3 street views and 1 look-back per approach + 6 per exit (billed); metadata is free",
         "geometry": "draft + up to 3 reviews per section, capped by the site policy",
         "evidence": "street-view contexts + surface + lane-use + movement per approach, capped by --max-evidence-calls",
         "exits": "1 per exit section",
         "controls": "street-view metadata/images (billed) and up to 24 model calls",
         "gsv_alignment": "readings, lane-offset choice and strip check per readable street view, capped by --max-align-calls"}


def fresh(output, *inputs):
    """An output file that exists and is no older than any of its inputs needs no rebuilding."""
    return output.exists() and all(not i.exists() or output.stat().st_mtime >= i.stat().st_mtime for i in inputs)


def done(skip):
    """A stage's skip flag; a callable is evaluated when the stage is reached."""
    return skip() if callable(skip) else skip


def complete(folder):
    status = folder / "status.json"
    return status.exists() and read_json(status).get("state") == "complete"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="suffix of the run directories, e.g. 351_20261002_autogeometry")
    p.add_argument("--node-id", type=int, help="network node to start from; imagery and configs are generated")
    p.add_argument("--node-csv", type=Path)
    p.add_argument("--link-csv", type=Path)
    p.add_argument("--acquisition-policy", type=Path, help="JSON overrides of the acquisition policy")
    p.add_argument("--control-policy", type=Path, help="JSON overrides of the signal and sign audit policy (e.g. a preferred date window)")
    p.add_argument("--acquire", action="store_true", help="Allow billed satellite and street-view requests for --node-id")
    p.add_argument("--site", type=Path, help="autoloop site config (satellite, georeference, network)")
    p.add_argument("--base-config", type=Path, help="evidence config supplying street-view sources and sampling")
    p.add_argument("--exit-views", type=Path, help="manifest of exit views looking away from and back at the junction")
    p.add_argument("--control-audit", type=Path, help="finished signal/sign audit to attach")
    p.add_argument("--reference-movements", type=Path,
                   help="reference movements CSV with lane ranges (e.g. net2cell_utdf movement_utdf.csv); enables the blind lane-count check")
    p.add_argument("--audit-controls", action="store_true", help="Run the signal/sign audit (billed street views, model calls)")
    p.add_argument("--geometry-run", type=Path, help="use this finished geometry run instead of <runs root>/geometry_loop_<name>")
    p.add_argument("--retrace", action="store_true",
                   help="If tracing left sections untraced, set this site's run folders aside (renamed, not deleted) and run again, "
                        "reusing every cached answer from them")
    p.add_argument("--refresh-outdated", action="store_true",
                   help="If the tracing run was made by an older tracing version, set the tracing, classification, exit-audit, "
                        "lane-check and lane-atlas folders aside (renamed, not deleted) and run them again, reusing every cached "
                        "answer from them; the signal and sign audit is kept and attached again")
    p.add_argument("--reuse-cache-from", type=Path, action="append", default=[])
    p.add_argument("--max-evidence-calls", type=int, help="default: scaled to the site's sections and street views")
    p.add_argument("--align-gsv", action="store_true", help="Also align the projected lanes with every street view (2-3 model calls per view)")
    p.add_argument("--max-align-calls", type=int, default=60)
    p.add_argument("--cache-only", action="store_true", help="Disallow new requests and model calls in every stage")
    p.add_argument("--skip-dashboard", action="store_true", help="Leave the combined dashboard alone (a batch rebuilds it once at the end)")
    p.add_argument("--plan-only", action="store_true", help="List the stages that would run; contact no service")
    a = p.parse_args()
    exp = runs_root()
    exp.mkdir(parents=True, exist_ok=True)
    imagery = exp / f"imagery_{a.name}"
    if a.node_id is not None:
        if not (a.node_csv and a.link_csv):
            p.error("--node-id needs --node-csv and --link-csv")
        site, base, exit_views = imagery / "site.json", imagery / "base_config.json", imagery / "exit_views.json"
    else:
        if not (a.site and a.base_config and a.exit_views):
            p.error("Give --node-id, or all of --site, --base-config and --exit-views")
        site, base, exit_views = a.site.resolve(), a.base_config.resolve(), a.exit_views.resolve()
    geometry = (a.geometry_run or exp / f"geometry_loop_{a.name}").resolve()
    evidence, exits = exp / f"hybrid_pipeline_{a.name}", exp / f"downstream_audit_{a.name}"
    atlas_base, controls = exp / f"geospatial_evidence_{a.name}", exp / f"control_generic_{a.name}"
    config = ROOT / "configs" / "pilots" / f"site_{a.name}.json"
    if a.retrace and not a.plan_only and untraced(geometry):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for folder in (geometry, evidence, exits, atlas_base, controls):
            if folder.exists():
                aside = folder.with_name(f"{folder.name}_superseded_{stamp}")
                folder.rename(aside)
                a.reuse_cache_from.append(aside)
                print(f"set aside {folder.name} -> {aside.name}", flush=True)
    if a.refresh_outdated and not a.plan_only and outdated(geometry):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for folder in (geometry, evidence, exits, atlas_base, exp / f"lane_check_{a.name}"):
            if folder.exists():
                aside = folder.with_name(f"{folder.name}_superseded_{stamp}")
                folder.rename(aside)
                a.reuse_cache_from.append(aside)
                print(f"set aside {folder.name} -> {aside.name}", flush=True)
    # A finished audit stays attached on later runs, with or without --audit-controls.
    with_controls = bool(a.control_audit or a.audit_controls or complete(controls / "audit"))
    final = controls / "atlas" if with_controls else atlas_base
    cache = ["--cache-only"] if a.cache_only else []
    reuse = [x for path in a.reuse_cache_from for x in ("--reuse-cache-from", path.resolve())]
    acquired = a.node_id is None or complete(imagery)
    stages = [("imagery", acquired, "pipeline/acquire_site_imagery.py",
               ["--node-id", a.node_id, "--node-csv", a.node_csv and a.node_csv.resolve(), "--link-csv", a.link_csv and a.link_csv.resolve(),
                "--output", imagery, *(["--policy", a.acquisition_policy.resolve()] if a.acquisition_policy else []), *cache])]
    later = lambda: site_stages(a, site, base, exit_views, geometry, evidence, exits, atlas_base, controls, final, config, cache, reuse)
    if acquired:  # the site's configs exist; otherwise its stages are defined once imagery has run
        stages += later()
    plan = [{"stage": name, "action": "skip (complete)" if done(skip) else "run", "model_calls": CALLS.get(name, "none")} for name, skip, _, _ in stages]
    plan += [{"stage": "export", "action": "run", "model_calls": "none", "into": str(results_root() / a.name)},
             {"stage": "dashboard", "action": "update this site", "model_calls": "none", "page": str(DASHBOARD)}]
    print(json.dumps({"runs": {"imagery": str(imagery) if a.node_id is not None else None, "geometry": str(geometry), "evidence": str(evidence),
                               "exits": str(exits), "atlas": str(final)}, "stages": plan}, ensure_ascii=False, indent=2), flush=True)
    if a.plan_only:
        if not acquired:  # show the request budget of the node; nothing is fetched
            run("pipeline/acquire_site_imagery.py", [*stages[0][3], "--plan-only"])
            print("\nLater stages are listed once imagery for the node exists.")
        return 0
    if not acquired and not (a.acquire or a.cache_only):
        run("pipeline/acquire_site_imagery.py", [*stages[0][3], "--plan-only"])
        raise SystemExit("Imagery for this node is not acquired. Rerun with --acquire to make the billed requests listed above.")
    if a.audit_controls and not a.control_audit and not complete(controls / "audit") and not a.cache_only and not a.acquire:
        raise SystemExit("--audit-controls makes billed street-view requests; add --acquire to allow them.")
    index = 0
    while index < len(stages):
        name, skip, script, args = stages[index]
        index += 1
        if done(skip):  # judged now, so a stage after one that just reran sees the new outputs
            continue
        print(f"\n>> {name}", flush=True)
        if name == "atlas" and atlas_base.exists():
            # The atlas freezes the classification code it was built with; a rebuild after a code change goes to a
            # fresh folder. It holds no model answers, so nothing is lost by setting the old one aside.
            aside = atlas_base.with_name(f"{atlas_base.name}_superseded_{time.strftime('%Y%m%d_%H%M%S')}")
            atlas_base.rename(aside)
            print(f"set aside {atlas_base.name} -> {aside.name}", flush=True)
        run(script, args)
        if name == "imagery":
            stages += later()
        if name == "street_check" and read_json(geometry / "geometry" / "street_check.json").get("classification_outdated"):
            # classification and everything built on it run again on the revised tracing, reusing their cached answers
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for folder in (evidence, exits, exp / f"lane_check_{a.name}"):
                if folder.exists():
                    aside = folder.with_name(f"{folder.name}_superseded_{stamp}")
                    folder.rename(aside)
                    a.reuse_cache_from.append(aside)
                    if folder == evidence:
                        a.freeze_width_from = aside / "geometry_hints.json"
                    print(f"tracing revised after the street-view check: set aside {folder.name} -> {aside.name}", flush=True)
            reuse[:] = [x for path in a.reuse_cache_from for x in ("--reuse-cache-from", path.resolve())]
            again = {s[0]: s for s in later()}
            stages[index:index] = [again[k] for k in ("config", "evidence", "exits", "lane_check")]
    print("\n>> export", flush=True)
    run("pipeline/export_results.py", [a.name, "--runs", exp])
    if not a.skip_dashboard:
        print("\n>> dashboard", flush=True)
        run("pipeline/build_combined_dashboard.py", ["--output", DASHBOARD, "--only", a.name])
    return 0


def run(script, args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], check=True, cwd=ROOT)


def outdated(geometry):
    from movement_fixer.autoloop.geometry_stage import VERSION
    manifest = geometry / "input_manifest.json"
    return manifest.exists() and read_json(manifest).get("version") != VERSION


def untraced(geometry):
    sections = geometry / "geometry" / "sections.json"
    return sections.exists() and any(not s["regions"] for s in read_json(sections))


def retrace(geometry, evidence):
    """A finished tracing run with failed drafts is resumed (new validation may now accept them, and their cached
    answers make a repeat cheap) as long as classification has not started on its result."""
    sections = geometry / "geometry" / "sections.json"
    return (complete(geometry) and sections.exists() and not (evidence / "input_manifest.json").exists()
            and any(s.get("review", {}).get("stop_reason") == "draft_failed" for s in read_json(sections)))


def evidence_budget(base, site):
    """One call per street-view context and per section, two per approach for lane use (with its detail
    follow-up) and one for its movements, plus a quarter for JSON repairs."""
    approaches = len({v["direction"] for v in base["context_views"]})
    sections = site["policy"]["max_vlm_calls"] // 6 if site else 2 * approaches
    return int(1.25 * (len(base["context_views"]) + sections + 3 * approaches)) + 1


def site_stages(a, site, base, exit_views, geometry, evidence, exits, atlas_base, controls, final, config, cache, reuse):
    site_config = read_json(site)
    legacy = site_config.get("legacy_gsv_records") or {}
    legacy_args = [x for key, flag in (("manifest", "--legacy-gsv-manifest"), ("pano_yaw", "--legacy-pano-yaw")) if legacy.get(key)
                   for x in (flag, resource_path(ROOT, legacy[key]))]
    node_csv = resource_path(ROOT, site_config["network"]["node_csv"])
    if a.control_audit:
        control = ("controls", False, "pipeline/attach_control_evidence.py", ["--base", atlas_base, "--audit", a.control_audit.resolve(), "--output", final])
    elif complete(controls / "audit"):
        # The audit itself is reused; its findings are attached again whenever the lane atlas was rebuilt.
        control = ("controls", lambda: fresh(final / "atlas_data.json", atlas_base / "atlas_data.json"), "pipeline/attach_control_evidence.py",
                   ["--base", atlas_base, "--audit", controls / "audit", "--output", final])
    else:
        control = ("controls", not a.audit_controls or complete(controls / "audit"), "pipeline/run_control_pipeline.py",
                   ["--node-id", site_config["node_id"], "--node-csv", node_csv, "--link-csv", resource_path(ROOT, site_config["network"]["link_csv"]),
                    "--baseline", evidence, "--output", controls, "--atlas-base", atlas_base, *cache,
                    *(["--policy", a.control_policy.resolve()] if a.control_policy else [])])
    return [
        ("geometry", complete(geometry) and not retrace(geometry, evidence), "pipeline/run_geometry_loop.py",
         ["--config", site, "--output", geometry, *cache, *reuse]),
        # Once the evidence run is complete its config is frozen in that run's manifest and is not rebuilt.
        ("config", complete(evidence), "pipeline/build_evidence_config_from_geometry.py",
         ["--geometry-run", geometry, "--base", base, "--output", config, "--pilot-id", f"site_{a.name}",
          "--max-api-calls", a.max_evidence_calls or evidence_budget(read_json(base), read_json(site) if site.exists() else None),
          *(["--freeze-width-reference", a.freeze_width_from] if getattr(a, "freeze_width_from", None) else [])]),
        ("evidence", complete(evidence), "pipeline/run_hybrid_pipeline.py", ["--config", config, "--output", evidence, *cache, *reuse]),
        # Approaches whose street-view lane count differs from the traced lanes go back to the tracing review once;
        # if a section changes, classification runs again (see the main loop).
        ("street_check", lambda: not complete(evidence) or (geometry / "geometry" / "street_check.json").exists(), "pipeline/street_check.py",
         ["--site", site, "--geometry", geometry, "--evidence", evidence, *cache, *reuse]),
        # Judged when reached: after a street-check rerun the re-inserted copies of these stages have already run.
        ("exits", lambda: complete(exits), "pipeline/audit_downstream.py",
         ["--baseline", evidence, "--views", exit_views, "--output", exits, *cache, *reuse]),
        # One blind model reading per approach whose lane count differs from the reference; recorded beside the run.
        ("lane_check", lambda: not a.reference_movements or complete(evidence.parent / f"lane_check_{a.name}"), "pipeline/check_lane_counts.py",
         ["--evidence", evidence, "--reference", a.reference_movements.resolve() if a.reference_movements else "",
          "--output", evidence.parent / f"lane_check_{a.name}", *cache, *reuse]),
        ("atlas", lambda: fresh(atlas_base / "atlas_data.json", evidence / "status.json", exits / "status.json"), "pipeline/build_lane_evidence_atlas.py",
         ["--baseline", evidence, "--output", atlas_base, "--downstream-views", exit_views, "--downstream-audit", exits / "predictions.json", *legacy_args]),
        control,
        ("labels", lambda: fresh(evidence / "labels" / "overview.png", evidence / "status.json", exits / "predictions.json"),
         "experiments/draw_lane_labels.py", ["--run", evidence, "--exits", exits / "predictions.json"]),
        ("gsv_alignment", not a.align_gsv, "pipeline/align_gsv_views.py",
         ["--atlas", final, "--run", evidence, "--output", final.parent / "gsv_alignment", "--max-api-calls", a.max_align_calls, *cache, *reuse]),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
