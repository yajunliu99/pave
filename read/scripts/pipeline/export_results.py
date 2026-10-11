"""Copy a site's compact results out of its run folders into the results folder.

Run folders hold every intermediate image and can live outside the synced project; what is exported is
small: configs, traced sections, lane and movement predictions, the exit audit, the final atlas data,
the signal audit, reports, every raw model answer, the source images (satellite and each street view,
named by section and position) and the annotated figures (lane labels, traced regions, crop plan,
movements; PNG renders saved as JPEG). Answers
keep the run layout (answers/<stage>/vlm/<stage name>/<signature>.raw.json), so an answers folder can be
passed to --reuse-cache-from to replay or extend the site without new model calls.
"""
import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.hybrid.common import read_json, resource_path, write_json
from movement_fixer.paths import results_root, runs_root

FILES = {
    "imagery": ("imagery_{name}", ["site.json", "base_config.json", "exit_views.json", "acquisition.json", "status.json"]),
    "geometry": ("geometry_loop_{name}", ["geometry/sections.json", "geometry/summary.json", "status.json"]),
    "classification": ("hybrid_pipeline_{name}", ["inference_config.json", "lanes.json", "movements.json", "lane_use_audits.json",
                                                  "surface_predictions.json", "gsv_context.json", "gsv_sampling.json", "summary.json",
                                                  "report.md", "lane_predictions.csv", "movement_candidates.csv", "status.json"]),
    "exits": ("downstream_audit_{name}", ["predictions.json", "summary.json", "status.json"]),
    "controls": ("control_generic_{name}/audit", ["predictions.json", "summary.json", "status.json"]),
    "lane_check": ("lane_check_{name}", ["results.json", "summary.json", "status.json"]),
}
ATLAS = ["atlas_data.json", "atlas.geojson", "active_lanes.geojson", "rejected_regions.geojson", "georeference.json", "report.md",
         "control_provenance.json"]
ANSWERS = {"geometry": "geometry_loop_{name}", "classification": "hybrid_pipeline_{name}", "exits": "downstream_audit_{name}",
           "controls": "control_generic_{name}/audit", "alignment": "control_generic_{name}/gsv_alignment",
           "lane_check": "lane_check_{name}"}


def as_jpeg(source, target, quality=88):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(target, quality=quality)


def source_images(imagery, out):
    """The satellite image and every street view as acquired, named by section, position and view."""
    copied = []
    if not (imagery / "base_config.json").exists():
        return copied
    base = read_json(imagery / "base_config.json")
    views = [(sid, s) for sid, s in base["sources"].items() if s.get("path")]
    if (imagery / "exit_views.json").exists():
        views += [(v["id"], v) for v in read_json(imagery / "exit_views.json")["views"]]
    for sid, source in views:
        path = resource_path(ROOT, source["path"])
        if path.exists():
            folder = out if sid == "satellite" else out / "street_views"
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, folder / f"{sid}{path.suffix.lower()}")
            copied.append(str((folder / f"{sid}{path.suffix.lower()}").relative_to(out.parent)))
    return copied


def annotated_images(runs, name, out):
    """Figures drawn on the imagery: lane labels, traced and supported regions, crop plan, movements, street-view alignment."""
    evidence = runs / f"hybrid_pipeline_{name}"
    figures = [(evidence / "labels" / p.name, f"labels/{p.stem}.jpg") for p in sorted((evidence / "labels").glob("*.png"))]
    figures += [(evidence / f, f"{Path(f).stem}.jpg") for f in ("candidate_geometry.png", "model_geometry.png")]
    figures += [(p, f"movements/{p.stem}.jpg") for p in sorted(evidence.glob("movements_*.png"))]
    figures += [(runs / f"geometry_loop_{name}/geometry/plan.png", "tracing_plan.jpg")]
    figures += [(p, f"gsv_alignment/{p.parent.name}.jpg") for p in sorted((runs / f"control_generic_{name}/gsv_alignment").glob("*/aligned.jpg"))]
    copied = []
    for source, target in figures:
        if source.exists():
            as_jpeg(source, out / target)
            copied.append(f"annotated/{target}")
    return copied


def export_site(name, runs=None, results=None):
    runs, results = Path(runs or runs_root()), Path(results or results_root())
    out = results / name
    copied = []
    for part, (folder, files) in FILES.items():
        for f in files:
            source = runs / folder.format(name=name) / f
            if source.exists():
                target = out / part / Path(f).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target.relative_to(out)))
    atlas = runs / f"control_generic_{name}/atlas"
    atlas = atlas if (atlas / "atlas_data.json").exists() else runs / f"geospatial_evidence_{name}"
    for f in ATLAS:
        if (atlas / f).exists():
            (out / "atlas").mkdir(parents=True, exist_ok=True)
            shutil.copy2(atlas / f, out / "atlas" / f)
            copied.append(f"atlas/{f}")
    sheets = sorted((runs / f"lane_check_{name}" / "sheets").glob("*.jpg"))
    for stale in set((out / "lane_check" / "sheets").glob("*.jpg")) - {out / "lane_check" / "sheets" / s.name for s in sheets}:
        stale.unlink()  # the check no longer covers that approach
    for sheet in sheets:
        (out / "lane_check" / "sheets").mkdir(parents=True, exist_ok=True)
        shutil.copy2(sheet, out / "lane_check" / "sheets" / sheet.name)
        copied.append(f"lane_check/sheets/{sheet.name}")
    copied += source_images(runs / f"imagery_{name}", out / "images")
    copied += annotated_images(runs, name, out / "annotated")
    answers = 0
    for part, folder in ANSWERS.items():
        for raw in (runs / folder.format(name=name) / "vlm").glob("*/*.raw.json"):
            target = out / "answers" / part / "vlm" / raw.parent.name / raw.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw, target)
            answers += 1
    write_json(out / "export.json", {"run_folders": str(runs), "dashboard": "dashboard/intersection_explorer.html",
                                     "files": copied, "raw_model_answers": answers,
                                     "replay": "pass answers/<stage> folders to --reuse-cache-from to rerun without new model calls"})
    return {"site": name, "files": len(copied), "answers": answers, "output": str(out)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+", help="run names, e.g. tempe_7750")
    p.add_argument("--runs", type=Path, help="default: the configured runs root")
    p.add_argument("--results", type=Path, help="default: the configured results root")
    a = p.parse_args()
    for name in a.names:
        print(export_site(name, a.runs, a.results), flush=True)


if __name__ == "__main__":
    main()
