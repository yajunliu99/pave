"""Approach lane counts before and after a refresh of the tracing and classification, against UTDF and the manual check.

"Before" is each site's earliest set-aside classification run (<runs root>/hybrid_pipeline_<name>_superseded_*), or
with --before previous the latest set-aside run made by another classification version (runs set aside by the street
check share the current version and are passed over); "after" is the current run. The manual check (results/reviews/lane_count_review.json) was made on the before
imagery reading: a verdict "utdf" means the imagery shows UTDF's count, "model" the count the VLM had then.
Those implied counts are used to say whether a change fixed or broke an approach. Reads only run outputs.
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.lane_check import reference_lanes, reference_name
from movement_fixer.paths import results_root, runs_root


def counts(run):
    return {s["section_id"][3:]: s["count"]["model_motor_count"] for s in json.loads((run / "lanes.json").read_text())
            if s["section_id"].startswith("in_")}


def version(run):
    manifest = run / "input_manifest.json"
    return json.loads(manifest.read_text()).get("version") if manifest.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", type=Path, default=ROOT / "configs/batches/asu_utdf_sites.json")
    p.add_argument("--prefix", default="tempe")
    p.add_argument("--reference", type=Path, default=ROOT.parent / "net2cell_utdf/build/movements/movement_utdf.csv")
    p.add_argument("--manual", type=Path, default=results_root() / "reviews/lane_count_review.json")
    p.add_argument("--before", choices=("earliest", "previous"), default="earliest")
    p.add_argument("--output", type=Path, default=results_root() / "reviews/refresh_comparison.json")
    a = p.parse_args()
    ref = reference_lanes(a.reference)
    manual = json.loads(a.manual.read_text())["reviews"]
    rows = []
    for site in json.loads(a.sites.read_text())["sites"]:
        name, runs = f"{a.prefix}_{site['node_id']}", runs_root()
        before_runs = sorted(glob.glob(str(runs / f"hybrid_pipeline_{name}_superseded_*")))
        after = runs / f"hybrid_pipeline_{name}"
        if not before_runs or not (after / "lanes.json").exists():
            continue
        reviewed = counts(Path(before_runs[0]))  # the reading the manual check was made on
        if a.before == "previous":
            before_runs = [r for r in before_runs if version(Path(r)) != version(after)][-1:]
            if not before_runs:
                continue
        before, now = counts(Path(before_runs[0])), counts(after)
        legs = json.loads((after / "inference_config.json").read_text())["legs"]
        for key in sorted(set(before) | set(now)):
            named = ref.get(str(legs.get(key, {}).get("in_link_id")), {})
            utdf = named.get(reference_name(named, key))
            m = manual.get(f"{site['node_id']}_{key}", {})
            truth = len(utdf) if m.get("verdict") == "utdf" and utdf else reviewed.get(key) if m.get("verdict") == "model" else None
            b, n = before.get(key), now.get(key)
            outcome = ("unchecked" if truth is None else "still right" if b == truth and n == truth else "fixed" if n == truth
                       else "broke" if b == truth else "still wrong")
            rows.append({"approach": f"{site['node_id']}_{key}", "before": b, "after": n, "utdf": len(utdf) if utdf else None,
                         "manual": m.get("verdict"), "manual_confidence": m.get("confidence"), "outcome": outcome,
                         "changed": b != n})
    with_utdf = [r for r in rows if r["utdf"]]
    summary = {"before": a.before, "approaches": len(rows), "changed": sum(r["changed"] for r in rows),
               "agree_with_utdf_before": sum(r["before"] == r["utdf"] for r in with_utdf),
               "agree_with_utdf_after": sum(r["after"] == r["utdf"] for r in with_utdf), "compared_with_utdf": len(with_utdf),
               "against_manual_check": dict(Counter(r["outcome"] for r in rows if r["outcome"] != "unchecked"))}
    a.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    for r in rows:
        if r["changed"] or r["outcome"] in ("broke", "still wrong"):
            print(f"{r['approach']:>12}  {r['before']} -> {r['after']}  UTDF {r['utdf']}  manual {r['manual'] or '-':7} {r['outcome']}")
    print(summary)


if __name__ == "__main__":
    main()
