"""Blind check of the approaches whose VLM lane count differs from the reference (e.g. UTDF).

One model call per such approach: it reads the stop-bar lane layout from the unmarked satellite crop and the
forward street views without being told either answer; code then compares the reading with both. Results are
written to --output and never change the classification run.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.lane_check import (blind_sheet, check_prompt, disagreements, layout, reference_lanes, street_views,
                                              validate_reading, verdict)
from movement_fixer.hybrid.common import ValidationError, read_json, write_json
from movement_fixer.hybrid.inference import StageClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True, help="finished classification run")
    p.add_argument("--reference", type=Path, required=True, help="reference movements CSV with lane ranges (movement_utdf.csv)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--approaches", nargs="*", help="check only these approach keys (e.g. EB link_15203); default every disagreement")
    p.add_argument("--max-api-calls", type=int, default=20)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--reuse-cache-from", type=Path, action="append", default=[])
    a = p.parse_args()
    evidence, output = a.evidence.resolve(), a.output.resolve()
    config, lanes = read_json(evidence / "inference_config.json"), read_json(evidence / "lanes.json")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    client = StageClient(ROOT, output, {**config["vlm"], "max_api_calls": a.max_api_calls}, cache_only=a.cache_only,
                         reuse_cache_from=[x.resolve() for x in a.reuse_cache_from])
    results = []
    for item in disagreements(config, lanes, reference_lanes(a.reference)):
        key = item["approach"]
        if a.approaches and key not in a.approaches:
            continue
        views = street_views(config, key, ROOT)
        sheet = blind_sheet(evidence / "surface_audits" / f"in_{key}" / "scene_raw.png", views, output / "sheets" / f"{key}.jpg")
        try:
            reading = client.run(f"lane_check_{key}", sheet, check_prompt([c for _, c in views]), validate_reading)
        except ValidationError as error:
            results.append({**item, "verdict": "unclear", "cause": "reading_failed", "error": str(error), "sheet": f"sheets/{key}.jpg"})
            continue
        found, cause = verdict(reading, item["vlm_count"], item["reference"])
        results.append({**item, "reading": reading, "layout_seen": layout(reading), "verdict": found, "cause": cause,
                        "confidence": reading["count_confidence"], "note": reading.get("limitations", ""), "sheet": f"sheets/{key}.jpg"})
        print(key, "VLM", item["vlm_count"], "reference", item["reference_count"], "seen", len(reading["lanes"]), "->", found, flush=True)
    write_json(output / "results.json", results)
    for stale in set((output / "sheets").glob("*.jpg")) - {output / r["sheet"] for r in results}:
        stale.unlink()  # approach no longer differs from the reference
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    write_json(output / "summary.json", {"checked": len(results), "verdicts": counts, "vlm_calls_this_run": client.calls,
                                         "vlm_cache_hits_this_run": client.hits, "model": f"{client.provider}/{client.model}",
                                         "method": "blind reading of the stop-bar lane layout; neither answer shown to the model",
                                         "scope": "A model reading of the same imagery, not ground truth."})
    write_json(output / "status.json", {"state": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    print({"checked": len(results), "verdicts": counts, "calls": client.calls, "hits": client.hits})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
