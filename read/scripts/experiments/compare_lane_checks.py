"""Agreement between the automated blind lane-count check and the manual check of the same approaches.

Neither is ground truth; agreement shows how far the automated stage can stand in for a manual look.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.paths import results_root

ORDER = ["utdf", "model", "neither", "unclear"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", type=Path, default=ROOT / "configs/batches/asu_utdf_sites.json")
    p.add_argument("--prefix", default="tempe")
    p.add_argument("--manual", type=Path, default=results_root() / "reviews/lane_count_review.json")
    p.add_argument("--output", type=Path, default=results_root() / "reviews/lane_check_vs_manual.json")
    a = p.parse_args()
    manual = json.loads(a.manual.read_text())["reviews"]
    pairs = []
    for site in json.loads(a.sites.read_text())["sites"]:
        path = results_root() / f"{a.prefix}_{site['node_id']}" / "lane_check/results.json"
        for check in json.loads(path.read_text()) if path.exists() else []:
            key = f"{site['node_id']}_{check['approach']}"
            pairs.append({"approach": key, "automated": check["verdict"], "manual": manual.get(key, {}).get("verdict"),
                          "automated_seen": check.get("layout_seen"), "manual_seen": manual.get(key, {}).get("layout_seen"),
                          "manual_confidence": manual.get(key, {}).get("confidence")})
    both = [x for x in pairs if x["manual"]]
    matrix = Counter((x["manual"], x["automated"]) for x in both)
    decided = [x for x in both if x["manual"] in ("utdf", "model") and x["automated"] in ("utdf", "model")]
    confident = [x for x in decided if x["manual_confidence"] in ("medium", "high")]
    report = {"checked_automatically": len(pairs), "with_manual_check": len(both),
              "same_verdict": sum(x["manual"] == x["automated"] for x in both),
              "both_decided": len(decided), "agree_when_both_decided": sum(x["manual"] == x["automated"] for x in decided),
              "agree_when_both_decided_and_manual_confident": [sum(x["manual"] == x["automated"] for x in confident), len(confident)],
              "matrix_manual_rows_automated_columns": {m: {c: matrix[(m, c)] for c in ORDER} for m in ORDER},
              "automated_verdicts": dict(Counter(x["automated"] for x in pairs)), "pairs": pairs}
    a.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"{'manual \\ automated':>18} " + " ".join(f"{c:>8}" for c in ORDER))
    for m in ORDER:
        print(f"{m:>18} " + " ".join(f"{matrix[(m, c)]:>8}" for c in ORDER))
    print({k: v for k, v in report.items() if k not in ("pairs", "matrix_manual_rows_automated_columns")})


if __name__ == "__main__":
    main()
