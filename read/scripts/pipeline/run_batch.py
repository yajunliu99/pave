"""Run run_site_pipeline.py for every site of a batch list, a few at a time.

The list (e.g. configs/batches/asu_utdf_sites.json) names network nodes; each site runs under the name
<prefix>_<node_id>, so a rerun resumes it: finished stages are skipped and cached answers are reused.
Every site writes its own log; summary.json records the outcome per site. Flags after `--` are passed
to each run_site_pipeline.py call unchanged (e.g. -- --acquire --audit-controls).
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.paths import runs_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--node-csv", type=Path, required=True)
    p.add_argument("--link-csv", type=Path, required=True)
    p.add_argument("--prefix", default="tempe")
    p.add_argument("--only", nargs="*", type=int, help="node IDs to run; default every site in the list")
    p.add_argument("--skip", nargs="*", type=int, default=[], help="node IDs to leave out")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--logs", type=Path, default=runs_root() / "batch_logs")
    p.add_argument("passthrough", nargs=argparse.REMAINDER)
    a = p.parse_args()
    extra = [x for x in a.passthrough if x != "--"]
    sites = [s for s in json.loads(a.sites.read_text())["sites"]
             if (not a.only or s["node_id"] in a.only) and s["node_id"] not in a.skip]
    a.logs = a.logs.resolve()
    a.logs.mkdir(parents=True, exist_ok=True)
    summary_path = a.logs / f"summary_{a.sites.stem}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    def run(site):
        name = f"{a.prefix}_{site['node_id']}"
        log = a.logs / f"{name}.log"
        started = time.time()
        with log.open("a") as stream:
            stream.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {' '.join(map(str, extra))}\n")
            stream.flush()
            code = subprocess.run([sys.executable, str(ROOT / "scripts/pipeline/run_site_pipeline.py"), "--name", name,
                                   "--node-id", str(site["node_id"]), "--node-csv", str(a.node_csv.resolve()),
                                   "--link-csv", str(a.link_csv.resolve()), *extra],
                                  stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, env={"PYTHONUNBUFFERED": "1", **_env()}).returncode
        record = {"name": name, "utdf_intid": site.get("utdf_intid"), "utdf_name": site.get("utdf_name"), "exit_code": code,
                  "minutes": round((time.time() - started) / 60, 1), "log": str(log),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        print(f"{name} exit {code} after {record['minutes']} min", flush=True)
        return site["node_id"], record

    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for node, record in pool.map(run, sites):
            summary[str(node)] = record
            summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    failed = [r["name"] for r in summary.values() if r["exit_code"]]
    print({"sites": len(sites), "failed": failed, "summary": str(summary_path)})
    return 1 if failed else 0


def _env():
    import os
    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
