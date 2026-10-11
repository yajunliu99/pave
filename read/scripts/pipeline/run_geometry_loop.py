"""Trace candidate lane regions for one junction with a draft -> review loop."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop.geometry_stage import run_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Write crop windows and the call budget; contact no service")
    parser.add_argument("--cache-only", action="store_true", help="Disallow new VLM API requests")
    parser.add_argument("--reuse-cache-from", type=Path, action="append", default=[],
                        help="Reuse content-identical VLM stages from a prior run instead of requesting them again")
    parser.add_argument("--max-vlm-calls", type=int, help="Override the config's cap on new VLM API requests")
    args = parser.parse_args()
    result = run_site(args.config.resolve(), ROOT, args.output.resolve(), plan_only=args.plan_only, cache_only=args.cache_only,
                      reuse_cache_from=[p.resolve() for p in args.reuse_cache_from], max_vlm_calls=args.max_vlm_calls)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
