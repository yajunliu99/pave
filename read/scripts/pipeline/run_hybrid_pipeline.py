"""Run the complete inlet/outlet + GSV + YOLO + VLM movement pilot."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.hybrid.pipeline import run
from movement_fixer.hybrid.common import write_json, read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--cache-only",action="store_true",help="Disallow new VLM API requests; reuse only matching evidence")
    parser.add_argument("--disable-yolo",action="store_true",help="Explicit ablation; use a separate run directory")
    parser.add_argument("--reuse-cache-from",type=Path,action="append",default=[],help="Reuse only content-identical, identity-verified VLM artifacts from a prior run")
    args=parser.parse_args()
    try:
        result=run(args.config,ROOT,args.output,prepare_only=args.prepare_only,cache_only=args.cache_only,disable_yolo=args.disable_yolo,reuse_cache_from=args.reuse_cache_from)
    except Exception as error:
        # Avoid dumping service response bodies or credentials in errors.
        info={"error_type":type(error).__name__,"message":str(error) if isinstance(error,ValueError) else "Stage failed; inspect stage artifacts and exception type"}
        for attr in ("filename","filename2","errno","winerror"):
            value=getattr(error,attr,None)
            if value is not None: info[attr]=str(value)
        status_path=args.output/"status.json"
        if status_path.exists() and read_json(status_path).get("state")=="running":
            write_json(args.output/"error.json",info)
            write_json(status_path,{"state":"failed",**info})
        print(json.dumps(info,ensure_ascii=False),file=sys.stderr)
        return 1
    print(json.dumps({k:result[k] for k in ("state","candidate_movements","lane_pair_rows","unresolved_movements","vlm_calls_this_run","vlm_cache_hits_this_run") if k in result},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
