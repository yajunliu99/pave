"""Worker side of the agent queue for vision requests (pave_read/vlm_bridge.py; how-to in docs/vlm_bridge.md).

  status                     how many requests are pending, claimed, answered
  claim [--max N] [--worker W] [--wait S] [--only-stages P ...] [--skip-stages P ...]
                             take the oldest pending requests, waiting up to S seconds for one to appear;
                             prints one JSON line per request (id, prompt_file, image, image_size);
                             stage-name prefixes keep kinds of request apart (e.g. readers --only-stages reading_,
                             everyone else --skip-stages reading_)
  answer --id ID --file F    record the JSON answer in F for a claimed request
  release [--older-than-min M]
                             put requests claimed more than M minutes ago back in the queue
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pave_read import vlm_bridge


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    c = sub.add_parser("claim")
    c.add_argument("--max", type=int, default=1)
    c.add_argument("--worker")
    c.add_argument("--wait", type=float, default=0, help="seconds to wait for a request when none is pending")
    c.add_argument("--only-stages", nargs="*", default=[], help="claim only stages starting with one of these")
    c.add_argument("--skip-stages", nargs="*", default=[], help="never claim stages starting with one of these")
    a = sub.add_parser("answer")
    a.add_argument("--id", required=True)
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--worker")
    r = sub.add_parser("release")
    r.add_argument("--older-than-min", type=float, default=30)
    args = p.parse_args()
    if args.command == "status":
        print(json.dumps(vlm_bridge.status()))
    elif args.command == "claim":
        deadline = time.time() + args.wait
        take = lambda: vlm_bridge.claim(args.max, args.worker, args.only_stages, args.skip_stages)
        claimed = take()
        while not claimed and time.time() < deadline:
            time.sleep(2)
            claimed = take()
        for request in claimed:
            print(json.dumps({k: request[k] for k in ("id", "stage", "prompt_file", "image", "image_size")}))
    elif args.command == "answer":
        try:
            body = vlm_bridge.answer(args.id, args.file.read_text(encoding="utf-8"), args.worker)
        except json.JSONDecodeError as error:
            raise SystemExit(f"not valid JSON: {error}; fix the file and run answer again")
        print(f"recorded {args.id}: {len(json.dumps(body))} characters")
    elif args.command == "release":
        print(json.dumps(vlm_bridge.release(args.older_than_min * 60)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
