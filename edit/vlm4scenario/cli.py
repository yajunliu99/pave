"""Run the scenario-editing pipeline on one scenario folder.

  python -m vlm4scenario                          # cached proposal, no network access
  python -m vlm4scenario --target spui            # same proposal, rebuilt as a single point interchange
  python -m vlm4scenario --backend openai --model <vision model>   # live VLM, OpenAI-compatible API (key in .env)
  python -m vlm4scenario --proposal scenarios/sr347_i10_ddi/mock_proposal_bad.json   # see the checks reject it

  A chat model as the proposer, without an API:
  python -m vlm4scenario --prepare-only --hide-names --out out/session   # writes the image and prompt, then stops
  ... give both files to the model and save its JSON answer ...
  python -m vlm4scenario --backend session --proposal <answer.json> --hide-names
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run
from .vlm import REPO_ROOT, MockProposer, OpenAIProposer, ProposerError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vlm4scenario", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=Path, default=REPO_ROOT / "scenarios" / "sr347_i10_ddi")
    ap.add_argument("--backend", choices=["mock", "session", "openai"], default="mock")
    ap.add_argument("--proposal", type=Path, help="proposal file for mock/session (mock default: <scenario>/mock_proposal.json)")
    ap.add_argument("--session-model", default="chat session", help="label recorded for --backend session")
    ap.add_argument("--prepare-only", action="store_true", help="render the model's input image and prompt, then stop")
    ap.add_argument("--env-file", type=Path, help="file with OPENAI_API_KEY etc. (default: $VLM4SCENARIO_ENV_FILE or ./.env)")
    ap.add_argument("--model", help="model for --backend openai; it must accept images (default: $OPENAI_MODEL)")
    ap.add_argument("--base-url", help="OpenAI-compatible API base URL (default: $OPENAI_BASE_URL or https://api.openai.com/v1)")
    ap.add_argument("--hide-names", action="store_true", help="blank node/link names in the tables the model sees")
    ap.add_argument("--target", choices=["ddi", "spui", "roundabout_diamond"],
                    help="facility form to build (default: the one in the proposal's operation)")
    ap.add_argument("--max-repairs", type=int, default=1)
    ap.add_argument("--out", type=Path, help="output folder (default: out/<scenario_id>/<backend>)")
    args = ap.parse_args(argv)

    if args.backend == "mock" or args.prepare_only:
        proposer = MockProposer(args.proposal or args.scenario / "mock_proposal.json")
        label = "mock" if not args.proposal else args.proposal.stem
    elif args.backend == "session":
        if not args.proposal:
            print("ERROR: --backend session needs --proposal <answer.json>", file=sys.stderr)
            return 2
        proposer = MockProposer(args.proposal, name="session", model=args.session_model)
        label = f"session_{args.proposal.stem}"
    else:
        try:
            proposer = OpenAIProposer(args.env_file, model=args.model, base_url=args.base_url,
                                      cache_dir=REPO_ROOT / "cache" / "openai")
        except ProposerError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        label = "openai_" + "".join(c if c.isalnum() or c in "-." else "_" for c in proposer.model) + ("_nonames" if args.hide_names else "")
    if args.target:
        label += f"_to_{args.target}"
    out = args.out or REPO_ROOT / "out" / args.scenario.name / label
    if args.prepare_only and not args.out:
        out = REPO_ROOT / "out" / args.scenario.name / ("prepared" + ("_nonames" if args.hide_names else ""))
    return run(args.scenario, out, proposer, max_repairs=args.max_repairs, hide_names=args.hide_names,
               prepare_only=args.prepare_only, target=args.target)


if __name__ == "__main__":
    sys.exit(main())
