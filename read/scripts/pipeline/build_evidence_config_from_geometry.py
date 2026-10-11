"""Put model-traced candidate regions into a visual-evidence config for the hybrid pipeline.

Sources and street-view sampling come from a base config; the satellite sections are
replaced. Legs are the network links of the traced sections, so any number of legs at
any angle is handled; a base config's legs are kept only when they describe exactly
these sections. A section the tracing left empty is left out and listed. Width references
are every inbound region rather than a hand-picked list, so their median is the typical
strip width.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.hybrid.common import read_json, require, write_json
from movement_fixer.hybrid.evidence_config import assert_no_answer_fields
from movement_fixer.hybrid.legs import check_sections

COLORS = {"inbound_stopbar": "#16d5ec", "outbound_receiving": "#ffbd59"}


def network_legs(sections):
    legs = {}
    for s in sections:
        legs.setdefault(s["direction"], {})["in_link_id" if s["kind"] == "inbound_stopbar" else "out_link_id"] = s["link_id"]
    return legs


def build(base, sections, pilot_id, yolo_device, max_api_calls, frozen_references=None):
    config = copy.deepcopy(base)
    colors = {s["id"]: s.get("color") for s in base.get("sections", [])}
    config["pilot_id"] = pilot_id
    config["sections"] = []
    traced = [s for s in sections if s["regions"]]
    config["untraced_sections"] = [{"id": s["id"], "reason": s.get("review", {}).get("stop_reason")} for s in sections if not s["regions"]]
    for section in traced:
        config["sections"].append({
            "id": section["id"], "source_id": "satellite", "kind": section["kind"], "direction": section["direction"],
            # The hybrid renderer rotates by quarter turns only; polygons keep their exact source pixels.
            "rotation_ccw": round(section["heading_deg"] / 90) % 4 * 90, "heading_deg": section["heading_deg"],
            **({"arm": section["arm"]} if section.get("arm") else {}),
            "color": colors.get(section["id"]) or COLORS[section["kind"]],
            "regions": [{k: r[k] for k in ("id", "polygon", "gate_point")} for r in section["regions"]],
            "coverage": "candidate_regions_only", "geometry_source": section["geometry_source"],
            "notes": "Candidate regions are unvalidated hypotheses and may contain non-road pavement or multiple facilities."})
    # Street views of an approach whose section could not be traced have nothing to classify.
    approaches = {s["direction"] for s in config["sections"] if s["kind"] == "inbound_stopbar"}
    config["context_views"] = [v for v in config.get("context_views", []) if v["direction"] in approaches]
    if config.get("gsv_sampling"):
        config["gsv_sampling"]["missing"] = [m for m in config["gsv_sampling"].get("missing", []) if m["direction"] in approaches]
    if config.get("legs") != network_legs(traced):  # equal dicts keep the base's key order, and so its prompt text
        config["legs"] = network_legs(traced)
    check_sections(config["sections"], config["legs"])
    config["geometry_checks"]["reference_regions"] = [f"{s['id']}:{r['id']}" for s in config["sections"]
                                                      if s["kind"] == "inbound_stopbar" for r in s["regions"]]
    if frozen_references:
        config["geometry_checks"]["frozen_reference_width_px"] = frozen_references
    config["yolo"]["device"] = yolo_device
    if max_api_calls is not None:
        config["vlm"]["max_api_calls"] = max_api_calls
    assert_no_answer_fields(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="base evidence config: street-view sources, sampling and policies")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--freeze-width-reference", type=Path,
                        help="geometry_hints.json of an earlier classification of this site; its strip-width reference is kept")
    args = parser.parse_args()
    sections = read_json(args.geometry_run / "geometry" / "sections.json")
    frozen = ({sid: r["width_px"] for sid, r in read_json(args.freeze_width_reference)["references"].items()}
              if args.freeze_width_reference and args.freeze_width_reference.exists() else None)
    config = build(read_json(args.base), sections, args.pilot_id, args.yolo_device, args.max_api_calls, frozen)
    write_json(args.output, config)
    print({s["id"]: len(s["regions"]) for s in config["sections"]}, "untraced:", config["untraced_sections"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
