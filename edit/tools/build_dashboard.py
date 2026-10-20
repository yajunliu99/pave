"""Run every facility template on one proposal and package the results for the dashboard.

  python tools/build_dashboard.py --proposal scenarios/sr347_i10_ddi/proposals/<answer>.json \
      --proposer-input <the exact image the proposer saw>

Fills the Edit view of PAVE's page template (../dashboard/pave_template.html) and writes a self-contained page with
that view alone (data and every image inlined as data: URIs), default out/dashboard/sr347_scenario_editor.html.
--data-out writes the view's data instead: dashboard/build_pave.py runs this script that way and puts the data into
the PAVE dashboard (../dashboard/pave.html).
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pave_edit.checks import route_table, run_checks  # noqa: E402
from pave_edit.georef import Georef  # noqa: E402
from pave_edit.gmns import Network  # noqa: E402
from pave_edit.pipeline import run  # noqa: E402
from pave_edit.proposal import (EXPECTED_FACTS, TERMINAL_ROLES, build_prompt, table_signature,  # noqa: E402
                            validate)
from pave_edit.render import view_extent  # noqa: E402
from pave_edit.standards import evaluate  # noqa: E402
from pave_edit.templates import baseline_spec  # noqa: E402
from pave_edit.vlm import MockProposer  # noqa: E402

TARGETS = [("diamond", "Conventional diamond"), ("ddi", "Diverging diamond"), ("spui", "Single point (SPUI)")]
GATES = [("P0 parse", "The answer can be read (one structured record)"),
         ("P1 schema", "It uses only the allowed interchange types and changes"),
         ("P2 id exists", "Every link and node it names exists in the network"),
         ("P3 echo neighbours", "It quotes each link's start and end node correctly"),
         ("P4 role fit", "The links it picked form a diamond ramp terminal"),
         ("P5 bridge continuity", "Its two ramp terminals connect over the freeway"),
         ("P6 anchor", "Each ramp terminal's grid square matches the network"),
         ("P7 class cross-check", "Its interchange type agrees with the network"),
         ("P8 operations", "The requested change is one the tool can build")]


def net_json(net: Network, geo: Georef):
    dn, dl = geo.network(net)
    return {"nodes": [{"id": n["node_id"], "x": n["x_coord"], "y": n["y_coord"], "X": dn[i][0], "Y": dn[i][1],
                       "t": n["node_type"], "g": n.get("control_group", ""), "c": n.get("ctrl_type", ""), "name": n.get("name", "")}
                      for i, n in net.nodes.items()],
            "links": [{"id": l["link_id"], "a": l["from_node_id"], "b": l["to_node_id"], "t": l["facility_type"],
                       "g": l.get("control_group", ""), "s": l.get("control_status", ""), "name": l.get("name", ""),
                       "len": round(l["length"] * 1000, 1), "geom": [[round(x, 2), round(y, 2)] for x, y in l["geometry"]],
                       "d": [[x, y] for x, y in dl[i]]}
                      for i, l in net.links.items()],
            "movements": len(net.movements)}


def routes_json(net):
    return [{"o": r["origin_node"], "d": r["destination_node"], "status": r["status"],
             "links": [int(x) for x in r["link_sequence"].split(";") if x],
             "nodes": [int(x) for x in r["node_sequence"].split(";") if x]} for r in route_table(net)]


def gate_rows(defects):
    failed = {d["gate"] for d in defects}
    first_fail = next((i for i, (g, _) in enumerate(GATES) if g in failed), None)
    rows = []
    for i, (g, text) in enumerate(GATES):
        status = "fail" if g in failed else ("skipped" if first_fail is not None and i > first_fail else "pass")
        rows.append({"gate": g, "text": text, "status": status,
                     "defects": [f"{d['where']}: {d['message']}" for d in defects if d["gate"] == g]})
    return rows


def data_uri(src: Path, width: int, quality=80):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return {"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "w": im.width, "h": im.height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, default=ROOT / "scenarios" / "sr347_i10_ddi")
    ap.add_argument("--proposal", type=Path, required=True)
    ap.add_argument("--proposer-input", type=Path, required=True, help="the exact image the proposer was shown")
    ap.add_argument("--proposer-prompt", type=Path, help="the exact prompt file the proposer was shown (checked against the current one)")
    ap.add_argument("--proposer-label", default="", help="optional description of who wrote the answer")
    ap.add_argument("--proposer-note", default="")
    ap.add_argument("--wrong-proposal", type=Path, default=None)
    ap.add_argument("--template", type=Path, default=ROOT.parent / "dashboard" / "pave_template.html", help="PAVE's page template")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "dashboard" / "sr347_scenario_editor.html", help="the page")
    ap.add_argument("--data-out", type=Path, help="write the Edit view's data (JSON) here instead of a page")
    args = ap.parse_args()

    scen_dir = args.scenario
    sc = json.loads((scen_dir / "scenario.json").read_text())
    base = Network.load(scen_dir / sc["base_network"])
    base.rebuild_movements()
    geo = Georef(scen_dir, base)
    extent = view_extent(base)
    proposal = json.loads(args.proposal.read_text())
    defects, terminals = validate(proposal, base, extent)
    if defects:
        raise SystemExit(f"the proposal does not pass the gates: {defects}")

    data = {"scenario": {k: sc[k] for k in ("scenario_id", "title", "sentence", "site", "legs")},
            "extent": extent, "roles": list(TERMINAL_ROLES), "facts": EXPECTED_FACTS,
            "base": net_json(base, geo), "table_signature": table_signature(base), "targets": {}}

    # proposer side: inputs, the answer, the gates (and a wrong answer for contrast)
    prompt = build_prompt(sc, base, extent, hide_names=True)
    if args.proposer_prompt:
        seen = args.proposer_prompt.read_text().split("=== QUERY ===\n", 1)[-1].rstrip("\n")
        if seen != prompt.rstrip("\n"):
            raise SystemExit("the prompt the proposer saw differs from the current prompt; rerun the proposer")
    data["proposer"] = {"label": args.proposer_label, "note": args.proposer_note,
                        "image": data_uri(args.proposer_input, 1600),
                        "prompt": prompt, "proposal": proposal,
                        "gates": gate_rows([])}
    wrong = args.wrong_proposal or scen_dir / "mock_proposal_bad.json"
    if wrong.exists():
        wp = json.loads(wrong.read_text())
        wd, _ = validate(wp, base, extent)
        sw = wp["terminals"][0]["bindings"]
        data["wrong"] = {"what": f"At the SW ramp terminal this answer picked links {sw['bridge_in']['link_id']} and "
                                 f"{sw['bridge_out']['link_id']} the wrong way round: each points in the opposite direction "
                                 "to the one it was chosen for",
                         "gates": gate_rows(wd)}

    # basemap: the wide aligned image the georeference was read from
    meta = json.loads((scen_dir / geo.meta["image_meta"]).read_text())
    half = meta["size_px"][0] / 2 * meta["metres_per_pixel"]
    # a sharper layer for the centre: same centre and bearing, zoom 16.6 (about 0.66 m per pixel)
    a = sc["aerial"]
    dmeta = json.loads((scen_dir / (Path(a["file"]).stem + ".meta.json")).read_text())
    assert (dmeta["center_lat"], dmeta["center_lon"], dmeta["bearing_up_deg"]) == (meta["center_lat"], meta["center_lon"], meta["bearing_up_deg"])
    detail = data_uri(scen_dir / a["file"], 2560, 72) | {"half_m": round(dmeta["size_px"][0] / 2 * dmeta["metres_per_pixel"], 1)}
    data["basemap"] = data_uri(scen_dir / geo.meta["image"], 2560, 74) | {"detail": detail,
        "half_m": round(half, 1), "attribution": sc["aerial"]["attribution"], "fetched_utc": meta["fetched_utc"],
        "zoom": meta["zoom"], "bearing": meta["bearing_up_deg"], "method": geo.meta["method"]}

    # the existing facility: same checks, run on the base itself
    bspec = baseline_spec(terminals)
    bchecks = run_checks(base, base, terminals, bspec, proposal.get("expected_facts", []))
    data["targets"]["diamond"] = {"label": TARGETS[0][1], "net": None, "routes": routes_json(base),
                                  "checks": bchecks, "standards": evaluate("diamond", base, base, terminals, bspec, bchecks, sc.get("site")),
                                  "log": ["no edit: this is the base network"], "changes": None, "changed": {},
                                  "groups": {g: {"required": len(v["required"]), "forbidden": len(v["forbidden"])} for g, v in bspec["groups"].items()}}

    for key, label in TARGETS[1:]:
        out = ROOT / "out" / sc["scenario_id"] / f"dashboard_{key}"
        with contextlib.redirect_stdout(io.StringIO()):
            code = run(scen_dir, out, MockProposer(args.proposal, name="session", model=args.proposer_label or "saved answer"),
                       hide_names=True, target=key)
        st = json.loads((out / "run_state.json").read_text())
        if code != 0:
            raise SystemExit(f"{key}: pipeline status {st['status']}")
        after = Network.load(out / "after_network")
        changed, deleted = {}, {"node": [], "link": []}
        import csv
        for r in csv.DictReader((out / "change_table.csv").open()):
            if r["object"] in ("node", "link") and r["change"] != "unchanged":
                if r["change"] == "deleted":
                    deleted[r["object"]].append(int(r["id"]))
                else:
                    changed[f"{r['object']}:{r['id']}"] = r["change"]
        data["targets"][key] = {"label": label, "net": net_json(after, geo), "routes": routes_json(after),
                                "checks": st["checks"], "standards": st["standards"], "changes": st["changes"],
                                "log": [line for a in st["applied"] for line in a["log"]], "changed": changed, "deleted": deleted,
                                "groups": {g: {"required": len(v["required"]), "forbidden": len(v["forbidden"])}
                                           for g, v in st["spec"]["groups"].items()}}

    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    if args.data_out:
        args.data_out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.data_out} (Edit view data), {len(payload) // 1024} KB")
        return
    page = args.template.read_text(encoding="utf-8")
    if page.count("/*__EDIT__*/") != 1:
        raise SystemExit(f"{args.template} must contain one /*__EDIT__*/ placeholder")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page.replace("/*__EDIT__*/", payload), encoding="utf-8")
    print(f"wrote {args.out} (the Edit view alone), {args.out.stat().st_size // 1024} KB, self-contained")


if __name__ == "__main__":
    main()
