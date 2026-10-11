"""The scenario-editing pipeline, in the seven steps the dashboard shows.

  1  Existing network       read the base GMNS tables and rebuild the turning movements
  2  What the model sees    render the ID-marked image and write the prompt
  3  Model proposal         ask the proposer: a live VLM, a cached answer, or a chat session
  4  Proposal checks        gates P0-P8 on the answer; on failure, one repair request with the defects
  5  Building the design    the template for the requested facility builds the new network
  6  Network checks         check chain C1-C6 plus the template's own checks, then the design guidance
  7  Export                 GMNS after-network, change table, edit summary, Network Wrangler project
                            card, before/after figure

Files written to the output folder:
  ai_input.png, ai_prompt.txt                  step 2
  proposal.json, proposal_checks.json          steps 3-4 (repair_prompt.txt when the answer is rejected)
  network_checks.json, design_guidance.json    step 6
  after_network/, change_table.csv, edit_summary.json, wrangler/, before_after.png   step 7
  report.md, run_state.json                    every run, also when it stops early
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import checks as chk
from .diff import change_counts, change_table
from .gmns import Network, write_csv
from .proposal import (IMPLEMENTED_CONVERSIONS, SYSTEM_PROMPT, arterial_bearing, build_prompt,
                       build_repair_prompt, table_signature, validate)
from .render import before_after_figure, proposer_image, view_extent
from .standards import evaluate as evaluate_standards
from .templates import TEMPLATES, TemplateError
from .vlm import ProposerError, parse_json_text
from .wrangler import LANES_PLACEHOLDER, SitePlacement, project_card, write_card, write_roadway

STEPS = ("Existing network", "What the model sees", "Model proposal", "Proposal checks",
         "Building the design", "Network checks", "Export")


def log(step: int | None, msg: str) -> None:
    print(f"[{step}/{len(STEPS)}] {msg}" if step else f"      {msg}", flush=True)


class Stop(Exception):
    """Ends a run early with a status and exit code; the report is still written."""

    def __init__(self, status: str, code: int, error: str | None = None):
        super().__init__(error or status)
        self.status, self.code, self.error = status, code, error


def run(scen_dir: Path, out: Path, proposer, *, max_repairs: int = 1, hide_names: bool = False,
        prepare_only: bool = False, target: str | None = None) -> int:
    """Run all seven steps on one scenario folder; returns 0 when every check passes.

    target: the facility form to build, replacing the to_class of every replace_facility operation
    (the user picks the new form; the proposal still supplies the existing class and the bindings)."""
    scen_dir, out = Path(scen_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scenario = json.loads((scen_dir / "scenario.json").read_text())
    state = {"scenario": scenario, "backend": proposer.name, "model": getattr(proposer, "model", ""),
             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts": [], "status": "running"}
    try:
        base, extent = existing_network(scen_dir, scenario)
        image, query = ai_inputs(scen_dir, scenario, base, extent, out, hide_names)
        if prepare_only:
            state["status"] = "prepared"
            print(f"\nModel inputs: {out / 'ai_input.png'}\n              {out / 'ai_prompt.txt'}")
            return 0
        proposal, terminals = propose_and_check(proposer, query, image, base, extent, out, state, max_repairs)
        ops = requested_operations(proposal, target, state)
        after, applied, spec = build_design(base, ops, terminals, state)
        results, n_ok = network_checks(scenario, base, after, terminals, spec, proposal, out, state)
        export(scen_dir, scenario, base, after, extent, applied, spec, proposal, results, out, state)
    except Stop as e:
        state["status"] = e.status
        if e.error:
            state["error"] = e.error
        return _finish(out, state, e.code)
    state["status"] = "accepted" if n_ok == len(results) else "checks_failed"
    return _finish(out, state, 0 if state["status"] == "accepted" else 1)


# 1. Existing network
def existing_network(scen_dir: Path, scenario: dict):
    base = Network.load(scen_dir / scenario["base_network"])
    base.rebuild_movements()
    log(1, f"{len(base.nodes)} nodes, {len(base.links)} links, {len(base.movements)} turning movements; "
           f"interchange type from the tables = {table_signature(base)}")
    return base, view_extent(base)


# 2. What the model sees
def ai_inputs(scen_dir: Path, scenario: dict, base: Network, extent, out: Path, hide_names: bool):
    image = proposer_image(scenario, scen_dir, base, extent, out / "ai_input.png")
    query = build_prompt(scenario, base, extent, hide_names=hide_names)
    (out / "ai_prompt.txt").write_text(f"=== SYSTEM ===\n{SYSTEM_PROMPT}\n\n=== QUERY ===\n{query}\n")
    log(2, f"rendered the ID-marked image ({len(image) // 1024} KB) and the prompt ({len(query)} chars)")
    return image, query


# 3-4. Model proposal and proposal checks (one repair request when the checks reject it)
def propose_and_check(proposer, query: str, image: bytes, base: Network, extent, out: Path, state: dict, max_repairs: int):
    prompt, proposal, defects, terminals = query, None, [], []
    for attempt in range(max_repairs + 1):
        t0 = time.time()
        try:
            text, meta = proposer.propose(prompt, image, SYSTEM_PROMPT)
        except ProposerError as e:
            log(3, f"proposer failed: {e}")
            raise Stop("proposer_error", 2, str(e)) from e
        meta["seconds"] = round(time.time() - t0, 1)
        try:
            proposal = parse_json_text(text)
            defects, terminals = validate(proposal, base, extent)
        except (ValueError, json.JSONDecodeError) as e:
            proposal, defects, terminals = text, [{"gate": "P0 parse", "where": "root", "message": f"not JSON: {e}"}], []
        state["attempts"].append({"attempt": attempt, "meta": meta, "proposal": proposal, "defects": defects})
        log(3, f"answer {attempt + 1} from {proposer.name} ({meta.get('model', meta.get('file'))}, "
               f"{meta['seconds']} s{', cache' if meta.get('cache_hit') else ''})")
        log(4, "all checks on the proposal passed" if not defects else f"the checks rejected the proposal: {len(defects)} defect(s)")
        for d in defects:
            log(None, f"[{d['gate']}] {d['where']}: {d['message']}")
        if not defects:
            break
        if proposer.name in ("mock", "session") or attempt == max_repairs:
            (out / "repair_prompt.txt").write_text(build_repair_prompt(query, proposal, defects))
            break
        prompt = build_repair_prompt(query, proposal, defects)
        log(4, "sending one repair request with the structured defects")
    (out / "proposal.json").write_text(json.dumps(proposal, indent=2) if isinstance(proposal, dict) else str(proposal))
    (out / "proposal_checks.json").write_text(json.dumps(state["attempts"], indent=2, default=str))
    if defects:
        state["defects"] = defects
        raise Stop("rejected", 1)
    state["proposal"] = proposal
    state["arterial_bearing_deg"] = arterial_bearing(base, proposal.get("arterial_axis_nodes"))
    return proposal, terminals


def requested_operations(proposal: dict, target: str | None, state: dict) -> list[dict]:
    ops = proposal["operations"]
    if not target:
        return ops
    ops = [dict(op, to_class=target) if op["op"] == "replace_facility" else op for op in ops]
    bad = [op for op in ops if op["op"] == "replace_facility" and (op["from_class"], target) not in IMPLEMENTED_CONVERSIONS]
    if bad:
        raise Stop("template_failed", 1, f"no template for {bad[0]['from_class']} -> {target}")
    state["target_override"] = target
    log(4, f"facility form set by the user: replace_facility -> {target}")
    return ops


# 5. Building the design
def build_design(base: Network, ops: list[dict], terminals, state: dict):
    after, applied, spec = base, [], None
    try:
        for op in ops:
            fn = TEMPLATES[(op["op"], op["from_class"], op["to_class"])]
            sel = [t for t in terminals if t.name in op["terminals"]]
            after, oplog, spec = fn(after, sel, f"after_{op['to_class']}")
            applied.append({"op": op, "log": oplog})
    except TemplateError as e:
        log(5, f"template failed: {e}")
        raise Stop("template_failed", 1, str(e)) from e
    state["applied"] = applied
    log(5, f"applied {len(applied)} operation(s): " + "; ".join(
        f"{a['op']['op']}({a['op']['from_class']}->{a['op']['to_class']})" for a in applied)
        + f" -> {len(after.nodes)} nodes, {len(after.links)} links, {len(after.movements)} turning movements")
    return after, applied, spec


# 6. Network checks and design guidance
def network_checks(scenario: dict, base: Network, after: Network, terminals, spec: dict, proposal: dict, out: Path, state: dict):
    results = chk.run_checks(base, after, terminals, spec, proposal.get("expected_facts", []))
    (out / "network_checks.json").write_text(json.dumps(results, indent=2))
    state["checks"] = results
    state["spec"] = {k: v for k, v in spec.items() if k != "groups"} | {
        "groups": {g: {k: [list(x) for x in v] for k, v in gs.items()} for g, gs in spec["groups"].items()}}
    n_ok = sum(c["passed"] for c in results)
    log(6, f"network checks: {n_ok}/{len(results)} passed")
    for c in results:
        if not c["passed"]:
            log(None, f"FAIL {c['check']}: {c['detail']}")
    crit = evaluate_standards(spec["target"], base, after, terminals, spec, results, scenario.get("site"))
    (out / "design_guidance.json").write_text(json.dumps(crit, indent=2))
    state["standards"] = crit
    tally = {}
    for c in crit:
        tally[c["status"]] = tally.get(c["status"], 0) + 1
    log(6, "design guidance: " + ", ".join(f"{v} {k}" for k, v in tally.items()))
    return results, n_ok


# 7. Export
def export(scen_dir: Path, scenario: dict, base: Network, after: Network, extent, applied: list, spec: dict,
           proposal: dict, results: list, out: Path, state: dict) -> None:
    target_class = spec["target"]
    gm = out / "after_network"
    after.write(gm, dataset_name=f"{scenario['title']} (after_{target_class})")
    write_csv(gm / "route_checks.csv", chk.route_table(after))
    rows = change_table(base, after)
    write_csv(out / "change_table.csv", rows)
    state["changes"] = change_counts(rows)
    (out / "edit_summary.json").write_text(json.dumps(_edit_summary(scenario, proposal, state, results), indent=2))
    if scenario.get("site"):
        n_ok = sum(c["passed"] for c in results)
        place = SitePlacement.for_scenario(scenario)
        write_roadway(base, place, out / "wrangler" / "base")
        ops = "; ".join(f"{a['op']['op']}({a['op']['from_class']} -> {a['op']['to_class']})" for a in applied)
        card = project_card(
            base, after, place, project=f"{scenario['title']}: {target_class}",
            tags=[scenario["scenario_id"], "replace_facility", f"diamond_to_{target_class}"],
            notes=(f"{scenario['sentence']} Operations: {ops}. Proposal: {state['backend']} {state['model']}. "
                   f"Generated by vlm4scenario on {state['started']}; all checks "
                   f"{'passed' if n_ok == len(results) else 'did NOT all pass'}. Coordinates are the schematic network "
                   f"placed on the site by a rigid rotation, not a survey. lanes = {LANES_PLACEHOLDER} with "
                   f"lanes_known = false means the lane count is unknown, not zero."))
        write_card(card, out / "wrangler" / "project_card.yml")
        state["wrangler_changes"] = len(card["changes"])
    changed = {"nodes": {r["id"] for r in rows if r["object"] == "node" and r["change"] in ("modified", "added")},
               "links": {r["id"] for r in rows if r["object"] == "link" and r["change"] in ("modified", "added")}}
    before_after_figure(scenario, scen_dir, base, after, extent, changed, out / "before_after.png", target_class)
    log(7, f"wrote the GMNS after-network, change table, edit summary, Network Wrangler project card "
           f"and before/after figure -> {out.name}/")


def _edit_summary(scenario, proposal, state, results):
    return {
        "project": scenario["title"],
        "scenario_id": scenario["scenario_id"],
        "sentence": scenario["sentence"],
        "category": "Interchange conversion",
        "facility": {"class_before": proposal["facility_class"],
                     "arterial_bearing_local_deg": state.get("arterial_bearing_deg")},
        "operations": proposal["operations"],
        "bindings": {t["name"]: {r: b["link_id"] for r, b in t["bindings"].items()} for t in proposal["terminals"]},
        "changes": state["changes"],
        "checks": {c["check"]: c["passed"] for c in results},
        "not_established": proposal.get("not_established", []),
        "provenance": {"proposer": state["backend"], "model": state["model"], "run": state["started"],
                       "attempts": len(state["attempts"])},
    }


def _finish(out: Path, state: dict, code: int) -> int:
    (out / "report.md").write_text(_report(state, out))
    (out / "run_state.json").write_text(json.dumps(state, indent=2, default=str))
    print(f"\nstatus: {state['status'].upper()}   report: {out / 'report.md'}")
    return code


def _report(s: dict, out: Path) -> str:
    sc = s["scenario"]
    L = [f"# {sc['title']}", "", f"> {sc['sentence']}", "",
         f"- Status: **{s['status']}**",
         f"- Proposer: `{s['backend']}` {s.get('model', '')}, {len(s['attempts'])} attempt(s)",
         f"- Target facility: `{s['target_override']}` (set by the user)" if s.get("target_override") else "- Target facility: from the proposal",
         f"- Run: {s['started']}", ""]
    for a in s["attempts"]:
        L.append(f"## Model proposal, answer {a['attempt'] + 1}")
        p = a["proposal"]
        if isinstance(p, dict):
            L += [f"- facility_class: `{p.get('facility_class')}`",
                  f"- operations: `{json.dumps(p.get('operations'))}`"]
            for t in p.get("terminals", []):
                if isinstance(t, dict) and isinstance(t.get("bindings"), dict):
                    b = ", ".join(f"{r}={v.get('link_id') if isinstance(v, dict) else v}" for r, v in t["bindings"].items())
                    L.append(f"- terminal {t.get('name')} (cell {t.get('anchor_cell')}): {b}")
        L.append("- checks on the proposal: " + ("all passed" if not a["defects"] else f"{len(a['defects'])} defect(s)"))
        L += [f"  - [{d['gate']}] {d['where']}: {d['message']}" for d in a["defects"]]
        L.append("")
    if s.get("applied"):
        L.append("## Building the design")
        for a in s["applied"]:
            L += [f"- {line}" for line in a["log"]]
        L.append("")
    if s.get("checks"):
        L += ["## Network checks", "", "| check | result | model expected it | detail |", "|---|---|---|---|"]
        for c in s["checks"]:
            det = c["detail"] if isinstance(c["detail"], str) else "; ".join(c["detail"])
            L.append(f"| {c['check']} | {'pass' if c['passed'] else '**FAIL**'} | "
                     f"{'' if c['asserted_by_proposal'] is None else ('yes' if c['asserted_by_proposal'] else 'no')} | {det} |")
        L.append("")
    if s.get("changes"):
        L += ["## Change table", "", "| object | unchanged | modified | added | deleted |", "|---|---|---|---|---|"]
        for obj, c in s["changes"].items():
            L.append(f"| {obj} | {c['unchanged']} | {c['modified']} | {c['added']} | {c['deleted']} |")
        L.append("")
    if s.get("standards"):
        L += ["## Design guidance", "", "| id | guidance | status | detail | source |", "|---|---|---|---|---|"]
        for c in s["standards"]:
            src = f"[{c['source']}]({c['url']})" if c["url"] else c["source"]
            L.append(f"| {c['id']} | {c['criterion']} | {c['status']} | {c['detail']} | {src} |")
        L.append("")
    if isinstance(s.get("proposal"), dict) and s["proposal"].get("not_established"):
        L += ["## Not established by the image or text", ""] + [f"- {x}" for x in s["proposal"]["not_established"]] + [""]
    if s.get("error"):
        L += ["## Error", "", s["error"], ""]
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "report.md")
    L += ["## Files", ""] + [f"- `{f}`" for f in files]
    return "\n".join(L) + "\n"
