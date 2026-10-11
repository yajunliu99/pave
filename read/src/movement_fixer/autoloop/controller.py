"""Bounded draft -> check -> review -> revise loop.

The loop guarantees termination and a recorded reason, not correctness. It never
ends on an unexplained state: each run finishes as accepted, accepted after
revision, or unresolved with the reason and the issues still open.
"""
from __future__ import annotations

from ..hybrid.common import ValidationError

DEFAULTS = {"max_rounds": 3, "converge_px": 2.0}
STOP_REASONS = ("accepted", "converged", "round_limit", "oscillation", "no_applicable_edits",
                "budget_exhausted", "review_failed")


def run_review_loop(state, *, check, review, apply, signature, magnitude, policy=None, budget_left=lambda: True):
    """Run review rounds on `state` until a stop reason is reached.

    check(state) -> issues, each with a `key` tied to the geometry it flags
    review(state, issues, round_number, rounds) -> {"edits", "issue_responses", ...}
    apply(state, edits) -> (new_state, applied, rejected)
    signature(state) -> hashable identity, used to detect a return to an earlier state
    magnitude(before, after) -> size of the change, or None for a structural change
    """
    p = {**DEFAULTS, **(policy or {})}
    seen = {signature(state)}
    dismissed, rounds, unresolved = {}, [], []
    revised, stop = False, "round_limit"
    for number in range(1, p["max_rounds"] + 1):
        issues = [i for i in check(state) if i["key"] not in dismissed]
        if not budget_left():
            stop = "budget_exhausted"
            break
        record = {"round": number, "issues": [i["key"] for i in issues]}
        rounds.append(record)
        try:
            response = review(state, issues, number, rounds[:-1])
        except ValidationError as error:
            record["error"] = str(error)
            stop = "review_failed"
            break
        decisions = {r["issue"]: r for r in response.get("issue_responses", [])}
        dismissed.update({k: r["reason"] for k, r in decisions.items() if r["decision"] == "dismiss"})
        unresolved = [k for k, r in decisions.items() if r["decision"] == "unresolved"]
        record.update(verdict=response.get("verdict"), dismissed=[k for k in decisions if k in dismissed],
                      unresolved=unresolved, notes=response.get("notes", []))
        edits = response.get("edits", [])
        if not edits:
            stop = "accepted"
            break
        candidate, applied, rejected = apply(state, edits)
        record.update(edits_applied=applied, edits_rejected=rejected)
        if not applied:
            stop = "no_applicable_edits"
            break
        mark = signature(candidate)
        if mark in seen:
            # The review undid an earlier revision; keep the current state and flag it.
            record["returned_to_earlier_state"] = True
            stop = "oscillation"
            break
        change = magnitude(state, candidate)
        record["change_px"] = change
        state, revised = candidate, True
        seen.add(mark)
        if change is not None and change < p["converge_px"]:
            stop = "converged"
            break
    if stop == "accepted":
        open_issues = unresolved
    else:
        open_issues = [i["key"] for i in check(state) if i["key"] not in dismissed]
    settled = stop in ("accepted", "converged") and not open_issues
    status = ("accepted_after_revision" if revised else "accepted") if settled else "unresolved"
    return {"state": state, "status": status, "stop_reason": stop, "revised": revised, "rounds": rounds,
            "open_issues": open_issues, "dismissed": dismissed}
