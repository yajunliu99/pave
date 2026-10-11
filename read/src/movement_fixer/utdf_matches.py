"""UTDF intersection -> network node, from net2cell_utdf's movement table with the checked decisions applied.

net2cell_utdf matches each UTDF intersection to a macronet node (movement_utdf.csv, utdf_intid -> node_id). A few
of those matches are wrong or have no node of their own; configs/utdf_node_overrides.json records the decisions
taken after checking them on the image (see scripts/pipeline/audit_utdf_matches.py). Everything that maps UTDF to
nodes in this project goes through here so the decisions apply everywhere.
"""
import csv
import json
from pathlib import Path

DEFAULT_OVERRIDES = Path(__file__).resolve().parents[2] / "configs" / "utdf_node_overrides.json"


def load_overrides(path=DEFAULT_OVERRIDES):
    path = Path(path)
    return {int(k): v for k, v in json.loads(path.read_text())["overrides"].items()} if path.exists() else {}


def utdf_nodes(movement_csv, overrides=None):
    """{utdf_intid: {"node_id": int or None, "note": str or None, "overridden": bool}}."""
    overrides = load_overrides() if overrides is None else overrides
    found = {}
    with open(movement_csv, newline="", encoding="utf-8-sig") as stream:
        for r in csv.DictReader(stream):
            if r.get("utdf_intid") and r.get("node_id"):
                found[int(float(r["utdf_intid"]))] = {"node_id": int(float(r["node_id"])), "note": None, "overridden": False}
    for uid, o in overrides.items():
        if o.get("keep"):
            if uid in found:
                found[uid]["note"] = o.get("note")
            continue
        found[uid] = {"node_id": o.get("node_id"), "note": o.get("note"), "overridden": True}
    return found


def node_utdf(movement_csv, overrides=None):
    """{node_id: (utdf_intid, note, overridden)} for every UTDF intersection that has a node of its own."""
    out = {}
    for uid, m in utdf_nodes(movement_csv, overrides).items():
        if m["node_id"] is not None and (m["overridden"] or m["node_id"] not in out or not out[m["node_id"]][2]):
            out[m["node_id"]] = (uid, m["note"], m["overridden"])
    return out
