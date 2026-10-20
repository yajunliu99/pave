"""Before/after change table (unchanged / modified / added / deleted)."""
from __future__ import annotations

from collections import Counter

from .gmns import Network

NODE_KEYS = ["x_coord", "y_coord", "node_type", "ctrl_type", "control_group"]
LINK_KEYS = ["from_node_id", "to_node_id", "facility_type", "geometry", "length", "lanes",
             "control_group", "control_status", "layer"]


def _mv_key(m):
    return (m["node_id"], m["ib_link_id"], m["ob_link_id"])


def change_table(before: Network, after: Network) -> list[dict]:
    rows = []

    def compare(kind, b_items, a_items, keys):
        for i in sorted(set(b_items) | set(a_items), key=str):
            if i not in a_items:
                rows.append(dict(object=kind, id=i, change="deleted", fields=""))
            elif i not in b_items:
                rows.append(dict(object=kind, id=i, change="added", fields=""))
            else:
                diff = [k for k in keys if b_items[i].get(k) != a_items[i].get(k)]
                rows.append(dict(object=kind, id=i, change="modified" if diff else "unchanged", fields=";".join(diff)))

    compare("node", before.nodes, after.nodes, NODE_KEYS)
    compare("link", before.links, after.links, LINK_KEYS)
    compare("movement", {_mv_key(m): m for m in before.movements},
            {_mv_key(m): m for m in after.movements}, ["ctrl_type"])
    for r in rows:
        if r["object"] == "movement":
            r["id"] = "{}:{}>{}".format(*r["id"])
    return rows


def change_counts(rows) -> dict:
    c = Counter((r["object"], r["change"]) for r in rows)
    return {obj: {ch: c[(obj, ch)] for ch in ("unchanged", "modified", "added", "deleted")}
            for obj in ("node", "link", "movement")}

