"""GMNS node/link/movement tables: load, copy, write, and derived helpers."""
from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

NODE_FIELDS = ["node_id", "name", "x_coord", "y_coord", "node_type", "ctrl_type",
               "scenario", "control_group", "geometry_status"]
LINK_FIELDS = ["link_id", "name", "from_node_id", "to_node_id", "directed", "dir_flag",
               "length", "facility_type", "lanes", "free_speed", "capacity", "geometry",
               "scenario", "control_group", "control_status", "geometry_status",
               "operating_data_status", "layer"]
MOVEMENT_FIELDS = ["mvmt_id", "node_id", "ib_link_id", "ob_link_id", "ctrl_type",
                   "penalty", "capacity", "control_status"]


def parse_wkt_linestring(wkt: str) -> list[tuple[float, float]]:
    body = wkt.strip()
    if not body.upper().startswith("LINESTRING"):
        raise ValueError(f"only LINESTRING geometry is supported: {wkt[:40]}")
    body = body[body.index("(") + 1: body.rindex(")")]
    return [tuple(float(v) for v in p.split()) for p in body.split(",")]


def fmt_num(v) -> str:
    """Write integral floats as ints (-600, not -600.0) so tables stay diff-friendly."""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return repr(round(v, 6))
    return str(v)


def to_wkt(points) -> str:
    return "LINESTRING (" + ", ".join(f"{fmt_num(x)} {fmt_num(y)}" for x, y in points) + ")"


def polyline_length_km(points) -> float:
    return round(sum(math.dist(a, b) for a, b in zip(points, points[1:])) / 1000, 6)


@dataclass
class Network:
    nodes: dict[int, dict]
    links: dict[int, dict]
    movements: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    # ---------- IO ----------
    @classmethod
    def load(cls, folder: Path) -> "Network":
        folder = Path(folder)
        nodes, links = {}, {}
        for r in _read(folder / "node.csv"):
            r["node_id"] = int(r["node_id"])
            r["x_coord"], r["y_coord"] = float(r["x_coord"]), float(r["y_coord"])
            if r["node_id"] in nodes:
                raise ValueError(f"duplicate node_id {r['node_id']}")
            nodes[r["node_id"]] = r
        for r in _read(folder / "link.csv"):
            r["link_id"] = int(r["link_id"])
            r["from_node_id"], r["to_node_id"] = int(r["from_node_id"]), int(r["to_node_id"])
            r["length"] = float(r["length"])
            r["geometry"] = parse_wkt_linestring(r["geometry"])
            r["layer"] = int(r.get("layer") or 0)
            if r["link_id"] in links:
                raise ValueError(f"duplicate link_id {r['link_id']}")
            links[r["link_id"]] = r
        movements = []
        if (folder / "movement.csv").exists():
            for r in _read(folder / "movement.csv"):
                for k in ("mvmt_id", "node_id", "ib_link_id", "ob_link_id"):
                    r[k] = int(r[k])
                movements.append(r)
        config = {}
        if (folder / "config.csv").exists():
            rows = _read(folder / "config.csv")
            config = rows[0] if rows else {}
        return cls(nodes, links, movements, config)

    def write(self, folder: Path, dataset_name: str | None = None) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        node_rows = [{**n, "x_coord": fmt_num(n["x_coord"]), "y_coord": fmt_num(n["y_coord"])}
                     for n in self.nodes.values()]
        link_rows = [{**l, "length": fmt_num(l["length"]), "geometry": to_wkt(l["geometry"])}
                     for l in self.links.values()]
        _write(folder / "node.csv", node_rows, NODE_FIELDS)
        _write(folder / "link.csv", link_rows, LINK_FIELDS)
        _write(folder / "movement.csv", self.movements, MOVEMENT_FIELDS)
        if self.config:
            cfg = dict(self.config)
            if dataset_name:
                cfg["dataset_name"] = dataset_name
            _write(folder / "config.csv", [cfg], list(cfg))

    def copy(self) -> "Network":
        return copy.deepcopy(self)

    # ---------- adjacency ----------
    def out_links(self, node_id: int) -> list[dict]:
        return [l for l in self.links.values() if l["from_node_id"] == node_id]

    def in_links(self, node_id: int) -> list[dict]:
        return [l for l in self.links.values() if l["to_node_id"] == node_id]

    def xy(self, node_id: int) -> tuple[float, float]:
        n = self.nodes[node_id]
        return (n["x_coord"], n["y_coord"])

    def find_link(self, a: int, b: int) -> dict | None:
        for l in self.links.values():
            if l["from_node_id"] == a and l["to_node_id"] == b:
                return l
        return None

    def refresh_link_geometry(self, link_id: int) -> None:
        """Snap a link's end vertices to its nodes and recompute its length."""
        l = self.links[link_id]
        pts = list(l["geometry"])
        pts[0], pts[-1] = self.xy(l["from_node_id"]), self.xy(l["to_node_id"])
        l["geometry"] = pts
        l["length"] = polyline_length_km(pts)

    def rebuild_movements(self) -> None:
        """Enumerate every inbound/outbound link pair at each node (GMNS movement.csv).

        The control point of a signalised connector is the node where it starts, unless
        the connector ends in a crossover node, which then holds the control itself. A roundabout
        entry yields where it meets the ring.
        """
        movements = []
        for nid, node in self.nodes.items():
            for a in self.in_links(nid):
                for b in self.out_links(nid):
                    if node["node_type"] == "crossover":
                        ctrl = "signal"
                    elif node.get("ctrl_type") == "yield" and a.get("control_status") == "yield_entry":
                        ctrl = "yield"
                    elif (str(b.get("control_status", "")).startswith("signal_required")
                          and self.nodes[b["to_node_id"]]["node_type"] != "crossover"):
                        ctrl = "signal"
                    else:
                        ctrl = ""
                    movements.append(dict(
                        mvmt_id=len(movements) + 1, node_id=nid,
                        ib_link_id=a["link_id"], ob_link_id=b["link_id"], ctrl_type=ctrl,
                        penalty="", capacity="",
                        control_status="timing_or_priority_unverified" if node.get("control_group")
                        else "not_modeled"))
        self.movements = movements


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        Path(path).write_text("")
        return
    _write(path, rows, fields or list(rows[0]))
