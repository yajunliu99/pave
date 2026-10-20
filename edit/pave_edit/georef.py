"""Display georeference: draw a schematic network on the site imagery.

The base network's nodes and link alignments were read off the satellite image (georef.json).
  - a node or link the template left unchanged is drawn exactly at its measured position;
  - a moved or new node is mapped with an affine fitted on the ports of its own terminal (or on
    all terminal ports for a new group such as the SPUI single point), so a crossover at the
    schematic midpoint of a terminal lands at the midpoint of the real terminal; nodes outside
    any terminal fall back to a moving-least-squares affine (Schaefer et al. 2006);
  - a base link whose ends moved keeps its real alignment, bent smoothly toward the new ends;
  - a new link is a straight line between its mapped nodes (arcs are mapped point by point).

This is for display only. The GMNS tables and all checks stay in schematic coordinates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .gmns import Network


def _along(pts, f):
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    target, acc = sum(seg) * f, 0.0
    for (a, b), L in zip(zip(pts, pts[1:]), seg):
        if acc + L >= target and L > 0:
            t = (target - acc) / L
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        acc += L
    return tuple(pts[-1])


class MovingAffine:
    """Moving-least-squares affine map from control pairs (weights 1/d^2)."""

    def __init__(self, src, dst):
        self.src, self.dst = np.asarray(src, float), np.asarray(dst, float)

    def __call__(self, p):
        p = np.asarray(p, float)
        d2 = ((self.src - p) ** 2).sum(1)
        if d2.min() < 1e-9:
            return tuple(self.dst[d2.argmin()])
        w = 1.0 / d2
        ps, qs = w @ self.src / w.sum(), w @ self.dst / w.sum()
        P, Q = self.src - ps, self.dst - qs
        M = (P.T * w) @ P + 1e-9 * np.eye(2)
        A = np.linalg.solve(M, (P.T * w) @ Q)
        x, y = (p - ps) @ A + qs
        return (float(x), float(y))


def _fit_affine(src, dst):
    S = np.hstack([np.asarray(src, float), np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(S, np.asarray(dst, float), rcond=None)
    return lambda p: tuple(float(v) for v in np.array([p[0], p[1], 1.0]) @ M)


class Georef:
    def __init__(self, scen_dir: Path, base: Network):
        g = json.loads((Path(scen_dir) / "georef.json").read_text())
        self.meta = g
        self.base = base
        self.node_xy = {int(k): tuple(v) for k, v in g["nodes"].items()}
        self.link_geom = {int(k): [tuple(p) for p in v] for k, v in g["links"].items()}
        src, dst = [], []
        for nid, xy in self.node_xy.items():
            src.append(base.xy(nid))
            dst.append(xy)
        for lid, real in self.link_geom.items():          # interior points carry the real alignment
            schem = base.links[lid]["geometry"]
            for f in (0.25, 0.5, 0.75):
                src.append(_along(schem, f))
                dst.append(_along(real, f))
        self.map = MovingAffine(src, dst)
        ports = {}
        for nid, n in base.nodes.items():
            if n["node_type"] == "port" and n.get("control_group") and nid in self.node_xy:
                ports.setdefault(n["control_group"], []).append(nid)
        self.local = {g: _fit_affine([base.xy(n) for n in ns], [self.node_xy[n] for n in ns]) for g, ns in ports.items()}
        every = [n for ns in ports.values() for n in ns]
        self.all_ports = _fit_affine([base.xy(n) for n in every], [self.node_xy[n] for n in every]) if every else None

    def point(self, xy, group: str | None = None):
        if group in self.local:
            x, y = self.local[group](xy)
        elif group and self.all_ports:
            x, y = self.all_ports(xy)
        else:
            x, y = self.map(xy)
        return (round(x, 2), round(y, 2))

    def network(self, net: Network):
        """(node id -> image xy, link id -> image polyline) for any network built from the base."""
        nodes = {}
        for nid, n in net.nodes.items():
            same = nid in self.node_xy and nid in self.base.nodes and self.base.xy(nid) == net.xy(nid)
            nodes[nid] = self.node_xy[nid] if same else self.point(net.xy(nid), n.get("control_group") or None)
        links = {}
        for lid, l in net.links.items():
            a, b = nodes[l["from_node_id"]], nodes[l["to_node_id"]]
            base = self.base.links.get(lid)
            real = self.link_geom.get(lid)
            same_ends = base is not None and (base["from_node_id"], base["to_node_id"]) == (l["from_node_id"], l["to_node_id"])
            if real and same_ends:
                pts = _rubber_band(real, a, b)             # real alignment, ends moved to the nodes
            elif len(l["geometry"]) > 2:
                pts = [self.point(p, l.get("control_group") or None) for p in l["geometry"]]
            else:
                pts = [a, b]
            pts[0], pts[-1] = a, b
            links[lid] = [(round(x, 2), round(y, 2)) for x, y in pts]
        return nodes, links


def _rubber_band(pts, a, b):
    """Shift a polyline so its ends land on a and b, spreading the shift along its length."""
    da = (a[0] - pts[0][0], a[1] - pts[0][1])
    db = (b[0] - pts[-1][0], b[1] - pts[-1][1])
    seg = [0.0] + [math.dist(p, q) for p, q in zip(pts, pts[1:])]
    tot = sum(seg) or 1.0
    out, acc = [], 0.0
    for p, s in zip(pts, seg):
        acc += s
        t = acc / tot
        out.append((p[0] + (1 - t) * da[0] + t * db[0], p[1] + (1 - t) * da[1] + t * db[1]))
    return out
