"""Render networks with ID marks (Set-of-Mark style) and a labelled grid for the proposer,
plus the before/after comparison figure."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from .gmns import Network  # noqa: E402
from .proposal import GRID_N, GRID_ROWS  # noqa: E402

COLORS = {"freeway": "#5b6576", "arterial": "#1f5fd6", "ramp": "#0b8a6f", "intersection_connector": "#e07b10"}
LABELLED = ("freeway", "arterial", "ramp")


def view_extent(net: Network, pad_frac: float = 0.12):
    """Square view around all nodes; the grid scaffold is laid over exactly this extent."""
    xs = [n["x_coord"] for n in net.nodes.values()]
    ys = [n["y_coord"] for n in net.nodes.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    half += 2 * half * pad_frac
    return (cx - half, cx + half, cy - half, cy + half)


def _point_along(pts, frac):
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    target, acc = sum(seg) * frac, 0.0
    for (a, b), s in zip(zip(pts, pts[1:]), seg):
        if acc + s >= target and s > 0:
            t = (target - acc) / s
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        acc += s
    return pts[-1]


def draw_network(ax, net: Network, extent, *, grid=True, label_links=True, title=None, changed=None):
    """changed: optional {"nodes": set, "links": set} drawn with a red halo."""
    changed = changed or {"nodes": set(), "links": set()}
    for l in net.links.values():
        pts, col = l["geometry"], COLORS.get(l["facility_type"], "#888")
        z = 1 if l["facility_type"] == "freeway" else 3
        xs, ys = zip(*pts)
        if l["link_id"] in changed["links"]:
            ax.plot(xs, ys, lw=9, color="#ff3b3b", alpha=.25, zorder=z - .5, solid_capstyle="round")
        ax.plot(xs, ys, lw=5, color="white", zorder=z, solid_capstyle="round")
        ax.plot(xs, ys, lw=1.8, color=col, zorder=z + .1)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12, lw=1.8,
                                     color=col, shrinkA=0, shrinkB=4, zorder=z + .2))
        if label_links and l["facility_type"] in LABELLED:
            x, y = _point_along(pts, .3 if l["facility_type"] == "freeway" else .5)
            ax.text(x, y, str(l["link_id"]), ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", zorder=12,
                    bbox=dict(boxstyle="round,pad=.25", fc=col, ec="white", lw=.8))
    seen = {}
    for n in net.nodes.values():
        seen.setdefault((n["x_coord"], n["y_coord"]), []).append(n)
    for (x, y), group in seen.items():
        ids = "/".join(str(n["node_id"]) for n in group)
        is_x = group[0]["node_type"] == "crossover"
        ring = "#ff3b3b" if any(n["node_id"] in changed["nodes"] for n in group) else "#14202b"
        ax.scatter([x], [y], s=46 if is_x else 26, facecolor="#c8283c" if is_x else "white",
                   edgecolor=ring, lw=1.4 if ring != "#14202b" else 1, zorder=10)
        ax.text(x + 9, y + 9, ids, fontsize=7.5 if not is_x else 8, color="#c8283c" if is_x else "#14202b",
                zorder=11, fontweight="bold" if is_x else None)
    xmin, xmax, ymin, ymax = extent
    if grid:
        w, h = (xmax - xmin) / GRID_N, (ymax - ymin) / GRID_N
        for i in range(1, GRID_N):
            ax.axvline(xmin + i * w, color="#9aa4b2", lw=.6, ls=(0, (4, 4)), zorder=0)
            ax.axhline(ymin + i * h, color="#9aa4b2", lw=.6, ls=(0, (4, 4)), zorder=0)
        for r in range(GRID_N):
            for c in range(GRID_N):
                ax.text(xmin + c * w + 8, ymax - r * h - 8, f"{GRID_ROWS[r]}{c + 1}", fontsize=7,
                        color="#8a94a3", ha="left", va="top", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c9d0d8")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")


def aerial_before(scenario: dict, scen_dir: Path):
    """(image array, caption): the site imagery named by the scenario's "aerial" entry, centre-cropped."""
    a = scenario.get("aerial")
    if not a:
        raise ValueError("scenario.json needs an 'aerial' entry (site imagery for the model)")
    img = mpimg.imread(str(scen_dir / a["file"]))
    f = a.get("crop_frac", 1.0)
    h, w = img.shape[:2]
    dy, dx = int(h * (1 - f) / 2), int(w * (1 - f) / 2)
    return img[dy:h - dy, dx:w - dx], a["caption"]


def _legend(fig, y=.02):
    handles = [Line2D([0], [0], color=c, lw=3, label=k.replace("_", " ")) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, y), fontsize=10)


def _credit(ax, scenario):
    """Mapbox terms require attribution wherever the imagery is shown; cropping removed its logo."""
    if scenario.get("aerial"):
        ax.text(.995, .005, scenario["aerial"].get("attribution", "Imagery © Mapbox © Maxar © OpenStreetMap contributors"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="white",
                bbox=dict(boxstyle="square,pad=.2", fc="black", alpha=.45, lw=0))


def proposer_image(scenario: dict, scen_dir: Path, net: Network, extent, out_path: Path) -> bytes:
    """One image for the proposer (the endpoint takes a single image): aerial left, IDs right."""
    aerial, caption = aerial_before(scenario, scen_dir)
    ratio = aerial.shape[1] / aerial.shape[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6 * ratio + 8.4, 7.6), gridspec_kw={"width_ratios": [ratio, 1]})
    a.imshow(aerial)
    a.set_axis_off()
    _credit(a, scenario)
    a.set_title(f"A  {caption}", fontsize=12, fontweight="bold", loc="left")
    draw_network(b, net, extent, title="B  GMNS base network: link IDs in boxes, node IDs at circles")
    _legend(fig)
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.08, wspace=.04)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, facecolor="white")
    plt.close(fig)
    data = buf.getvalue()
    Path(out_path).write_bytes(data)
    return data


TARGET_NAMES = {"ddi": "diverging diamond", "spui": "single point urban interchange",
                "roundabout_diamond": "diamond with roundabout ramp terminals", "diamond": "conventional diamond"}


def draw_on_imagery(ax, scen_dir: Path, scenario: dict, geo, net: Network, crop, *, changed=None, title=None):
    """The network drawn on the site imagery through the hand-read registration (display only)."""
    changed = changed or {"nodes": set(), "links": set()}
    meta = json.loads((scen_dir / geo.meta["image_meta"]).read_text())
    half = meta["size_px"][0] / 2 * meta["metres_per_pixel"]
    ax.imshow(mpimg.imread(str(scen_dir / geo.meta["image"])), extent=[-half, half, -half, half])
    nodes, links = geo.network(net)
    for lid, pts in links.items():
        ft = net.links[lid]["facility_type"]
        z, xs, ys = (1 if ft == "freeway" else 3), *zip(*pts)
        if lid in changed["links"]:
            ax.plot(xs, ys, lw=10, color="#ff3b3b", alpha=.45, zorder=z - .5, solid_capstyle="round")
        ax.plot(xs, ys, lw=4.6, color="black", alpha=.45, zorder=z, solid_capstyle="round")
        ax.plot(xs, ys, lw=2.2, color=COLORS.get(ft, "#888"), zorder=z + .1, solid_capstyle="round")
    for xy in {xy for xy in nodes.values()}:
        group = [nid for nid, p in nodes.items() if p == xy]
        is_x = net.nodes[group[0]]["node_type"] == "crossover"
        ring = "#ff3b3b" if any(n in changed["nodes"] for n in group) else "#111111"
        ax.scatter([xy[0]], [xy[1]], s=44 if is_x else 18, facecolor="#c8283c" if is_x else "white", edgecolor=ring,
                   lw=1.2, zorder=10)
    (x0, x1), (y0, y1) = crop
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    _credit(ax, scenario)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")


def before_after_figure(scenario: dict, scen_dir: Path, before: Network, after: Network, extent,
                        changed: dict, out_path: Path, target: str = "ddi") -> None:
    """Before and after, each as the GMNS drawing and, when the scenario has a registration
    (georef.json), drawn on the site imagery. Without one, the plain imagery is shown instead."""
    name = TARGET_NAMES.get(target, target)
    if scenario.get("aerial") and (scen_dir / "georef.json").exists():
        from .georef import Georef
        geo = Georef(scen_dir, before)
        # frame the ramp terminals (nodes with a control group), where the design changes
        pts = [xy for net in (before, after) for nid, xy in geo.network(net)[0].items() if net.nodes[nid].get("control_group")]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 110
        crop = ((cx - r, cx + r), (cy - r, cy + r))
        fig, axs = plt.subplots(2, 2, figsize=(15, 15.6))
        draw_on_imagery(axs[0, 0], scen_dir, scenario, geo, before, crop, title="Existing: network on the satellite image")
        draw_on_imagery(axs[1, 0], scen_dir, scenario, geo, after, crop, changed=changed,
                        title=f"After: {name} on the same image (display only)")
    else:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15.6))
        aerial, caption = aerial_before(scenario, scen_dir)
        axs[0, 0].imshow(aerial)
        _credit(axs[0, 0], scenario)
        axs[0, 0].set_title(f"Before: {caption}", fontsize=12, fontweight="bold", loc="left")
        for ax in axs[:, 0]:
            ax.set_axis_off()
    draw_network(axs[0, 1], before, extent, grid=False, title="GMNS before (input)")
    draw_network(axs[1, 1], after, extent, grid=False, changed=changed, title=f"GMNS after: {name} (red halo = changed)")
    fig.suptitle(scenario["title"], fontsize=17, fontweight="bold", y=.985)
    _legend(fig, y=.01)
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.04, wspace=.04, hspace=.07)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
