"""Imagery for one junction, planned from its network node; nothing in here is site-specific.

* A satellite image centred on the node, requested by centre and zoom, so its Web Mercator viewport follows
  from the tile arithmetic alone (no registration against other images).
* For every approach: forward street views near, mid and far upstream, and a look-back view from the near
  camera. For every exit: views along and against outgoing travel near, mid and far downstream.

Legs, their names, headings and road arms are the frames the tracing stage builds from the same network, so
the files written here feed tracing, classification, the exit audit and the atlas unchanged. Every camera
pose comes from Street View metadata; a band without an acceptable panorama is recorded as missing, never
filled. plan_site() contacts no service; fetching goes through budgeted, cached clients.

By default the satellite image is the current Mapbox mosaic and each street view the panorama the metadata service
prefers (usually the latest). A policy can instead name a dated orthophoto tile service ("satellite": {"source":
"xyz_tiles", ...}), assembled over exactly the same viewport, and a "capture_before" month: each band then searches the
points around its requested position for an earlier panorama ("date_search").
"""
import csv
import io
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

from ..autoloop.frames import _along, _at_projection, _int, _lanes, _motor, _unit, frames_from_network, parse_linestring, shares_axis
from ..hybrid.common import digest, file_hash, read_json, require, write_json
from ..hybrid.sampling import offsets
from .control_acquisition import AcquisitionUnavailable, official, panoramas_near, probe_points
from .coordinates import WORLD, Viewport, local_lonlat, local_xy, mercator

POLICY = {
    "satellite": {"style": "mapbox/satellite-v9", "zoom": 19, "size": [1280, 1280], "scale": 2, "tile_size": 512},
    "positions": {"near": {"target_m": 25, "allowed_distance_m": [15, 40]},
                  "mid": {"target_m": 50, "allowed_distance_m": [40, 65]},
                  "far": {"target_m": 100, "allowed_distance_m": [80, 120]}},
    "fallback_fractions": [-0.2, 0.2],
    "minimum_position_separation_m": 10.0,
    "approach_view": {"near": {"fov": 90, "pitch": -12}, "mid": {"fov": 60, "pitch": 0}, "far": {"fov": 60, "pitch": 0}},
    "look_back_view": {"fov": 90, "pitch": -12},
    "exit_view": {"fov": 85, "pitch": -12},
    "metadata_radius_m": 12, "max_snap_distance_m": 14.0, "max_offset_from_leg_m": 14.0,
    "lane_width_prior_m": 3.6, "driving_side": "right", "image_size": [640, 640],
    "max_metadata_requests": 80, "max_image_requests": 64, "max_satellite_requests": 1,
    "capture_before": None,   # "YYYY-MM": use only panoramas captured before this month
    "official_only": False,   # use only Google's own panoramas (contributed ones can be indoors or on a sidewalk)
    "fallback_earliest": False,  # with a cutoff: if no earlier panorama is usable, take the earliest usable one found nearby
    "date_search": {"along_m": 9, "across_m": 12, "step_m": 3, "radius_m": 3},
    # On a road whose directions are separate links (a divided road), reject a panorama off this direction's
    # carriageway (more than 6 m from its link): from across a median the lanes of this direction are barely visible.
    "same_carriageway": False,
    # "section_id/position" -> panorama id, for a panorama the metadata service no longer returns at its location
    # because a later capture there hides it (Google Maps' time travel still lists it). A pinned panorama is checked
    # like any other and fails the run if it does not fit its band.
    "pinned_panoramas": {},
}
YOLO = {"weights": "~/.cache/net2cell-vlm/weights/yolo26n.pt",
        "expected_sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef", "device": "cpu", "imgsz": 1280,
        "confidence": 0.25, "high_bbox_coverage": 0.25, "occluder_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"]}
GEOMETRY_CHECKS = {"merged_width_ratio": 1.65, "minimum_motor_width_ratio": 0.65, "straight_review_lane_widths": 0.65,
                   "straight_hard_lane_widths": 1.5, "straight_review_angle_deg": 8}
STOP = ("metadata_budget_exhausted", "image_budget_exhausted", "missing_streetview_key")


def policy_with(overrides=None):
    p = {**POLICY, **(overrides or {})}
    p["satellite"] = {**POLICY["satellite"], **(overrides or {}).get("satellite", {})}
    require(p["driving_side"] in ("left", "right"), "driving_side must be left or right")
    require({"near", "mid"} <= set(p["positions"]), "Street-view bands need at least near and mid")
    require(all(isinstance(v, int) and 1 <= v <= 640 for v in p["image_size"]), "Street View images are at most 640 px a side")
    return p


def heading(vector):
    """Compass heading of an (east, north) vector."""
    return math.degrees(math.atan2(vector[0], vector[1])) % 360


def mapbox_viewport(center, zoom, size, scale=2, tile_size=512):
    """Web Mercator extent of a Static Images request by centre and zoom (styles render on 512 px tiles)."""
    width, height = size[0] * scale, size[1] * scale
    step = WORLD / (tile_size * 2 ** zoom * scale)
    cx, cy = mercator(*center)
    return Viewport(width, height, (cx - width / 2 * step, cy - height / 2 * step, cx + width / 2 * step, cy + height / 2 * step),
                    f"mapbox_static_centre_zoom: zoom {zoom}, {tile_size} px tiles, @{scale}x; imagery alignment itself unverified")


def read_env(root, *names):
    env = dict(os.environ)
    path = Path(root) / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return next((env[n] for n in names if env.get(n)), None)


class MapboxClient:
    """One cached Static Images request per distinct centre/zoom/size/style; the token never reaches a record."""

    def __init__(self, root, cache, cache_only=False, session=None, max_requests=1):
        self.root, self.cache = Path(root), Path(cache).resolve()
        self.cache_only, self.session, self.max_requests = cache_only, session or requests.Session(), max_requests
        self.token = None
        self.counts = {"satellite_requests": 0, "satellite_cache_hits": 0}

    def fetch(self, center, spec):
        request = {"style": spec["style"], "center_lonlat": [round(center[0], 7), round(center[1], 7)], "zoom": spec["zoom"],
                   "size": list(spec["size"]), "scale": spec["scale"]}
        key = digest(request)
        sidecar = self.cache / f"{key}.json"
        if sidecar.exists():
            record = read_json(sidecar)
            path = self.cache / record["file"]
            if path.exists() and file_hash(path) == record["image_sha256"]:
                self.counts["satellite_cache_hits"] += 1
                return path, record
        if self.cache_only:
            raise AcquisitionUnavailable("satellite_cache_miss")
        if self.counts["satellite_requests"] >= self.max_requests:
            raise AcquisitionUnavailable("satellite_budget_exhausted")
        self.token = self.token or read_env(self.root, "MAPBOX_TOKEN")
        if not self.token:
            raise AcquisitionUnavailable("missing_mapbox_token")
        self.counts["satellite_requests"] += 1
        lon, lat = request["center_lonlat"]
        retina = "@2x" if spec["scale"] == 2 else ""
        url = f"https://api.mapbox.com/styles/v1/{spec['style']}/static/{lon},{lat},{spec['zoom']},0/{spec['size'][0]}x{spec['size'][1]}{retina}"
        try:
            response = self.session.get(url, params={"access_token": self.token, "attribution": "false", "logo": "false"}, timeout=60)
        except requests.RequestException:
            raise AcquisitionUnavailable("satellite_transport_error") from None
        if response.status_code != 200:
            raise AcquisitionUnavailable(f"satellite_HTTP_{response.status_code}")
        try:
            with Image.open(io.BytesIO(response.content)) as image:
                size, form = list(image.size), image.format
        except (OSError, ValueError):
            raise AcquisitionUnavailable("invalid_satellite_response") from None
        expected = [spec["size"][0] * spec["scale"], spec["size"][1] * spec["scale"]]
        if size != expected:
            raise AcquisitionUnavailable(f"satellite_size_{size[0]}x{size[1]}_not_{expected[0]}x{expected[1]}")
        path = self.cache / f"{key}.{'jpg' if form == 'JPEG' else 'png'}"
        self.cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)  # the provider's bytes, not re-encoded
        record = {"request": request, "file": path.name, "image_sha256": file_hash(path), "image_size": size,
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(), "provider_docs": "https://docs.mapbox.com/api/maps/static-images/"}
        write_json(sidecar, record)
        return path, record


def read_network(node_csv, link_csv, node_id):
    """The node, its incident links, and every motor link by node for following roads past a link's end."""
    with open(node_csv, newline="", encoding="utf-8-sig") as stream:
        node = next((r for r in csv.DictReader(stream) if _int(r["node_id"]) == int(node_id)), None)
    require(node is not None, f"Node {node_id} is not in {node_csv}")
    with open(link_csv, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    links = [r for r in rows if int(node_id) in (_int(r["from_node_id"]), _int(r["to_node_id"]))]
    bynode = {}
    for r in rows:
        if _motor(r):
            for key in ("from_node_id", "to_node_id"):
                bynode.setdefault(_int(r[key]), []).append(r)
    return node, links, bynode


def site_name(node, links):
    if (node.get("name") or "").strip() not in ("", "nan"):
        return node["name"].strip()
    names = sorted({(l.get("name") or "").strip() for l in links} - {"", "nan"})
    return " x ".join(names[:2]) or f"Network node {node['node_id']}"


def leg_line(link, node_id, center):
    """Link geometry in local metres (east, north) from the node, walking away from the junction."""
    points = [local_xy(lon, lat, center) for lon, lat in parse_linestring(link["geometry"])]
    return points[::-1] if _int(link["to_node_id"]) == int(node_id) else points


def _length(line):
    return sum(math.dist(a, b) for a, b in zip(line, line[1:]))


def road_beyond(link, node_id, center, bynode, minimum=140., max_turn=35.):
    """The leg walked away from the junction, continued through the following nodes along the most nearly
    straight motor link, until it is `minimum` metres long or the road ends or turns. Network links are often
    shorter than the far camera distance."""
    line = leg_line(link, node_id, center)
    previous, far = {_int(link["link_id"])}, _int(link["from_node_id"] if _int(link["to_node_id"]) == int(node_id) else link["to_node_id"])
    visited = {int(node_id)}
    while _length(line) < minimum and far not in visited:
        visited.add(far)
        direction = _unit(line[-2], line[-1])
        best = None
        for other in bynode.get(far, []):
            if _int(other["link_id"]) in previous:
                continue
            ends = (_int(other["from_node_id"]), _int(other["to_node_id"]))
            if set(ends) & visited - {far}:
                continue  # the reverse link of a road already walked
            points = leg_line(other, far, center)
            turn = math.degrees(math.acos(max(-1., min(1., sum(a * b for a, b in zip(direction, _unit(points[0], _along(points, 10.))))))))
            if turn <= max_turn and (best is None or turn < best[0]):
                best = (turn, other, points)
        if best is None:
            break
        _, other, points = best
        line = line + points[1:]
        previous.add(_int(other["link_id"]))
        far = _int(other["to_node_id"] if _int(other["from_node_id"]) == far else other["from_node_id"])
    return line


def offset_from(line, point, extend=100.):
    """Distance from `point` to the polyline, continued straight `extend` metres past its end, and the signed
    side (positive right of the walking direction)."""
    tail = _unit(line[-2], line[-1])
    line = list(line) + [(line[-1][0] + tail[0] * extend, line[-1][1] + tail[1] * extend)]
    best = None
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        size = math.hypot(dx, dy)
        if not size:
            continue
        t = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / size ** 2))
        cx, cy = a[0] + dx * t, a[1] + dy * t
        distance = math.hypot(point[0] - cx, point[1] - cy)
        if best is None or distance < best[0]:
            best = (distance, ((point[0] - cx) * dy - (point[1] - cy) * dx) / size)
    return best


def plan_site(node_csv, link_csv, node_id, policy=None):
    """Satellite viewport, sections and every camera request, without contacting a service."""
    p = policy_with(policy)
    node, links, bynode = read_network(node_csv, link_csv, node_id)
    center = (float(node["x_coord"]), float(node["y_coord"]))
    s = p["satellite"]
    viewport = mapbox_viewport(center, s["zoom"], s["size"], s["scale"], s["tile_size"])
    mpp = viewport.ground_mpp(center[1])
    specs = frames_from_network(int(node_id), center, links, viewport.to_pixel, mpp, (viewport.width, viewport.height),
                                {"driving_side": p["driving_side"]})
    bylink = {_int(l["link_id"]): l for l in links}
    legs = []
    for spec in specs:
        link = bylink[spec["link_id"]]
        line = road_beyond(link, node_id, center, bynode)
        legs.append({"section_id": spec["id"], "kind": spec["kind"], "direction": spec["direction"], "heading_deg": spec["heading_deg"],
                     "arm": spec["arm"], "link_id": spec["link_id"], "road_name": link.get("name"), "line": line,
                     "unit": _unit(_along(line, 25.), _along(line, 55.)), "shared": shares_axis(link, links), "lanes": _lanes(link)})
    for leg in legs:
        leg["requests"] = {pos: [request_for(leg, pos, band["target_m"] * (1 + f), band["target_m"], p, center)
                                 for f in [0.] + list(p["fallback_fractions"])] for pos, band in p["positions"].items()}
    approaches = [l for l in legs if l["kind"] == "inbound_stopbar"]
    exits = [l for l in legs if l["kind"] == "outbound_receiving"]
    bands = len(p["positions"])
    budget = {"satellite_requests": 1,
              "street_view_metadata": {"typical": bands * len(legs), "upper_bound": bands * len(legs) * (1 + len(p["fallback_fractions"]))},
              "street_view_images": {"upper_bound": len(approaches) * (bands + 1) + len(exits) * bands * 2},
              "note": "Street View metadata requests are not billed; images and the satellite image are."}
    return {"node_id": int(node_id), "name": site_name(node, links), "center_lonlat": list(center), "viewport": viewport,
            "ground_mpp": mpp, "specs": specs, "legs": legs, "policy": p, "budget": budget,
            "network": {"node_csv": str(node_csv), "link_csv": str(link_csv)}}


class TileMosaicClient:
    """A dated orthophoto assembled from a Web Mercator tile service over the viewport a Mapbox request would cover.

    The tiles are fetched at `level`, mosaicked, cropped to the viewport and resampled to the Mapbox pixel size, so the
    georeference of the site is unchanged. The source and its capture date are recorded with the image."""

    def __init__(self, root, cache, cache_only=False, session=None, max_requests=1):
        self.root, self.cache = Path(root), Path(cache).resolve()
        self.cache_only, self.session, self.max_requests = cache_only, session or requests.Session(), max_requests
        self.counts = {"satellite_requests": 0, "satellite_cache_hits": 0, "tile_requests": 0}

    def fetch(self, center, spec):
        viewport = mapbox_viewport(center, spec["zoom"], spec["size"], spec["scale"], spec["tile_size"])
        request = {"url": spec["url"], "level": spec["level"], "bounds": [round(v, 3) for v in viewport.mercator_bounds],
                   "size": [viewport.width, viewport.height]}
        key = digest(request)
        sidecar = self.cache / f"{key}.json"
        if sidecar.exists():
            record = read_json(sidecar)
            path = self.cache / record["file"]
            if path.exists() and file_hash(path) == record["image_sha256"]:
                self.counts["satellite_cache_hits"] += 1
                return path, record
        if self.cache_only:
            raise AcquisitionUnavailable("satellite_cache_miss")
        if self.counts["satellite_requests"] >= self.max_requests:
            raise AcquisitionUnavailable("satellite_budget_exhausted")
        self.counts["satellite_requests"] += 1
        res = WORLD / 256 / 2 ** spec["level"]
        half = WORLD / 2
        x0, y0, x1, y1 = viewport.mercator_bounds
        cols = range(int((x0 + half) // (res * 256)), int((x1 + half) // (res * 256)) + 1)
        rows = range(int((half - y1) // (res * 256)), int((half - y0) // (res * 256)) + 1)
        mosaic = Image.new("RGB", (len(cols) * 256, len(rows) * 256))
        for i, col in enumerate(cols):
            for j, row in enumerate(rows):
                self.counts["tile_requests"] += 1
                try:
                    response = self.session.get(spec["url"].format(z=spec["level"], x=col, y=row), timeout=60)
                except requests.RequestException:
                    raise AcquisitionUnavailable("satellite_transport_error") from None
                if response.status_code != 200:
                    raise AcquisitionUnavailable(f"satellite_tile_HTTP_{response.status_code}")
                try:
                    with Image.open(io.BytesIO(response.content)) as tile:
                        mosaic.paste(tile.convert("RGB"), (i * 256, j * 256))
                except (OSError, ValueError):
                    raise AcquisitionUnavailable("invalid_satellite_tile") from None
        left, top = (x0 + half) / res - cols[0] * 256, (half - y1) / res - rows[0] * 256
        crop = mosaic.crop((left, top, left + (x1 - x0) / res, top + (y1 - y0) / res))
        image = crop.resize((viewport.width, viewport.height), Image.LANCZOS)
        path = self.cache / f"{key}.jpg"
        self.cache.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=92)
        record = {"request": request, "file": path.name, "image_sha256": file_hash(path), "image_size": [viewport.width, viewport.height],
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(), "provider": spec.get("provider"),
                  "capture_date": spec.get("capture_date"), "date_note": spec.get("date_note"), "attribution": spec.get("attribution"),
                  "processing": f"tiles at level {spec['level']} ({res:.3f} m/px Web Mercator) mosaicked, cropped to the viewport, resampled"}
        write_json(sidecar, record)
        return path, record


def satellite_client(p, root, cache, cache_only=False):
    """The client the policy's satellite source calls for."""
    source = p["satellite"].get("source", "mapbox")
    require(source in ("mapbox", "xyz_tiles"), f"Unknown satellite source {source}")
    client = TileMosaicClient if source == "xyz_tiles" else MapboxClient
    return client(root, cache, cache_only=cache_only, max_requests=p["max_satellite_requests"])


def before(meta, cutoff):
    """Whether a panorama was captured before the cutoff month (an undated panorama is not)."""
    return not cutoff or bool(meta.get("capture_date")) and meta["capture_date"] < cutoff


def wanted(meta, p):
    """Whether a panorama meets the policy's date cutoff and source rule."""
    return before(meta, p["capture_before"]) and (official(meta) or not p["official_only"])


def earlier_panorama(street, leg, request, p, center, taken, direct=None):
    """The usable panorama meeting the date cutoff and source rule nearest the requested position, searched on the
    points around it; (metadata, assessment, search record), or (None, None, record) when there is none.

    With p["fallback_earliest"], a position with no usable earlier panorama takes the earliest usable one found
    there instead (the panorama the direct query returned included), and the assessment says so."""
    search = p["date_search"]
    h = math.radians(request["travel_heading_deg"])
    points = [local_lonlat(*q, center) for q in probe_points(request["requested_local_xy"], (math.sin(h), math.cos(h)), search)]
    nearby = panoramas_near(street, request, points, search["radius_m"], official_only=p["official_only"])
    if direct and direct.get("status") == "OK" and all(m["pano_id"] != direct["pano_id"] for m in nearby):
        nearby.append(direct)
    assessed = [(m, c) for m in nearby if official(m) or not p["official_only"] for c in [assess(leg, request, m, p, center, taken)] if c["usable"]]
    usable = [(c["request_snap_distance_m"], m, c) for m, c in assessed if before(m, p["capture_before"])]
    record = {"probes": len(points), "earlier_panoramas": sum(before(m, p["capture_before"]) for m in nearby), "usable": len(usable)}
    if usable:
        _, meta, check = min(usable, key=lambda x: x[0])
        return meta, check, record
    if p["fallback_earliest"] and assessed:
        meta, check = min(assessed, key=lambda x: (x[0].get("capture_date") or "9999", x[1]["request_snap_distance_m"]))
        record["date_fallback"] = meta.get("capture_date")
        return meta, {**check, "date_fallback": True}, record
    return None, None, record


def request_for(leg, position, distance, target, p, center):
    point, beyond = _at_projection(leg["line"], (0., 0.), leg["unit"], distance)
    a = _at_projection(leg["line"], (0., 0.), leg["unit"], max(distance - 5., 0.))[0]
    b = _at_projection(leg["line"], (0., 0.), leg["unit"], distance + 5.)[0]
    away = _unit(a, b)
    travel = away if leg["kind"] == "outbound_receiving" else (-away[0], -away[1])
    right = (travel[1], -travel[0])
    # A two-way link runs down the road axis; its directional carriageway lies to the driving side.
    shift = (1 if p["driving_side"] == "right" else -1) * leg["lanes"] * p["lane_width_prior_m"] / 2 if leg["shared"] else 0.
    xy = [point[0] + shift * right[0], point[1] + shift * right[1]]
    return {"id": f"{leg['section_id']}_{position}_{distance:.0f}m", "section_id": leg["section_id"], "direction": leg["direction"],
            "position": position, "target_distance_m": target, "requested_distance_m": round(distance, 1), "requested_local_xy": xy,
            "requested_lonlat": list(local_lonlat(*xy, center)), "radius_m": p["metadata_radius_m"], "travel_heading_deg": heading(travel),
            "extrapolated_beyond_link": beyond}


def assess(leg, request, meta, p, center, taken):
    """Whether a snapped panorama can stand for this leg's band, and the pose facts recorded with it."""
    if meta.get("status") != "OK":
        return {"usable": False, "reasons": [meta.get("status", "metadata_error")]}
    lon, lat = meta["actual_lonlat"]
    xy = local_xy(lon, lat, center)
    north, east = offsets(lat, lon, (center[1], center[0]))
    distance = math.hypot(north, east)
    low, high = p["positions"][request["position"]]["allowed_distance_m"]
    along = xy[0] * leg["unit"][0] + xy[1] * leg["unit"][1]
    off, side = offset_from(leg["line"], xy)
    # Walking away from the node, the right side is the travel-right side of an exit and the left side of an approach.
    right_of_travel = side if leg["kind"] == "outbound_receiving" else -side
    reasons = []
    if along <= 0:
        reasons.append("not_on_this_leg_side_of_the_junction")
    if not low <= distance <= high:
        reasons.append("outside_distance_band")
    if math.dist(xy, request["requested_local_xy"]) > p["max_snap_distance_m"]:
        reasons.append("excessive_snap_distance")
    if off > p["max_offset_from_leg_m"]:
        reasons.append("off_the_leg")
    if any(t["pano_id"] == meta["pano_id"] for t in taken):
        reasons.append("duplicate_panorama")
    if any(math.dist(xy, t["local_xy"]) < p["minimum_position_separation_m"] for t in taken):
        reasons.append("too_close_to_another_band")
    nominal = right_of_travel > 1. if leg["shared"] else off < 6.
    if p["same_carriageway"] and not leg["shared"] and not nominal:
        reasons.append("opposite_carriageway")
    return {"usable": not reasons, "reasons": reasons, "local_xy": list(xy), "distance_to_center_m": round(distance, 2),
            "true_distance_to_center_m": math.hypot(*xy), "along_leg_m": along, "right_of_centerline_m": right_of_travel,
            "offset_from_link_m": off, "request_snap_distance_m": math.dist(xy, request["requested_local_xy"]),
            "nominal_travel_side": nominal}


def travel_heading_at(leg, xy):
    """Direction of travel on the leg at the camera's position along it."""
    station = max(xy[0] * leg["unit"][0] + xy[1] * leg["unit"][1], 0.)
    a = _at_projection(leg["line"], (0., 0.), leg["unit"], max(station - 5., 0.))[0]
    b = _at_projection(leg["line"], (0., 0.), leg["unit"], station + 5.)[0]
    away = _unit(a, b)
    return heading(away if leg["kind"] == "outbound_receiving" else (-away[0], -away[1]))


def acquire(plan, street, satellite=None):
    """Resolve and fetch the planned imagery. Returns the satellite record, accepted cameras, views and missing bands."""
    p, center = plan["policy"], tuple(plan["center_lonlat"])
    result = {"satellite": None, "cameras": [], "views": [], "missing": [], "attempts": [], "stopped": None}
    if satellite is not None:
        path, record = satellite.fetch(center, p["satellite"])
        result["satellite"] = {"path": str(path), **record}
    for leg in plan["legs"]:
        taken = []
        for position, candidates in leg["requests"].items():
            accepted = None
            pinned = p["pinned_panoramas"].get(f"{leg['section_id']}/{position}")
            if pinned:  # one request: the named panorama, checked against the band's first requested position
                candidates = [{**candidates[0], "pano_id": pinned}]
            for request in candidates:
                if result["stopped"]:
                    break
                try:
                    meta = street.metadata(request)
                except AcquisitionUnavailable as error:
                    result["attempts"].append({"request": request, "status": str(error)})
                    if str(error) in STOP:
                        result["stopped"] = str(error)
                    break
                check = assess(leg, request, meta, p, center, taken)
                search = None
                if pinned:
                    require(check["usable"], f"pinned panorama {pinned} for {leg['section_id']}/{position} does not fit: "
                            f"{meta.get('status')} {check.get('reasons')}")
                    check = {**check, "pinned": True, **({"date_fallback": True} if not wanted(meta, p) else {})}
                elif meta.get("status") == "OK" and not wanted(meta, p):
                    try:
                        found, found_check, search = earlier_panorama(street, leg, request, p, center, taken, direct=meta)
                    except AcquisitionUnavailable as error:
                        result["attempts"].append({"request": request, "metadata": meta, "status": str(error)})
                        if str(error) in STOP:
                            result["stopped"] = str(error)
                        break
                    if found:
                        meta, check = found, found_check
                    else:
                        reason = f"captured_on_or_after_{p['capture_before']}" if not before(meta, p["capture_before"]) else "contributed_panorama"
                        check = {**check, "usable": False, "reasons": check.get("reasons", []) + [reason]}
                result["attempts"].append({"request": request, "metadata": meta, "assessment": check, "date_search": search,
                                           "status": "accepted" if check["usable"] else "rejected"})
                if check["usable"]:
                    accepted = {**check, "pano_id": meta["pano_id"], "actual_lonlat": meta["actual_lonlat"],
                                "capture_date": meta.get("capture_date"), "copyright": meta.get("copyright"),
                                "section_id": leg["section_id"], "direction": leg["direction"], "kind": leg["kind"],
                                "position": position, "target_distance_m": request["target_distance_m"],
                                "travel_heading_deg": travel_heading_at(leg, check["local_xy"])}
                    break
            if accepted:
                taken.append(accepted)
                result["cameras"].append(accepted)
            else:
                result["missing"].append({"section_id": leg["section_id"], "direction": leg["direction"], "position": position,
                                          "reason": result["stopped"] or "no_acceptable_panorama"})
    for camera in result["cameras"]:
        for role, turn, settings in views_for(camera, p):
            if result["stopped"] in ("image_budget_exhausted", "missing_streetview_key"):
                result["missing"].append({"section_id": camera["section_id"], "position": camera["position"], "view_role": role,
                                          "reason": result["stopped"]})
                continue
            pose = {"pano_id": camera["pano_id"], "compass_heading_deg": round((camera["travel_heading_deg"] + turn) % 360, 3),
                    "pitch_deg": settings["pitch"], "hfov_deg": settings["fov"]}
            try:
                image = street.image(pose)
            except AcquisitionUnavailable as error:
                result["stopped"] = str(error) if str(error) in STOP else result["stopped"]
                result["missing"].append({"section_id": camera["section_id"], "position": camera["position"], "view_role": role,
                                          "reason": str(error)})
                continue
            result["views"].append({**image, "camera": camera, "view_role": role})
    result["counts"] = {**street.counts, **(satellite.counts if satellite else {})}
    return result


def views_for(camera, p):
    """(role, heading offset from travel, view settings) for one accepted camera."""
    if camera["kind"] == "outbound_receiving":
        return [("away", 0., p["exit_view"]), ("toward", 180., p["exit_view"])]
    views = [("forward", 0., p["approach_view"][camera["position"]])]
    if camera["position"] == "near":
        views.append(("look_back", 180., p["look_back_view"]))
    return views


def relative(root, path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def site_files(plan, result, root):
    """The tracing site config, the base classification config and the exit-view manifest."""
    p, node_id = plan["policy"], plan["node_id"]
    center = plan["center_lonlat"]
    network = {k: relative(root, v) for k, v in plan["network"].items()}
    georeference = {"kind": "viewport", "viewport": plan["viewport"].serialize(), "ground_mpp": plan["ground_mpp"]}
    satellite = relative(root, result["satellite"]["path"]) if result["satellite"] else None
    site = {"schema": "autoloop-site-1", "site_id": str(node_id), "node_id": node_id, "name": plan["name"],
            "satellite": {"path": satellite, "georeference": georeference}, "network": network,
            "models": {"draft": {"preset": "default", "max_tokens": 6144, "temperature": 0.0},
                       "review": {"preset": "default", "max_tokens": 6144, "temperature": 0.0}},
            # Six calls per section: a draft, one wider redraft and up to three reviews, with room for a repair.
            "policy": {"max_rounds": 3, "max_vlm_calls": 6 * sum(s["usable"] for s in plan["specs"])},
            "reference": {"status": "withheld_from_inference", "file": None}}
    sat = result["satellite"] or {}
    sources = {"satellite": {"kind": "image", "path": satellite, "role": "satellite", "capture_date": sat.get("capture_date"),
                             "date_status": "provider_reported" if sat.get("capture_date") else "not_retained",
                             **{k: sat[k] for k in ("provider", "date_note", "attribution") if sat.get(k)}, "georeference": georeference}}
    context_views, exit_views = [], []
    for view in result["views"]:
        c = view["camera"]
        record = {"pano_id": c["pano_id"], "actual_lat": c["actual_lonlat"][1], "actual_lon": c["actual_lonlat"][0],
                  "capture_date": c["capture_date"], "compass_heading_deg": view["compass_heading_deg"], "path": relative(root, view["path"]),
                  "image_sha256": view["image_sha256"], "image_origin": view["image_origin"], "request_snap_distance_m": c["request_snap_distance_m"],
                  "copyright": c["copyright"]}
        if c["kind"] == "inbound_stopbar":
            sid = f"gsv_{c['direction']}_{c['position']}_{view['view_role']}"
            back = view["view_role"] == "look_back"
            sources[sid] = {"kind": "image", "role": "gsv", "direction": c["direction"], "reverse_view": back,
                            "camera_role": "reverse_auxiliary" if back else "forward_primary", "sampling_position": c["position"],
                            "target_distance_m": c["target_distance_m"], "distance_to_center_m": c["distance_to_center_m"],
                            "view_settings": {"pitch": view["pitch_deg"], "fov": view["hfov_deg"], "size": view["image_size"]}, **record}
            if not back:
                context_views.append({"id": f"gsv_context_{c['direction']}_{c['position']}", "source_id": sid,
                                      "kind": "gsv_forward_context", "direction": c["direction"], "sampling_position": c["position"]})
        else:
            exit_views.append({"id": f"gsv_{c['section_id']}_{c['target_distance_m']:03d}_{view['view_role']}", "target_section": c["section_id"],
                               "direction": c["direction"], "sampling_position": c["position"], "target_distance_m": c["target_distance_m"],
                               "view_role": view["view_role"], "hfov_deg": view["hfov_deg"], "pitch_deg": view["pitch_deg"],
                               "image_size": view["image_size"], "actual_lonlat": c["actual_lonlat"],
                               "distance_to_center_m": c["true_distance_to_center_m"],
                               "carriageway_status": "nominal_outgoing_side" if c["nominal_travel_side"] else "opposite_or_unresolved",
                               "along_outbound_m": c["along_leg_m"], "right_of_centerline_m": c["right_of_centerline_m"],
                               "camera_position_error_m": None, **{k: v for k, v in record.items() if k not in ("actual_lat", "actual_lon")}})
    missing = [{"direction": m["direction"], "position": m["position"], "reason": m["reason"]} for m in result["missing"]
               if m["section_id"].startswith("in_") and "view_role" not in m]
    base = {"schema_version": 1, "pilot_id": f"site_{node_id}", "node_id": node_id, "name": plan["name"],
            "network": {"link_csv": network["link_csv"], "node_csv": network["node_csv"]},
            "vlm": {"preset": "default", "max_tokens": 6144, "temperature": 0.0}, "yolo": dict(YOLO), "inference_mode": "visual_evidence",
            "sources": sources, "sections": [], "gsv_views": [], "context_views": context_views,
            "gsv_sampling": {"center_lat_lon": [center[1], center[0]], "positions": p["positions"],
                             "minimum_position_separation_m": p["minimum_position_separation_m"], "missing": missing},
            "geometry_checks": dict(GEOMETRY_CHECKS), "reference": {"status": "withheld_from_inference", "file": None},
            "far_upstream": {"status": "context_views_only_not_full_lane_crosssections"}}
    manifest = {"schema": "downstream-views-1", "origin_lonlat": center, "acquisition_mode": "official_location_query",
                "positions": [c for c in result["cameras"] if c["kind"] == "outbound_receiving"],
                "metadata_attempts": [{"status": a.get("metadata", {}).get("status", a["status"]), "request_id": a["request"]["id"]}
                                      for a in result["attempts"] if a["request"]["section_id"].startswith("out_")],
                "views": exit_views, "unique_panoramas": len({v["pano_id"] for v in exit_views}),
                "note": "Pose side is based on metadata and a nominal road axis, not surveyed lane localization. Two headings of a panorama are one original source."}
    return site, base, manifest
