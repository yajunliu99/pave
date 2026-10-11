"""Mapbox Static Images client: one top-down satellite image per intersection cluster.

A cluster is the set of signal/non-signal GMNS nodes that together represent one
physical intersection. osm2gmns routinely expands a wide or dual-carriageway
intersection into 3-6 nodes forming a "box"; the VLM audit operates on the
physical intersection, so we fetch ONE image per cluster that covers the whole
box plus a configurable margin.

Content-hash caching: repeated calls with the same bbox + size + style hit the
local cache and cost zero API calls. The cache is idempotent and safe to commit
under `data/cache/` if you want reproducible pipeline runs.

Usage:
    from movement_fixer.satellite_client import (
        MapboxSatelliteClient, NodeCoord, MapboxUnavailable,
    )

    client = MapboxSatelliteClient(
        access_token="pk....",
        cache_dir=Path("./cache/mapbox"),
    )
    view = client.fetch_cluster([
        NodeCoord(11, 33.4295773, -111.939963),
        NodeCoord(12, 33.4296814, -111.939967),
        NodeCoord(24, 33.4296678, -111.940091),
        NodeCoord(44, 33.4295692, -111.940087),
    ])
    # view.image_bytes  -> JPEG bytes, ready to feed a VLM
    # view.image_path   -> cached path on disk
    # view.bbox         -> (min_lon, min_lat, max_lon, max_lat)

CLI smoke test:
    python -m movement_fixer.satellite_client \\
        --token "$MAPBOX_TOKEN" \\
        --node-csv "data/Tempe Rio Salado Pkwy/output/node.csv" \\
        --cluster 11,12,24,44 \\
        --out ./tempe_mill_riosalado.jpg
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from math import cos, radians
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests


log = logging.getLogger(__name__)


_STATIC_BASE = "https://api.mapbox.com/styles/v1"
_DEFAULT_STYLE = "mapbox/satellite-v9"
_METERS_PER_LAT_DEG = 111_000.0
_MAX_DIM = 1280   # Mapbox Static Images limit per dimension


class MapboxUnavailable(RuntimeError):
    """Raised when Mapbox returns a non-recoverable error for a cluster."""


@dataclass(frozen=True)
class NodeCoord:
    """One GMNS node's geographic location. Used as input to cluster fetch."""
    node_id: int
    lat: float
    lon: float


@dataclass
class SatelliteImage:
    """One top-down image covering an intersection cluster.

    `width` / `height` are the **effective** pixel counts (doubled when retina=True).
    `bbox` is (min_lon, min_lat, max_lon, max_lat) — the requested fit bounds
    after buffering, NOT a verified rendered viewport. Aspect fitting, padding
    and service zoom behavior must be checked before pixel georeferencing.
    """
    cluster_id: str
    node_ids: tuple[int, ...]
    bbox: tuple[float, float, float, float]
    center_lat: float
    center_lon: float
    width: int
    height: int
    style: str
    buffer_meters: float
    image_bytes: bytes
    image_path: Path
    cache_hit: bool

    def to_meta_dict(self) -> dict:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        d["image_bytes_len"] = len(self.image_bytes)
        del d["image_bytes"]
        d["bbox_semantics"] = "requested_fit_bounds_not_verified_rendered_extent"
        d["rendered_viewport_status"] = "unverified"
        return d


class MapboxSatelliteClient:
    """Fetches one top-down satellite image per intersection cluster.

    Args:
        access_token:    Mapbox token (pk.* for public, sk.* for secret). Required.
        cache_dir:       Directory for cached JPEGs + JSON metadata.
        buffer_meters:   Margin around cluster bounding box. Default 40m — tight
                         enough for a single intersection, loose enough to capture
                         the first few meters of each approach's pavement markings.
        image_size:      (width, height) in pixels, each ≤ 1280.
        style:           Mapbox style, default satellite-v9.
        retina:          If True (default), appends @2x for 2× pixel density.
        max_retries:     Exponential-backoff retries for 429/5xx and network errors.
        timeout_seconds: Per-request timeout.
        session:         Optional requests.Session for connection reuse.
    """

    def __init__(
        self,
        access_token: str,
        cache_dir: Path,
        buffer_meters: float = 40.0,
        image_size: tuple[int, int] = (1280, 1280),
        style: str = _DEFAULT_STYLE,
        retina: bool = True,
        max_retries: int = 3,
        timeout_seconds: float = 20.0,
        session: Optional[requests.Session] = None,
    ):
        if not access_token or not (access_token.startswith("pk.") or access_token.startswith("sk.")):
            raise ValueError("access_token must be a Mapbox token (pk.* or sk.*)")
        w, h = image_size
        if not (1 <= w <= _MAX_DIM and 1 <= h <= _MAX_DIM):
            raise ValueError(f"image_size must be within 1..{_MAX_DIM} per dimension, got {image_size}")
        if buffer_meters < 0:
            raise ValueError("buffer_meters must be non-negative")

        self.access_token = access_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_meters = float(buffer_meters)
        self._width_req, self._height_req = image_size
        self.style = style
        self.retina = retina
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    # ---------- public API ----------

    def fetch_cluster(
        self,
        nodes: Iterable[NodeCoord],
        cluster_id: Optional[str] = None,
    ) -> SatelliteImage:
        """Fetch (or load from cache) one image covering all given nodes."""
        node_list = list(nodes)
        if not node_list:
            raise ValueError("cluster must contain at least one node")

        bbox = self._bbox_with_buffer(node_list)
        cluster_id = cluster_id or self._auto_cluster_id(node_list)

        cache_key = self._cache_key(bbox)
        cached = self._load_cache(cache_key)
        if cached is not None:
            log.info("cache hit: cluster=%s key=%s", cluster_id, cache_key)
            return self._build_image(
                cluster_id, node_list, bbox, cached, image_path=self._image_path(cache_key), cache_hit=True
            )

        image_bytes = self._request_with_retry(bbox)
        image_path = self._save_cache(cache_key, image_bytes, bbox, cluster_id, node_list)
        log.info("cache miss: cluster=%s key=%s bytes=%d", cluster_id, cache_key, len(image_bytes))
        return self._build_image(
            cluster_id, node_list, bbox, image_bytes, image_path=image_path, cache_hit=False
        )

    # ---------- internals ----------

    def _build_image(
        self,
        cluster_id: str,
        nodes: Sequence[NodeCoord],
        bbox: tuple[float, float, float, float],
        image_bytes: bytes,
        image_path: Path,
        cache_hit: bool,
    ) -> SatelliteImage:
        center_lat = 0.5 * (bbox[1] + bbox[3])
        center_lon = 0.5 * (bbox[0] + bbox[2])
        scale = 2 if self.retina else 1
        return SatelliteImage(
            cluster_id=cluster_id,
            node_ids=tuple(n.node_id for n in nodes),
            bbox=bbox,
            center_lat=center_lat,
            center_lon=center_lon,
            width=self._width_req * scale,
            height=self._height_req * scale,
            style=self.style,
            buffer_meters=self.buffer_meters,
            image_bytes=image_bytes,
            image_path=image_path,
            cache_hit=cache_hit,
        )

    def _bbox_with_buffer(self, nodes: Sequence[NodeCoord]) -> tuple[float, float, float, float]:
        lats = [n.lat for n in nodes]
        lons = [n.lon for n in nodes]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = 0.5 * (min_lat + max_lat)

        dlat = self.buffer_meters / _METERS_PER_LAT_DEG
        meters_per_lon_deg = _METERS_PER_LAT_DEG * max(0.01, abs(cos(radians(center_lat))))
        dlon = self.buffer_meters / meters_per_lon_deg

        return (
            min_lon - dlon,
            min_lat - dlat,
            max_lon + dlon,
            max_lat + dlat,
        )

    def _build_url(self, bbox: tuple[float, float, float, float]) -> str:
        # Mapbox path parameter — brackets and commas are permitted literally.
        retina_suffix = "@2x" if self.retina else ""
        bbox_str = f"[{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}]"
        return (
            f"{_STATIC_BASE}/{self.style}/static/"
            f"{bbox_str}/{self._width_req}x{self._height_req}{retina_suffix}"
            f"?access_token={self.access_token}&attribution=false&logo=false"
        )

    def _request_with_retry(self, bbox: tuple[float, float, float, float]) -> bytes:
        url = self._build_url(bbox)
        redacted = url.replace(self.access_token, "***")
        attempt = 0
        backoff = 1.0

        while True:
            try:
                resp = self.session.get(url, timeout=self.timeout_seconds)
            except requests.RequestException as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise MapboxUnavailable(f"network error for {redacted}: {e}") from e
                log.warning("network error (attempt %d/%d): %s", attempt, self.max_retries, e)
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "")
                if not ctype.startswith("image/"):
                    raise MapboxUnavailable(
                        f"unexpected content-type {ctype!r} for {redacted}: {resp.text[:300]}"
                    )
                return resp.content

            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > self.max_retries:
                    raise MapboxUnavailable(
                        f"mapbox {resp.status_code} after {attempt} attempts for {redacted}"
                    )
                log.warning(
                    "mapbox %d (attempt %d/%d); backing off %.1fs",
                    resp.status_code, attempt, self.max_retries, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue

            # 401/403/422 etc. — do not retry.
            raise MapboxUnavailable(
                f"mapbox {resp.status_code} for {redacted}: {resp.text[:300]}"
            )

    # ---------- cache ----------

    def _cache_key(self, bbox: tuple[float, float, float, float]) -> str:
        payload = json.dumps(
            {
                "bbox":   [round(b, 6) for b in bbox],
                "size":   [self._width_req, self._height_req],
                "style":  self.style,
                "retina": self.retina,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _image_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.jpg"

    def _meta_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Optional[bytes]:
        p = self._image_path(cache_key)
        if p.exists() and p.stat().st_size > 0:
            return p.read_bytes()
        return None

    def _save_cache(
        self,
        cache_key: str,
        image_bytes: bytes,
        bbox: tuple[float, float, float, float],
        cluster_id: str,
        nodes: Sequence[NodeCoord],
    ) -> Path:
        img_path = self._image_path(cache_key)
        img_path.write_bytes(image_bytes)
        meta = {
            "cluster_id":    cluster_id,
            "node_ids":      [n.node_id for n in nodes],
            "node_coords":   [{"node_id": n.node_id, "lat": n.lat, "lon": n.lon} for n in nodes],
            "bbox":          list(bbox),
            "style":         self.style,
            "size":          [self._width_req, self._height_req],
            "retina":        self.retina,
            "buffer_meters": self.buffer_meters,
            "metadata_version": 2,
            "bbox_semantics": "requested_fit_bounds_not_verified_rendered_extent",
            "request_bbox_rounded": [float(f"{x:.6f}") for x in bbox],
            "request_padding": None,
            "rendered_viewport_status": "unverified",
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
        self._meta_path(cache_key).write_text(json.dumps(meta, indent=2))
        return img_path

    # ---------- helpers ----------

    @staticmethod
    def _auto_cluster_id(nodes: Sequence[NodeCoord]) -> str:
        ids = sorted(n.node_id for n in nodes)
        return "c_" + "_".join(str(i) for i in ids)


# ---------- CLI smoke test ----------

def _load_nodes_from_csv(csv_path: Path, target_ids: set[int]) -> list[NodeCoord]:
    """Read a GMNS node.csv and return NodeCoord objects for the target ids."""
    import csv
    nodes: list[NodeCoord] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nid = int(row["node_id"])
            except (KeyError, ValueError):
                continue
            if nid in target_ids:
                nodes.append(
                    NodeCoord(
                        node_id=nid,
                        lat=float(row["y_coord"]),
                        lon=float(row["x_coord"]),
                    )
                )
    return nodes


def _main() -> int:
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(
        description="Fetch a top-down satellite image for one intersection cluster."
    )
    ap.add_argument(
        "--token",
        default=os.environ.get("MAPBOX_TOKEN"),
        help="Mapbox access token (pk.*). Falls back to MAPBOX_TOKEN env var.",
    )
    ap.add_argument("--cache-dir", default="./cache/mapbox", type=Path)
    ap.add_argument(
        "--node-csv",
        default="data/Tempe Rio Salado Pkwy/output/node.csv",
        type=Path,
        help="GMNS node.csv with x_coord / y_coord columns.",
    )
    ap.add_argument(
        "--cluster",
        default="11,12,24,44",
        help="comma-separated node_ids forming one intersection cluster (default: Mill × Rio Salado)",
    )
    ap.add_argument("--buffer", type=float, default=40.0, help="buffer in meters around the cluster bbox")
    ap.add_argument("--size", default="1280x1280", help="image size WxH, each ≤ 1280")
    ap.add_argument("--no-retina", action="store_true", help="disable @2x retina")
    ap.add_argument("--style", default=_DEFAULT_STYLE)
    ap.add_argument("--out", type=Path, default=None, help="copy image to this path (jpg)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.token:
        print("ERROR: missing --token and MAPBOX_TOKEN env var", file=sys.stderr)
        return 2
    if not args.node_csv.exists():
        print(f"ERROR: node csv not found: {args.node_csv}", file=sys.stderr)
        return 2

    try:
        target_ids = {int(x) for x in args.cluster.split(",") if x.strip()}
    except ValueError:
        print(f"ERROR: --cluster must be comma-separated ints, got {args.cluster!r}", file=sys.stderr)
        return 2

    nodes = _load_nodes_from_csv(args.node_csv, target_ids)
    missing = target_ids - {n.node_id for n in nodes}
    if missing:
        print(f"ERROR: nodes not found in {args.node_csv}: {sorted(missing)}", file=sys.stderr)
        return 2

    try:
        w_str, h_str = args.size.lower().split("x")
        image_size = (int(w_str), int(h_str))
    except ValueError:
        print(f"ERROR: --size must be WxH, got {args.size!r}", file=sys.stderr)
        return 2

    client = MapboxSatelliteClient(
        access_token=args.token,
        cache_dir=args.cache_dir,
        buffer_meters=args.buffer,
        image_size=image_size,
        style=args.style,
        retina=not args.no_retina,
    )

    try:
        view = client.fetch_cluster(nodes)
    except MapboxUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"cluster_id:   {view.cluster_id}")
    print(f"nodes:        {view.node_ids}")
    print(f"bbox:         ({view.bbox[0]:.6f}, {view.bbox[1]:.6f}) - ({view.bbox[2]:.6f}, {view.bbox[3]:.6f})")
    print(f"center:       ({view.center_lat:.6f}, {view.center_lon:.6f})")
    print(f"size (px):    {view.width} x {view.height}  (retina={not args.no_retina})")
    print(f"style:        {view.style}")
    print(f"buffer:       {view.buffer_meters:.1f} m")
    print(f"cache_hit:    {view.cache_hit}")
    print(f"cached at:    {view.image_path}")
    print(f"bytes:        {len(view.image_bytes)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(view.image_bytes)
        print(f"copied to:    {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
