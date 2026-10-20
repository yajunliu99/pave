"""Satellite imagery for a site from the Mapbox Static Images API (MAPBOX_TOKEN from the environment or the env file).

The image keeps Mapbox's logo and attribution, which its terms require wherever it is shown.
`bearing` rotates the map: it is the compass direction that points up in the image.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import vision_client
from .vlm import environment

STATIC = "https://api.mapbox.com/styles/v1/{style}/static/{lon:.6f},{lat:.6f},{zoom},{bearing}/{w}x{h}{retina}"
ATTRIBUTION = "Imagery © Mapbox © OpenStreetMap contributors © Maxar"


def metres_per_pixel(lat: float, zoom: float, retina: bool) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom) / (2 if retina else 1)


def fetch_mapbox(lat: float, lon: float, out_path: Path, *, zoom: float = 17, bearing: float = 0,
                 size=(1280, 1280), retina: bool = True, style: str = "mapbox/satellite-v9",
                 env_file: Path | None = None) -> dict:
    token = environment(env_file).get("MAPBOX_TOKEN")
    if not token:
        raise RuntimeError("MAPBOX_TOKEN not found (environment, .env, or env_file)")
    url = STATIC.format(style=style, lon=lon, lat=lat, zoom=zoom, bearing=round(bearing, 2) % 360,
                        w=size[0], h=size[1], retina="@2x" if retina else "")
    req = urllib.request.Request(url + f"?access_token={token}")
    try:
        with urllib.request.urlopen(req, timeout=60, context=vision_client.ssl_context()) as r:
            data, ctype = r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:   # never echo the URL: it carries the token
        raise RuntimeError(f"Mapbox HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
    out_path = Path(out_path)
    out_path.write_bytes(data)
    meta = {"source": "Mapbox Static Images API", "style": style, "center_lat": lat, "center_lon": lon,
            "zoom": zoom, "bearing_up_deg": bearing % 360, "size_px": [size[0] * (2 if retina else 1), size[1] * (2 if retina else 1)],
            "metres_per_pixel": round(metres_per_pixel(lat, zoom, retina), 4), "content_type": ctype,
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "attribution": ATTRIBUTION,
            "request": url}   # without the token
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta
