"""A street view beside the satellite crop while lanes are traced.

The satellite crop and a forward street view of the same approach are tied together by the camera's recorded
position and heading (street-view metadata, in the same coordinates as the satellite georeference): the camera
is marked on the crop with its field of view, and the boundaries being traced are projected onto the street view
over flat ground at an assumed camera height. Neither pose is calibrated, so the projection can sit up to about a
lane to the side; it shows which painted line a boundary stands for, not where it is to the centimetre.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from ..hybrid.common import resource_path
from ..hybrid.geometry import font
from . import render

HEIGHT = 2.5          # assumed camera height, metres
NEAR = 1.5            # ground closer than this to the camera is left out
CAMERA, CONE = "#ffd400", (255, 212, 0, 70)
EDGE_COLORS = ("#16d5ec", "#ff8a3d", "#a3e635", "#f472b6", "#facc15", "#60a5fa", "#fb7185", "#34d399")


def pick_view(base_config, direction, positions=("near", "mid")):
    """The forward street view of an approach closest to the junction that was acquired, or None."""
    sources = (base_config or {}).get("sources", {})
    for position in positions:
        for sid, source in sources.items():
            if (source.get("role") == "gsv" and source.get("direction") == direction and not source.get("reverse_view")
                    and source.get("sampling_position") == position and source.get("actual_lat") is not None):
                return {"id": sid, **source}
    return None


def load(view, root):
    image = Image.open(resource_path(root, view["path"])).convert("RGB")
    size = view.get("view_settings", {}).get("size") or list(image.size)
    return image if list(image.size) == list(size) else image.resize(tuple(size))


def camera_on_crop(view, frame, to_pixel):
    """Camera position in crop coordinates, and its heading relative to the crop's up direction (degrees)."""
    px = to_pixel(view["actual_lon"], view["actual_lat"])
    x, y = frame.to_local(*px)
    return (x, y), (view["compass_heading_deg"] - frame.heading_deg) % 360, px


def mark_camera(canvas, at, relative_heading, hfov, scale, label="C"):
    """Draw the camera and its field of view on a ruled crop (render.ruled layout: margins LEFT/TOP)."""
    x, y = render.LEFT + at[0] * scale, render.TOP + at[1] * scale
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    reach = 140
    a0, a1 = math.radians(relative_heading - hfov / 2), math.radians(relative_heading + hfov / 2)
    d.polygon([(x, y), (x + reach * math.sin(a0), y - reach * math.cos(a0)), (x + reach * math.sin(a1), y - reach * math.cos(a1))], fill=CONE)
    out = Image.alpha_composite(canvas.convert("RGBA"), layer)
    d = ImageDraw.Draw(out)
    d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=CAMERA, outline="black", width=2)
    d.text((x + 13, y), label, font=font(20), fill=CAMERA, stroke_width=3, stroke_fill="black", anchor="lm")
    return out.convert("RGB")


def project(points_px, camera_px, view, mpp, height=HEIGHT):
    """Source-image ground points as street-view pixels (None for points behind or too close to the camera)."""
    w, h = view.get("view_settings", {}).get("size") or (640, 640)
    fov = math.radians(view.get("view_settings", {}).get("fov", 90))
    pitch = math.radians(view.get("view_settings", {}).get("pitch", 0))
    heading = math.radians(view["compass_heading_deg"])
    f = w / 2 / math.tan(fov / 2)
    out = []
    for sx, sy in points_px:
        east, north = (sx - camera_px[0]) * mpp, -(sy - camera_px[1]) * mpp
        ahead = east * math.sin(heading) + north * math.cos(heading)
        right = east * math.cos(heading) - north * math.sin(heading)
        up, forward = -ahead * math.sin(pitch) - height * math.cos(pitch), ahead * math.cos(pitch) - height * math.sin(pitch)
        out.append(None if forward < NEAR else (w / 2 + f * right / forward, h / 2 - f * up / forward))
    return out


def projected_edges(image, view, frame, strips, camera_px, mpp, extend=40):
    """The street view with each traced boundary e1..eN drawn where it falls on the road, labelled at its near end.

    Boundaries are extended `extend` crop pixels past both ends of the section so they reach into the view."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    stations = strips["stations"]
    for i, xs in enumerate(strips["edges"]):
        color = EDGE_COLORS[i % len(EDGE_COLORS)]
        ys = [stations[0] - extend] + list(stations) + [stations[-1] + extend]
        xs = [xs[0]] + list(xs) + [xs[-1]]
        points = project([frame.to_source(x, y) for x, y in zip(xs, ys)], camera_px, view, mpp)
        seen = [p for p in points if p is not None and -200 < p[0] < out.width + 200 and -200 < p[1] < out.height + 200]
        if len(seen) < 2:
            continue
        render._dashed(d, seen, color, on=14, off=8, width=3)
        lowest = max(seen, key=lambda p: p[1])
        x, y = min(max(lowest[0], 18), out.width - 18), min(max(lowest[1] - 16, 16), out.height - 16)
        d.text((x, y), f"e{i + 1}", font=font(20), fill=color, stroke_width=3, stroke_fill="black", anchor="mm")
    return out
