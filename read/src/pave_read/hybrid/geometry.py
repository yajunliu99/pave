from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import require, resource_path


def rotate(image, degrees):
    if degrees == 0:
        return image.copy()
    return image.transpose({90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                            270: Image.Transpose.ROTATE_270}[degrees])


def unrotate_point(x, y, width, height, degrees):
    if degrees == 0: return x, y
    if degrees == 90: return width - 1 - y, x
    if degrees == 180: return width - 1 - x, height - 1 - y
    if degrees == 270: return y, height - 1 - x
    raise ValueError(degrees)


def perspective(pano, heading, pitch, fov, width, height):
    h, w = pano.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    focal = .5 * width / math.tan(math.radians(fov) / 2)
    cx, cy, cz = xs - width / 2, ys - height / 2, np.full_like(xs, focal, dtype=float)
    yaw, tilt = math.radians(heading), math.radians(pitch)
    ry, rz = cy * math.cos(tilt) - cz * math.sin(tilt), cy * math.sin(tilt) + cz * math.cos(tilt)
    wx, wz = cx * math.cos(yaw) + rz * math.sin(yaw), -cx * math.sin(yaw) + rz * math.cos(yaw)
    lon, lat = np.arctan2(wx, wz), np.arctan2(-ry, np.hypot(wx, wz))
    px = np.mod((lon / (2 * math.pi) + .5) * w, w).astype(int)
    py = np.clip((.5 - lat / math.pi) * h, 0, h - 1).astype(int)
    return Image.fromarray(pano[py, px])


def load_source(root, source):
    if source["kind"] == "image":
        with Image.open(resource_path(root, source["path"])) as im:
            return im.convert("RGB")
    with Image.open(resource_path(root, source["pano_path"])) as im:
        pano = np.asarray(im.convert("RGB"))
    p = source["projection"]
    return perspective(pano, p["heading"], p["pitch"], p["fov"], *p["size"])


def polygon_mask(size, polygon):
    require(all(0 <= x < size[0] and 0 <= y < size[1] for x, y in polygon), "Polygon lies outside image")
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))) / 2
    require(area > 1, "Degenerate polygon")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], fill=255)
    require(mask.getbbox() is not None, "Empty polygon")
    return mask


def bbox_union_coverage(mask, detections):
    """Coverage by union of predicted rectangles; NOT true object-mask occlusion."""
    occupied = Image.new("L", mask.size, 0)
    d = ImageDraw.Draw(occupied)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        d.rectangle((max(0, x1), max(0, y1), min(mask.width-1, x2), min(mask.height-1, y2)), fill=255)
    region = np.asarray(mask) > 0
    return float(np.count_nonzero(region & (np.asarray(occupied) > 0)) / np.count_nonzero(region))


def font(size):
    try: return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError: return ImageFont.load_default(size=size)


def render_section(image, section, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    context = image.copy(); draw = ImageDraw.Draw(context)
    crops, masks, records = [], {}, []
    for region in section["regions"]:
        mask = polygon_mask(image.size, region["polygon"])
        masks[region["id"]] = mask
        isolated = Image.composite(image, Image.new("RGB", image.size, (55, 55, 55)), mask)
        rotated_mask = rotate(mask, section.get("rotation_ccw", 0))
        box = rotated_mask.getbbox()
        crop = rotate(isolated, section.get("rotation_ccw", 0)).crop(box)
        crop.save(output / (region["id"] + "_native.png"))
        mask.save(output / (region["id"] + "_mask.png"))
        crops.append(crop)
        pts = [tuple(p) for p in region["polygon"]]
        draw.line(pts + [pts[0]], fill=section.get("color", "#00e5ff"), width=3)
        x, y = region.get("label", [sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)])
        draw.rectangle((x-13, y-13, x+20, y+14), fill="#161a22")
        draw.text((x-9, y-10), region["id"], fill="white", font=font(16))
        records.append({"region_id": region["id"], "polygon": region["polygon"],
                        "geometry_quality": region.get("quality", "unverified"), "native_size": list(crop.size),
                        "gate_point": region["gate_point"]})
    # Consistent scale within a section; small satellite strips can be enlarged.
    scale = min(2., 1100 / max(c.height for c in crops), 600 / max(c.width for c in crops))
    cell_w = max(160, int(max(c.width for c in crops)*scale) + 24)
    cell_h = int(max(c.height for c in crops)*scale) + 50
    panel = Image.new("RGB", (cell_w*len(crops), cell_h), (55,55,55)); pd = ImageDraw.Draw(panel)
    for i, (region, crop) in enumerate(zip(section["regions"], crops)):
        thumb = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))), Image.Resampling.NEAREST)
        panel.paste(thumb, (i*cell_w+(cell_w-thumb.width)//2, 45))
        pd.text((i*cell_w+10, 9), region["id"], fill="white", font=font(23))
    panel.save(output / "region_panels.png")
    context.save(output / "annotated_source.png")
    extent = np.array([p for r in section["regions"] for p in r["polygon"]])
    cropbox = (max(0,int(extent[:,0].min())-35), max(0,int(extent[:,1].min())-35),
               min(image.width,int(extent[:,0].max())+36), min(image.height,int(extent[:,1].max())+36))
    # Keep labels upright in normalized per-section context views.
    normalized=rotate(image.crop(cropbox),section.get("rotation_ccw",0))
    nd=ImageDraw.Draw(normalized); cw,ch=cropbox[2]-cropbox[0],cropbox[3]-cropbox[1]
    angle=section.get("rotation_ccw",0)
    def local_point(p):
        x,y=p[0]-cropbox[0],p[1]-cropbox[1]
        if angle==90: return y,cw-1-x
        if angle==180: return cw-1-x,ch-1-y
        if angle==270: return ch-1-y,x
        return x,y
    for region in section["regions"]:
        pts=[local_point(p) for p in region["polygon"]]
        nd.line(pts+[pts[0]],fill=section.get("color","#00e5ff"),width=3)
        center=region.get("label",[sum(p[0] for p in region["polygon"])/len(pts),sum(p[1] for p in region["polygon"])/len(pts)])
        x,y=local_point(center)
        nd.rectangle((x-12,y-12,x+18,y+13),fill="#161a22")
        nd.text((x-9,y-10),region["id"],font=font(16),fill="white")
    normalized.save(output / "context.png")
    return records, masks
