"""Travel-up crops and the annotated views sent to the draft and review stages.

Rulers, station labels and boundary IDs sit in margins outside the pixel area.
Only dashed station lines and dashed boundary lines are drawn on the imagery, so
the paint underneath stays visible, and the review sheet adds every region's
unmarked pixels. Sheets stay compact: a provider that downsizes a large image
would blur the ruler the model has to read.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..hybrid.geometry import font
from .strips import region_polygon

LEFT, TOP, BOTTOM, RIGHT = 46, 54, 32, 14
STATION, EDGE, PAPER, MASKED = "#ff3df2", "#16d5ec", "#161a22", (55, 55, 55)


def crop(image, frame):
    """Rotate the frame's window so travel points up. Cardinal headings keep source pixels."""
    resample = Image.Resampling.NEAREST if frame.pixels_preserved() else Image.Resampling.BICUBIC
    return image.transform((frame.width, frame.height), Image.Transform.AFFINE, frame.affine(), resample=resample)


def lit(image, mean_below=80., p90_below=90.):
    """A crop lying wholly in shadow, contrast-stretched on lightness so shaded paint shows; None for any other crop.

    Both limits must hold, so a crop with sunlit pavement anywhere in it is left as it is (see `shade_lifted`)."""
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    if lightness.mean() >= mean_below or np.percentile(lightness, 90) >= p90_below:
        return None
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 8)).apply(lightness)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def shade_fraction(image, shade_ratio=0.72):
    level = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    sunlit = float(np.percentile(level, 75))
    return float((cv2.GaussianBlur(level, (0, 0), 6) < shade_ratio * sunlit).mean())


def shade_lifted(image, shade_ratio=0.72, min_shade=0.2):
    """A copy of a partly shaded crop with the shadow lightened; None when under `min_shade` of it is shaded.

    The sunlit level is the crop's upper-quartile lightness. Shadow is treated as dimmed light: pixels whose
    smoothed neighbourhood is darker than `shade_ratio` of that level are scaled by the ratio of the sunlit
    level to the smoothed local lightness, blended in by how dark the neighbourhood is. Dark objects (cars,
    new asphalt) are lightened too, so the copy is shown beside the original, never instead of it."""
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    level = lab[..., 0].astype(np.float32)
    sunlit = float(np.percentile(level, 75))
    limit = shade_ratio * sunlit
    smooth = cv2.GaussianBlur(level, (0, 0), 6)
    if (smooth < limit).mean() < min_shade:
        return None
    illumination = cv2.GaussianBlur(level, (0, 0), 12)
    gain = np.clip(0.92 * sunlit / np.maximum(illumination, 1), 1, 4)
    weight = np.clip((limit + 0.1 * sunlit - smooth) / (0.2 * sunlit), 0, 1)
    lab[..., 0] = np.clip(level * (1 + (gain - 1) * weight), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def enlarge(image, scale):
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _dashed(d, points, fill, on=12, off=7, width=2):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        if not length:
            continue
        t = 0.0
        while t < length:
            a, b = t / length, min(t + on, length) / length
            d.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=fill, width=width)
            t += on + off


def ruled(raw, stations, scale=2, step=50):
    """Crop with x rulers in crop units (top and bottom) and dashed station lines S1..SK."""
    body = enlarge(raw, scale)
    canvas = Image.new("RGB", (LEFT + body.width + RIGHT, TOP + body.height + BOTTOM), PAPER)
    canvas.paste(body, (LEFT, TOP))
    d = ImageDraw.Draw(canvas)
    base = TOP + body.height
    for x in range(0, raw.width + 1, 10):
        px, major = LEFT + x * scale, x % step == 0
        d.line((px, TOP - (9 if major else 4), px, TOP - 1), fill="white")
        d.line((px, base, px, base + (9 if major else 4)), fill="white")
        if major:
            d.text((px, 2), str(x), font=font(17), fill="white", anchor="ma")
            d.text((px, base + 10), str(x), font=font(17), fill="white", anchor="ma")
    for i, y in enumerate(stations):
        py = TOP + round(y * scale)
        _dashed(d, [(LEFT, py), (LEFT + body.width - 1, py)], STATION, on=8, off=12)
        d.text((4, py), f"S{i + 1}", font=font(18), fill=STATION, anchor="lm")
    return canvas


def overlay(raw, strips, scale=2, step=50):
    """Ruled crop with the current boundaries e1..eN and neutral region IDs r1..r(N-1)."""
    canvas = ruled(raw, strips["stations"], scale, step)
    d = ImageDraw.Draw(canvas)
    stations, edges = strips["stations"], strips["edges"]
    for i, xs in enumerate(edges):
        _dashed(d, [(LEFT + x * scale, TOP + y * scale) for x, y in zip(xs, stations)], EDGE)
        d.text((LEFT + xs[0] * scale, 23), f"e{i + 1}", font=font(17), fill=EDGE, anchor="ma")
    # Region IDs sit between two stations so they never cover a station crossing.
    j = max(0, len(stations) // 2 - 1)
    y = TOP + (stations[j] + stations[j + 1]) / 2 * scale
    for i, (left, right) in enumerate(zip(edges, edges[1:])):
        x = LEFT + (left[j] + left[j + 1] + right[j] + right[j + 1]) / 4 * scale
        d.rectangle((x - 14, y - 11, x + 14, y + 11), fill=PAPER)
        d.text((x, y), f"r{i + 1}", font=font(16), fill="white", anchor="mm")
    return canvas


def panels(raw, strips, scale=2):
    """Each region's own pixels, isolated, at one common scale, driver-left to driver-right."""
    cells = []
    for i in range(len(strips["edges"]) - 1):
        mask = Image.new("L", raw.size, 0)
        ImageDraw.Draw(mask).polygon(region_polygon(strips, i), fill=255)
        box = mask.getbbox() or (0, 0, 1, 1)
        isolated = Image.composite(raw, Image.new("RGB", raw.size, MASKED), mask)
        cells.append(enlarge(isolated.crop(box), scale))
    gap, head = 26, 40
    width = sum(max(c.width, 44) + gap for c in cells) + gap
    panel = Image.new("RGB", (width, head + max(c.height for c in cells) + 12), MASKED)
    d = ImageDraw.Draw(panel)
    x = gap
    for i, cell in enumerate(cells):
        slot = max(cell.width, 44)
        panel.paste(cell, (x + (slot - cell.width) // 2, head))
        d.text((x + slot / 2, 8), f"r{i + 1}", font=font(20), fill="white", anchor="ma")
        x += slot + gap
    return panel
