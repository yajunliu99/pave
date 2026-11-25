"""YOLO detections of traffic lights and stop signs in street views, as hints for the signal and sign audit.

The detector only proposes where to look: its boxes go into the model's prompt as proposals, can miss targets and
say nothing about which lanes a signal governs. Each view's detections are cached by image hash, model checksum
and options. Needs the optional dependencies (pip install -e ".[signals]") and the yolo26n checkpoint; its path
comes from PAVE_YOLO_WEIGHTS (default ~/.cache/pave/weights/yolo26n.pt) and its sha256 is checked.
"""
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..hybrid.common import digest, file_hash, read_json, require, write_json
from ..hybrid.geometry import font

DEFAULTS = {"model": "yolo26n", "expected_sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
            "device": "auto", "imgsz": 1280, "confidence": 0.25, "classes": ["traffic light", "stop sign"]}


def weights_path():
    return Path(os.environ.get("PAVE_YOLO_WEIGHTS") or "~/.cache/pave/weights/yolo26n.pt").expanduser()


def detect(views, options, paths, output):
    """{view id: {"detections": [...]}} for every view; `paths` maps a view id to its image file."""
    folder = Path(output) / "yolo"
    folder.mkdir(parents=True, exist_ok=True)
    model, results = None, {}
    for view in views:
        vid, path = view["id"], Path(paths[view["id"]])
        signature = digest({"image": file_hash(path), "options": options})
        record = folder / f"{vid}_{signature[:16]}.json"
        if record.exists():
            data = read_json(record)
        else:
            if model is None:
                model, torch, version = _load(options)
            start = time.perf_counter()
            device = options["device"] if options["device"] != "auto" else (0 if torch.cuda.is_available() else "cpu")
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            result = model.predict(np.array(rgb)[:, :, ::-1].copy(), device=device, imgsz=options["imgsz"],
                                   conf=options["confidence"], quantize=32, rect=False, verbose=False, save=False)[0]
            detections = []
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    if result.names[int(cls)] in options["classes"]:
                        detections.append({"class": result.names[int(cls)], "confidence": round(conf, 5), "bbox_xyxy": [round(v, 2) for v in box]})
            data = {"model": options["model"], "weights_sha256": options["expected_sha256"], "ultralytics": version,
                    "elapsed_s": round(time.perf_counter() - start, 3), "image_size": list(rgb.size), "detections": detections}
            write_json(record, data)
            overlay(rgb, detections, vid).save(folder / f"{vid}_detections.jpg", quality=85)
        results[vid] = data
    write_json(folder / "summary.json", results)
    return results


def _load(options):
    cache = Path.home() / ".cache/pave/ultralytics-config"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    weights = weights_path()
    require(weights.exists(), f"YOLO checkpoint not found at {weights}: set PAVE_YOLO_WEIGHTS")
    require(file_hash(weights) == options["expected_sha256"], "YOLO checkpoint checksum mismatch")
    return YOLO(str(weights), task="detect"), torch, ultralytics.__version__


def overlay(image, detections, title):
    out = image.copy()
    d = ImageDraw.Draw(out)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        d.rectangle((x1, y1, x2, y2), outline="#ff7c99", width=3)
        d.text((x1, max(0, y1 - 18)), det["class"], font=font(17), fill="#ff7c99", stroke_width=1, stroke_fill="black")
    d.rectangle((8, 8, min(out.width - 8, 640), 40), fill="#161a22")
    d.text((15, 13), title, font=font(18), fill="white")
    return out
