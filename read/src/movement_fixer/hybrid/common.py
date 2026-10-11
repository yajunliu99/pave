from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class ValidationError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content=json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        if path.exists() and path.read_text(encoding="utf-8")==content:
            return  # Avoid unnecessary replacements while Dropbox/indexers inspect results.
    except PermissionError:
        pass
    # Keep temporary basenames short: nested run/cache paths on Windows can
    # otherwise exceed MAX_PATH when a second hash is appended to a long name.
    temp = path.parent / (".write-"+uuid.uuid4().hex[:16]+".tmp")
    temp.write_text(content, encoding="utf-8")
    for attempt in range(7):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==6: raise
            time.sleep(.05 * 2**attempt)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resource_path(root, value):
    # Configs and manifests written on Windows use backslashes, and some hold absolute paths from
    # that machine. When such a path is not found, re-root it at this checkout by the project folder name.
    text = str(value).replace("\\", "/")
    parts = text.split("/")
    if len(parts[0]) == 2 and parts[0][1] == ":" and not Path(text).exists() and Path(root).name in parts:
        text = "/".join(parts[len(parts) - parts[::-1].index(Path(root).name):])
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def load_config(path, root):
    data = read_json(path)
    require(data.get("schema_version") == 1, "Unsupported config schema_version")
    from .legs import check_sections
    check_sections(data["sections"], data["legs"])
    for section in data["sections"] + data["gsv_views"]:
        require(section["source_id"] in data["sources"], f"Unknown source for {section['id']}")
        region_ids = [r["id"] for r in section["regions"]]
        require(region_ids and len(region_ids) == len(set(region_ids)), "Empty/duplicate regions")
        for region in section["regions"]:
            require(len(region["polygon"]) >= 3, f"Invalid polygon {section['id']}:{region['id']}")
        require(section.get("rotation_ccw", 0) in (0, 90, 180, 270), "Rotation must preserve pixels")
    context_ids=[v["id"] for v in data.get("context_views",[])]
    require(len(context_ids)==len(set(context_ids)),"Duplicate context view IDs")
    for view in data.get("context_views",[]):
        require(view["source_id"] in data["sources"],"Context references missing source")
        require(not data["sources"][view["source_id"]].get("reverse_view",False),"Primary context must face forward")
    for source in data["sources"].values():
        key = "path" if source["kind"] == "image" else "pano_path"
        require(resource_path(root, source[key]).is_file(), f"Missing input {source[key]}")
    from .sampling import audit_sampling
    audit_sampling(data)
    return data


def request_workers():
    """How many independent model requests a stage may have open at once (NET2CELL_PARALLEL_REQUESTS, default 1)."""
    return max(1, int(os.environ.get("NET2CELL_PARALLEL_REQUESTS", "1")))


def parallel_map(fn, items, workers=None):
    """fn over items, results in the order of items; up to `workers` at once (1 runs a plain loop)."""
    items, workers = list(items), workers or request_workers()
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))

