"""Build dashboard/pave.html: one self-contained page with the PAVE overview and both worked examples.

pave_template.html is PAVE's one page template: the overview and the Read and Edit views, each with a placeholder for
its data. This script fills all three:
- Overview: from pave_content.json; the read example's figures are counted from its exported results.
- Edit: the view's data, built from edit/ by edit/tools/build_dashboard.py (its arguments are in pave_content.json).
- Read: the read export builds a page with the Read view filled (dashboard/build/read_example.html, not kept in git);
  its parts are taken from there, or, without it, from the pave.html already built.
"""
import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def inline_json(value):
    """JSON safe to place inside a <script> element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def thumbnail(path, width=1100):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, width))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def read_facts(example):
    results = ROOT / example / "results"
    lanes = json.loads((results / "classification/lanes.json").read_text())
    movements = json.loads((results / "classification/movements.json").read_text())
    approaches = [s for s in lanes if s["section_id"].startswith("in_")]
    return [[str(len(lanes)), "sections traced"],
            [str(sum(s["count"]["model_motor_count"] for s in approaches)), f"motor lanes on {len(approaches)} approaches"],
            [str(sum(m.get("status") == "candidate" for m in movements)), "movements proposed"],
            [str(len(list((results / "images/street_views").glob("*.jpg")))), "street views"],
            [str(len(list((results / "answers").rglob("*.raw.json")))), "model answers"]]


def edit_data(build):
    """The Edit view's data, built from edit/."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "edit.json"
        subprocess.run([sys.executable, *build["command"], "--data-out", str(out)], cwd=ROOT / build["cwd"], check=True,
                       stdout=subprocess.DEVNULL)
        return out.read_text(encoding="utf-8")


# The Read view's parts: placeholder -> where a page built from the template holds it once filled.
READ_PARTS = {"/*__LEAFLET_CSS__*/": r'<style id="leaflet-css">(.*?)</style>',
              "/*__LEAFLET_JS__*/": r'<script id="leaflet-js">(.*?)</script>',
              "/*__INDEX__*/": r'<script type="application/json" id="index">(.*?)</script>',
              "<!--__SITES__-->": r"<!--sites-->\n(.*?)\n<!--/sites-->"}


def read_parts(path):
    text = path.read_text(encoding="utf-8")
    parts = {}
    for key, pattern in READ_PARTS.items():
        match = re.search(pattern, text, re.S)
        if not match or match.group(1).strip() == key:
            raise SystemExit(f"{path} holds no Read data ({key}); run the read export first")
        parts[key] = match.group(1)
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--content", type=Path, default=ROOT / "dashboard/pave_content.json")
    p.add_argument("--template", type=Path, default=ROOT / "dashboard/pave_template.html")
    p.add_argument("--output", type=Path, default=ROOT / "dashboard/pave.html")
    a = p.parse_args()
    c = json.loads(a.content.read_text())
    data = {k: c[k] for k in ("tagline", "eyebrow", "lead", "stages", "loop_back", "principles", "footer")}
    data["apps"] = [
        {"view": "read", "color": "P", "kind": c["read"]["kind"], "title": c["read"]["title"], "text": c["read"]["text"],
         "facts": read_facts(c["read"]["example"]), "image": thumbnail(ROOT / c["read"]["image"])},
        {"view": "edit", "color": "E", "kind": c["edit"]["kind"], "title": c["edit"]["title"], "text": c["edit"]["text"],
         "facts": c["edit"]["facts"], "image": thumbnail(ROOT / c["edit"]["image"])},
    ]
    fresh = ROOT / c["read"]["dashboard"]
    if not fresh.exists():
        print(f"{fresh.relative_to(ROOT)} not found: the Read view keeps the data already in {a.output.name}")
    parts = {**read_parts(fresh if fresh.exists() else a.output), "/*__PAVE__*/": inline_json(data), "/*__EDIT__*/": edit_data(c["edit"]["build"])}
    page = a.template.read_text(encoding="utf-8")
    for key in parts:
        if page.count(key) != 1:
            raise SystemExit(f"template must contain one {key} placeholder")
    page = re.sub("|".join(map(re.escape, parts)), lambda m: parts[m.group(0)], page)   # one pass: no part is read as a placeholder
    a.output.write_text(page, encoding="utf-8")
    print({"output": str(a.output), "mb": round(a.output.stat().st_size / 1e6, 1)})


if __name__ == "__main__":
    main()
