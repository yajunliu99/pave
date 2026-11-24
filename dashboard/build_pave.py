"""Build dashboard/pave.html: one self-contained page with the PAVE overview and both worked examples.

pave_template.html is PAVE's one page template: the overview and the Read and Edit views, each with a placeholder for
its data. This script fills all three:
- Overview: from pave_content.json; the read example's figures are counted from its run.
- Read: the view's data, made from the read example's run and imagery by read_view.py; Leaflet, the map library of
  the view, comes from vendor/.
- Edit: the view's data, built from edit/ by edit/tools/build_dashboard.py (its arguments are in pave_content.json).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_view import read_view  # noqa: E402


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
    run = ROOT / example
    lanes = json.loads((run / "lanes.json").read_text())
    movements = json.loads((run / "movements.json").read_text())
    approaches = [s for s in lanes if s["kind"] == "inbound_stopbar"]
    return [[str(len(lanes)), "sections traced"],
            [str(sum(s["count"]["model_motor_count"] for s in approaches)), f"motor lanes on {len(approaches)} approaches"],
            [str(sum(m.get("status") == "candidate" for m in movements)), "turns proposed"],
            [str(len(list((run / "vlm").rglob("*.raw.json")))), "model requests"]]


def edit_data(build):
    """The Edit view's data, built from edit/."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "edit.json"
        subprocess.run([sys.executable, *build["command"], "--data-out", str(out)], cwd=ROOT / build["cwd"], check=True,
                       stdout=subprocess.DEVNULL)
        return out.read_text(encoding="utf-8")


def read_parts(c, vendor):
    """The Read view's parts: its index, one data block per junction, and Leaflet."""
    run = ROOT / c["example"]
    index, sites = read_view(run, ROOT / c["imagery"], c["view_title"], ROOT / "read")
    blocks = "\n".join(f'<script type="application/json" id="site-{key}">{inline_json(data)}</script>' for key, data in sites.items())
    return {"/*__INDEX__*/": inline_json(index), "<!--__SITES__-->": blocks,
            "/*__LEAFLET_CSS__*/": (vendor / "leaflet.css").read_text(encoding="utf-8"),
            "/*__LEAFLET_JS__*/": (vendor / "leaflet.js").read_text(encoding="utf-8").replace("</script", "<\\/script")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--content", type=Path, default=ROOT / "dashboard/pave_content.json")
    p.add_argument("--template", type=Path, default=ROOT / "dashboard/pave_template.html")
    p.add_argument("--vendor", type=Path, default=ROOT / "dashboard/vendor", help="folder holding leaflet.js and leaflet.css")
    p.add_argument("--output", type=Path, default=ROOT / "dashboard/pave.html")
    a = p.parse_args()
    c = json.loads(a.content.read_text())
    data = {k: c[k] for k in ("tagline", "eyebrow", "lead", "stages", "loop_back", "principles", "footer")}
    data["apps"] = [
        {"view": "read", "color": "P", "kind": c["read"]["kind"], "title": c["read"]["title"],
         "text": c["read"]["text"], "facts": read_facts(c["read"]["example"]), "image": thumbnail(ROOT / c["read"]["image"])},
        {"view": "edit", "color": "E", "kind": c["edit"]["kind"], "title": c["edit"]["title"], "text": c["edit"]["text"],
         "facts": c["edit"]["facts"], "image": thumbnail(ROOT / c["edit"]["image"])},
    ]
    parts = {**read_parts(c["read"], a.vendor), "/*__PAVE__*/": inline_json(data), "/*__EDIT__*/": edit_data(c["edit"]["build"])}
    page = a.template.read_text(encoding="utf-8")
    for key in parts:
        if page.count(key) != 1:
            raise SystemExit(f"template must contain one {key} placeholder")
    page = re.sub("|".join(map(re.escape, parts)), lambda m: parts[m.group(0)], page)   # one pass: no part is read as a placeholder
    a.output.write_text(page, encoding="utf-8")
    print({"output": str(a.output), "mb": round(a.output.stat().st_size / 1e6, 1)})


if __name__ == "__main__":
    main()
