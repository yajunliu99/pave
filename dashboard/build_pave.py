"""Build dashboard/pave.html: one self-contained page with the PAVE overview and both worked examples.

The overview text comes from pave_content.json; the read example's figures are counted from its exported results.
The two example dashboards (each already a single self-contained file) load in a frame when their tab is first opened.
Inside PAVE they share one look and one vocabulary: pave_skin.css restyles both, each gets the same header (its
harness stages, from pave_content.json), and the read dashboard's wording is mapped onto PAVE's terms with the
lexicon in pave_content.json (the read app is exported from another project, so its files are not edited here).
The page's light or dark theme is passed on to both.
"""
import argparse
import base64
import io
import json
import re
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


def reworded(html, lexicon, name):
    """Apply [old, new] pairs; every old phrase must still be in the page, so a changed dashboard is noticed."""
    for old, new in lexicon:
        if old not in html:
            raise SystemExit(f"{name}: lexicon phrase not found: {old!r}")
        html = html.replace(old, new)
    return html


def offline(html):
    """Drop web-font links: inside PAVE the pages use the system font, and the file stays usable offline."""
    return re.sub(r'<link[^>]+fonts\.(?:googleapis|gstatic)\.com[^>]*>\s*', "", html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--content", type=Path, default=ROOT / "dashboard/pave_content.json")
    p.add_argument("--template", type=Path, default=ROOT / "dashboard/pave_template.html")
    p.add_argument("--skin", type=Path, default=ROOT / "dashboard/pave_skin.css")
    p.add_argument("--output", type=Path, default=ROOT / "dashboard/pave.html")
    a = p.parse_args()
    c = json.loads(a.content.read_text())
    data = {k: c[k] for k in ("tagline", "eyebrow", "lead", "stages", "loop_back", "principles", "footer")}
    data["skin"] = a.skin.read_text(encoding="utf-8")
    data["frames"] = {v: {k: c[v][k] for k in ("kind", "title", "context", "stages", "marks")} for v in ("read", "edit")}
    data["apps"] = [
        {"view": "read", "color": "P", "kind": c["read"]["kind"], "title": c["read"]["title"], "text": c["read"]["text"],
         "facts": read_facts(c["read"]["example"]), "image": thumbnail(ROOT / c["read"]["image"])},
        {"view": "edit", "color": "E", "kind": c["edit"]["kind"], "title": c["edit"]["title"], "text": c["edit"]["text"],
         "facts": c["edit"]["facts"], "image": thumbnail(ROOT / c["edit"]["image"])},
    ]
    page = a.template.read_text(encoding="utf-8")
    parts = {"/*__DATA__*/": inline_json(data),
             "/*__READ__*/": inline_json(offline(reworded((ROOT / c["read"]["dashboard"]).read_text(encoding="utf-8"),
                                                           c["read"].get("lexicon", []), "read"))),
             "/*__EDIT__*/": inline_json(offline(reworded((ROOT / c["edit"]["dashboard"]).read_text(encoding="utf-8"),
                                                           c["edit"].get("lexicon", []), "edit")))}
    for key, value in parts.items():
        if page.count(key) != 1:
            raise SystemExit(f"template must contain one {key} placeholder")
        page = page.replace(key, value)
    a.output.write_text(page, encoding="utf-8")
    print({"output": str(a.output), "mb": round(a.output.stat().st_size / 1e6, 1)})


if __name__ == "__main__":
    main()
