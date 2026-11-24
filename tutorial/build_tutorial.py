"""Build tutorial/PAVE_tutorial.html: the step-by-step tutorial of both worked examples, one self-contained page.

    python tutorial/build_tutorial.py

tutorial/src/sections.html holds the page's table of contents and every chapter except Edit's. A line
<!--BASE:<id>--> in it stands for the section <id> of the page already built (--base, by default the output itself):
that is where the Edit chapter, its calls and images, the page style and script, and the tools come from.

The Read calls are every request of the Read worked example (read/examples/broadway_ames_2021/run): for each, the
image sent, the system prompt, the prompt and the answer, as the run saved them. Images are embedded as reduced JPEG
copies (at most 2000 px on the long edge); prompts and answers are complete.
"""
import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "read/examples/broadway_ames_2021"
sys.path.insert(0, str(ROOT / "read"))
from pave_read.hybrid.prompts import SYSTEM  # noqa: E402

# Request steps in pipeline order, with what the viewer calls each one. A stage folder is <step>_<arm or direction>
# (and _repair1 for a repair). Steps the run did not take are left out.
STEPS = ["trace", "read", "retrace", "use", "detail", "sharedask", "sharedaskdetail", "addlane", "reuse", "redetail",
         "keepuse", "keepdetail", "widekeepuse", "widekeepusedetail", "leftlane", "leftuse", "leftdetail", "leftkeepuse",
         "leftkeepdetail", "wideleftkeepuse", "wideleftkeepusedetail", "leftask", "leftaskdetail", "leftaskkeep",
         "leftaskkeepdetail"]
DATA_KEYS = ("window.PAVE_DATA=", "window.PAVE_FILES=", "window.PAVE_IMG=")
# The tools' texts that named the earlier Read pipeline (no change once applied).
TOOL_EDITS = [
    ("tools/call_vlm.py", "calls/read/draft_in_EB", "calls/read/trace_SE"),
    ("tools/pave_gateway.py", "PAVE Read waits 150 s per request (one retry); for\nslower backends (subagents take minutes) start "
     "the pipeline with PYTHONPATH=tools/pave_hook and PAVE_GATEWAY_URL set.",
     "PAVE Read waits 150 s per request (one retry); for\na slower backend, set PAVE_VLM_TIMEOUT_S in the pipeline's "
     "environment. Agents answering Read's requests\nneed no gateway: Read has its own queue (read/docs/agent_queue.md)."),
]


def reduced(path, longest=2000):
    """A data URI of a reduced JPEG copy of the image."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((longest, longest))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def split_page(html):
    """(head up to the table of contents, sections by id, foot after </main> up to the data, data blocks, script)."""
    nav = re.search(r'<nav class="toc" id="toc">.*?</nav>', html, re.S)
    main_start, main_end = html.index("<main>", nav.end()), html.index("</main>")
    sections = {m.group(1): m.group(0) for m in re.finditer(r'<section id="([^"]+)".*?</section>', html[main_start:main_end], re.S)}
    data_start = html.index("<script>", main_end)
    script_start = html.index("<script>", data_start + 8)
    dec, blocks = json.JSONDecoder(), {}
    chunk = html[data_start:script_start]
    for key in DATA_KEYS:
        i = chunk.index(key) + len(key)
        blocks[key], _ = dec.raw_decode(chunk[i:])
    return html[:nav.start()], sections, html[main_end:data_start], blocks, html[script_start:]


def stage_info(name):
    """(step, place, repair) of a stage folder name such as use_EB or trace_SE_repair1."""
    repair = name.endswith("_repair1")
    base = name[:-len("_repair1")] if repair else name
    step, place = base.split("_", 1)
    return step, place, repair


def read_calls(run, places):
    calls, images = [], {}
    folders = [f for f in (run / "vlm").iterdir() if (f / "latest.json").exists()]
    order = lambda f: (STEPS.index(stage_info(f.name)[0]), stage_info(f.name)[2], f.name)
    for folder in sorted(folders, key=order):
        step, place, repair = stage_info(folder.name)
        latest = json.loads((folder / "latest.json").read_text())
        raw = json.loads((folder / latest["raw_file"]).read_text())
        prompt = (folder / (latest["signature"] + ".prompt.txt")).read_text(encoding="utf-8")
        sheet = run / "sheets" / f"{step}_{place}.jpg"
        key = f"images/read/inputs/{step}_{place}.jpg"
        if key not in images:
            images[key] = reduced(sheet)
        reason = None
        if repair:
            parent = run / "vlm" / folder.name[:-len("_repair1")]
            errors = sorted(parent.glob("*.validation_error.json"))
            reason = json.loads(errors[-1].read_text())["error"] if errors else None
        with Image.open(sheet) as im:
            size = list(im.size)
        calls.append({"id": f"read/{folder.name}", "pipeline": "read", "step": step, "section": place,
                      "label": place + (" repair" if repair else ""), "section_zh": places.get(place, place),
                      "image": key, "original": f"calls/read/{folder.name}/input.jpg", "size": size,
                      "bytes": sheet.stat().st_size, "panels": [], "system": SYSTEM, "prompt": prompt,
                      "answer": raw["response"], "repair_reason": reason, "outputs": []})
    return calls, images


def describe_places(run):
    """A short description of every arm and approach, from the traces."""
    traces = json.loads((run / "traces.json").read_text())
    out = {}
    for arm, t in traces.items():
        s = t["sections"]
        parts = [f"approach {s['inbound']['direction']}" if "inbound" in s else None,
                 f"exit {s['outbound']['direction']}" if "outbound" in s else None]
        out[arm] = f"arm {arm}: " + " and ".join(p for p in parts if p)
        if "inbound" in s:
            out[s["inbound"]["direction"]] = f"approach {s['inbound']['direction']} (arm {arm}), link {s['inbound']['link_id']}"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=ROOT / "tutorial/PAVE_tutorial.html")
    p.add_argument("--sections", type=Path, default=ROOT / "tutorial/src/sections.html")
    p.add_argument("--outputs", type=Path, default=ROOT / "tutorial/src/read_outputs.json",
                   help="what each Read call's answer becomes: images or text shown under the answer")
    p.add_argument("--script", type=Path, default=ROOT / "tutorial/src/read_script.js",
                   help="the Read chapter's page script (street views, step cards)")
    p.add_argument("--output", type=Path, default=ROOT / "tutorial/PAVE_tutorial.html")
    a = p.parse_args()
    head, base_sections, foot, data, script = split_page(a.base.read_text(encoding="utf-8"))
    src = a.sections.read_text(encoding="utf-8")
    nav = re.search(r'<nav class="toc" id="toc">.*?</nav>', src, re.S).group(0)
    body = src[src.index("<main>") + 6:src.index("</main>")]
    body = re.sub(r"<!--BASE:([\w-]+)-->", lambda m: base_sections[m.group(1)], body)

    run = EXAMPLE / "run"
    calls, images = read_calls(run, describe_places(run))
    outputs = json.loads(a.outputs.read_text(encoding="utf-8"))
    for c in calls:
        for o in outputs.get(c["id"].split("/", 1)[1], []):
            if o.get("image"):
                key = "images/read/results/" + Path(o["image"]).name
                images.setdefault(key, reduced(EXAMPLE / o["image"]))
                c["outputs"].append({"image": key, "caption": o["caption"]})
            else:
                c["outputs"].append({"text": o["text"]})
    for f in sorted((EXAMPLE / "imagery/street_views").glob("*.jpg")):
        images[f"images/read/source/street_views/{f.name}"] = reduced(f, 1200)
    images["images/read/source/satellite.jpg"] = reduced(EXAMPLE / "imagery/satellite.jpg", 1600)
    for f in sorted((run / "sheets").glob("arm_*_raw.jpg")):
        images[f"images/read/source/{f.name}"] = reduced(f, 1200)
    images["images/read/results/result.jpg"] = reduced(run / "result.jpg", 2000)
    for m in re.finditer(r'data-img="(images/read/[^"]+)"', body):
        if m.group(1) not in images:
            raise SystemExit(f"sections.html shows {m.group(1)}, which this build does not make")

    pave_data = data["window.PAVE_DATA="]
    pave_data["calls"] = calls + [c for c in pave_data["calls"] if c["pipeline"] != "read"]
    pave_data["street_views"] = sorted(f.stem for f in (EXAMPLE / "imagery/street_views").glob("*.jpg"))
    plan = ROOT / "tutorial/src/plan_only.json"
    pave_data["plan_only"] = plan.read_text(encoding="utf-8") if plan.exists() else ""
    img = {k: v for k, v in data["window.PAVE_IMG="].items() if not k.startswith("images/read/")}
    img.update(images)
    files = data["window.PAVE_FILES="]
    files["network"] = {f"calls/read/_network/{n}": (EXAMPLE / "network" / n).read_text(encoding="utf-8") for n in ("link.csv", "node.csv")}
    files["tools"] = {k: v for k, v in files["tools"].items() if not k.startswith("tools/pave_hook/")}
    for name, old, new in TOOL_EDITS:
        files["tools"][name] = files["tools"][name].replace(old, new)

    read_js = a.script.read_text(encoding="utf-8")
    start, end = script.index("  // step 2: street views"), script.index("  // copy buttons on code blocks")
    script = script[:start] + read_js + script[end:]
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    data_block = ("<script>window.PAVE_DATA=" + dump(pave_data) + ";\nwindow.PAVE_FILES=" + dump(files) +
                  ";\nwindow.PAVE_IMG=" + dump(img) + ";\n</script>\n")
    page = head + nav + "\n<main>" + body + foot + data_block + script
    a.output.write_text(page, encoding="utf-8")
    print({"output": str(a.output.relative_to(ROOT)), "mb": round(len(page.encode()) / 1e6, 1),
           "read_calls": len(calls), "calls": len(pave_data["calls"])})


if __name__ == "__main__":
    main()
