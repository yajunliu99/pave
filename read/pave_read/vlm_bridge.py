"""A file queue that lets agents answer the pipeline's vision requests in place of an API.

With PAVE_VLM_BACKEND=bridge, StageClient writes each request it cannot answer from cache to the queue:
<queue>/pending/<id>.json (metadata) and <queue>/prompts/<id>.txt (system prompt, prompt and image path in plain
text). It then waits for <queue>/answers/<id>.json. Workers, such as Claude Code subagents, claim requests,
look at the image, and write their JSON answers with scripts/pipeline/vlm_bridge.py (see docs/agent_queue.md).
The answer is wrapped in the same envelope as an API answer (vision_client), so the stage code, its validators and
its caches work unchanged. The model identity recorded with each answer is the queue's (provider
claude_code_subagent by default), so these answers never mix with cached API answers.

Settings come from the environment:
- PAVE_VLM_BRIDGE_DIR: the queue folder (default <runs root>/vlm_bridge)
- PAVE_BRIDGE_MODEL: the model that answers (default claude-opus-5-5)
- PAVE_BRIDGE_PROVIDER: who answers (default claude_code_subagent; for example codex_subagent when Codex
  subagents answer), recorded with the model so answers from different agents never share a cache entry
- PAVE_BRIDGE_TIMEOUT_S: how long a request waits for its answer (default 4 h)
- PAVE_BRIDGE_REUSE_IDENTICAL: 0 to send requests the API model already answered (same image and prompt) to the
  agents as well; by default those keep the API answer, and latest.json records which model answered each request
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .paths import runs_root

PROVIDER = "claude_code_subagent"
STATES = ("pending", "claimed", "answers", "done", "prompts", "work")


def backend():
    return os.environ.get("PAVE_VLM_BACKEND", "api").strip().lower() or "api"


def identity():
    """(model, provider) recorded for bridge answers."""
    return (os.environ.get("PAVE_BRIDGE_MODEL", "claude-opus-5-5"),
            os.environ.get("PAVE_BRIDGE_PROVIDER", "").strip() or PROVIDER)


def reuse_identical():
    """Whether a bridge run keeps the API model's cached answer to a byte-identical request (default yes;
    PAVE_BRIDGE_REUSE_IDENTICAL=0 makes every request go to the agents)."""
    return os.environ.get("PAVE_BRIDGE_REUSE_IDENTICAL", "1").strip() not in ("0", "false", "no")


def queue_dir():
    folder = Path(os.environ["PAVE_VLM_BRIDGE_DIR"]).expanduser() if os.environ.get("PAVE_VLM_BRIDGE_DIR") else runs_root() / "vlm_bridge"
    for state in STATES:
        (folder / state).mkdir(parents=True, exist_ok=True)
    return folder


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_answer(text):
    """The JSON object in a worker's answer; ``` fences and text around the object are tolerated."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(t[start:end + 1])


def ask(name, image_path, prompt, system, signature, output, timeout_s=None, poll_s=2.0):
    """Queue one request and wait for its answer; returns an answer envelope (as vision_client returns)."""
    folder = queue_dir()
    model, provider = identity()
    stem = f"{re.sub(r'[^A-Za-z0-9_]+', '_', name)[:60]}_{signature[:8]}"
    # The same request already in the queue (answered, being answered or waiting), e.g. from a run that was stopped
    # and resumed, is waited for instead of being asked again.
    earlier = sorted(p for state in ("answers", "claimed", "pending") for p in (folder / state).glob(f"*_{stem}.json"))
    if earlier:
        return _await(folder, earlier[-1].stem, model, provider, timeout_s, poll_s)
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}"
    image = Path(image_path).resolve()
    try:
        from PIL import Image
        size = list(Image.open(image).size)
    except Exception:
        size = None
    _write(folder / "prompts" / f"{rid}.txt",
           f"REQUEST {rid}\nSTAGE {name}\nIMAGE {image}\nIMAGE SIZE (pixels, width x height) {size}\n"
           f"ANSWER WITH: python scripts/pipeline/vlm_bridge.py answer --id {rid} --file <your JSON file>\n\n"
           f"=== SYSTEM PROMPT ===\n{system}\n\n=== PROMPT ===\n{prompt}\n")
    request = {"id": rid, "stage": name, "image": str(image), "image_size": size, "prompt_file": str(folder / "prompts" / f"{rid}.txt"),
               "answer_file": str(folder / "answers" / f"{rid}.json"), "run": str(output), "model": model, "provider": provider,
               "created_utc": _now()}
    _write(folder / "pending" / f"{rid}.json", json.dumps(request, indent=1))
    return _await(folder, rid, model, provider, timeout_s, poll_s)


def _await(folder, rid, model, provider, timeout_s, poll_s):
    """Wait for the answer to request `rid`; returns an answer envelope (as vision_client returns)."""
    answer_path = folder / "answers" / f"{rid}.json"
    timeout = float(timeout_s or os.environ.get("PAVE_BRIDGE_TIMEOUT_S", 4 * 3600))
    started = time.time()
    while not answer_path.exists():
        if time.time() - started > timeout:
            for state in ("pending", "claimed"):
                stale = folder / state / f"{rid}.json"
                if stale.exists():
                    stale.unlink()
            raise TimeoutError(f"bridge request {rid} not answered within {timeout:.0f} s")
        time.sleep(poll_s)
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    return {"response": answer["response"],
            "metadata": {"model_details": {"inference_provider": provider, "inference_model": model},
                         "bridge": {"request_id": rid, "worker": answer.get("worker"), "claimed_utc": answer.get("claimed_utc"),
                                    "answered_utc": answer.get("answered_utc"), "waited_s": round(time.time() - started, 1)}}}


def stage_of(name):
    """The stage named in a request file name (<date>_<time>_<stage>_<8 hex>)."""
    return name.split("_", 2)[2].rsplit("_", 1)[0] if name.count("_") >= 3 else ""


def claim(limit=1, worker=None, only=(), skip=()):
    """Move up to `limit` pending requests, oldest first, to claimed/; returns their metadata.

    `only` / `skip` are stage-name prefixes, so that different workers answer different kinds of request (a worker that
    read an approach blind should not also review it)."""
    folder = queue_dir()
    out = []
    for path in sorted((folder / "pending").glob("*.json"), key=lambda p: p.name):
        if len(out) >= limit:
            break
        stage = stage_of(path.stem)
        if (only and not stage.startswith(tuple(only))) or (skip and stage.startswith(tuple(skip))):
            continue
        target = folder / "claimed" / path.name
        try:
            os.rename(path, target)
        except FileNotFoundError:
            continue  # another worker took it
        request = json.loads(target.read_text(encoding="utf-8"))
        request.update({"worker": worker, "claimed_utc": _now()})
        _write(target, json.dumps(request, indent=1))
        out.append(request)
    return out


def answer(rid, text, worker=None):
    """Record a worker's answer for a claimed request."""
    folder = queue_dir()
    claimed = folder / "claimed" / f"{rid}.json"
    if not claimed.exists():
        raise FileNotFoundError(f"request {rid} is not claimed (answered already, or claim it first)")
    request = json.loads(claimed.read_text(encoding="utf-8"))
    body = load_answer(text)
    _write(folder / "answers" / f"{rid}.json", json.dumps({"response": body, "worker": worker or request.get("worker"),
                                                            "claimed_utc": request.get("claimed_utc"), "answered_utc": _now()},
                                                           ensure_ascii=False, indent=1))
    os.replace(claimed, folder / "done" / f"{rid}.json")
    return body


def release(older_than_s=1800):
    """Return requests claimed longer than `older_than_s` ago to pending/ (their worker stopped)."""
    folder = queue_dir()
    moved = []
    for path in (folder / "claimed").glob("*.json"):
        if time.time() - path.stat().st_mtime > older_than_s:
            os.replace(path, folder / "pending" / path.name)
            moved.append(path.stem)
    return moved


def status():
    folder = queue_dir()
    return {state: len(list((folder / state).glob("*.json"))) for state in ("pending", "claimed", "answers", "done")}
