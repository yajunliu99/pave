"""Proposer backends.

mock     replays a saved proposal JSON (no network, no key)
session  the same, for an answer a chat model wrote from the prepared image and prompt
openai   any OpenAI-compatible Chat Completions endpoint that accepts images, through PAVE's one model client
         (vision_client.py, the same file read uses): one image and one text part per request, JSON mode,
         answers cached by a hash of the request
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import vision_client

REPO_ROOT = Path(__file__).resolve().parents[1]
# Settings (OPENAI_*, MAPBOX_TOKEN) come from the process environment, then from --env-file when one is given,
# otherwise from .env in this folder or in the PAVE root, so one .env serves read and edit.
DEFAULT_ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT.parent / ".env")


class ProposerError(RuntimeError):
    pass


def environment(env_file: Path | None = None) -> dict:
    return vision_client.environment(*([env_file] if env_file else DEFAULT_ENV_FILES))


# a JSON object that may be wrapped in ``` fences or surrounded by prose
parse_json_text = vision_client.load_json_lenient


class MockProposer:
    """Replays a proposal file: a cached answer (mock) or one written in a chat session (session)."""

    def __init__(self, path: Path, name: str = "mock", model: str | None = None):
        self.path, self.name = Path(path), name
        self.model = model or f"cached:{self.path.name}"

    def propose(self, query: str, image: bytes, system_prompt: str) -> tuple[str, dict]:
        return self.path.read_text(), {"backend": self.name, "model": self.model, "file": self.path.name}


class OpenAIProposer:
    """A vision model behind an OpenAI-compatible /chat/completions endpoint (vision_client.VisionClient).

    Reads OPENAI_API_KEY, OPENAI_MODEL and OPENAI_BASE_URL (default: the OpenAI API); `model` and `base_url`
    override them. Any server that speaks the same protocol works: a model gateway, OpenRouter, a self-hosted server."""
    name = "openai"

    def __init__(self, env_file: Path | None = None, model: str | None = None, base_url: str | None = None,
                 cache_dir: Path | None = None, max_tokens: int = 6000, timeout_s: float = 300.0,
                 max_retries: int = 3, json_mode: bool = True):
        try:
            s = vision_client.settings(environment(env_file), model=model, base_url=base_url)
        except vision_client.VisionError as e:
            raise ProposerError(str(e)) from None
        self.client = vision_client.VisionClient(s["api_key"], s["model"], s["base_url"], timeout_s=timeout_s,
                                                 max_retries=max_retries, json_mode=json_mode)
        self.model, self.base_url, self.url = self.client.model, self.client.base_url, self.client.url
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_tokens, self.json_mode = max_tokens, json_mode

    def payload(self, query: str, image: bytes, system_prompt: str) -> dict:
        return self.client.payload(query, image, system_prompt=system_prompt, max_tokens=self.max_tokens)

    def _post(self, payload: dict) -> dict:
        return self.client.post(payload)

    def _cache_key(self, query, image, system_prompt) -> str:
        h = hashlib.sha256()
        for part in (vision_client.PROTOCOL, self.base_url, self.model, query, system_prompt, str(self.max_tokens), str(self.json_mode)):
            h.update(part.encode() + b"\x00")
        h.update(image)
        return h.hexdigest()[:20]

    def propose(self, query: str, image: bytes, system_prompt: str) -> tuple[str, dict]:
        key = self._cache_key(query, image, system_prompt)
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        try:
            if cache and cache.exists():
                raw, hit = json.loads(cache.read_text()), True
            else:
                raw, hit = vision_client.checked(self._post(self.payload(query, image, system_prompt)), self.model), False
            text = vision_client.response_text(raw)
        except vision_client.VisionError as e:
            raise ProposerError(str(e)) from e
        if cache and not hit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=1))
        usage = raw.get("usage") or {}
        return text, {"backend": "openai", "model": raw.get("model") or self.model, "cache_key": key, "cache_hit": hit,
                      "tokens": usage.get("total_tokens"), "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason")}
