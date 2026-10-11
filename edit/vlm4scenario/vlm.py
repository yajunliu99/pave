"""Proposer backends.

mock     replays a saved proposal JSON (no network, no key)
session  the same, for an answer a chat model wrote from the prepared image and prompt
openai   any OpenAI-compatible Chat Completions endpoint that accepts images: one image and one
         text part per request, JSON mode, answers cached by a hash of the request
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
# API keys come from the environment or an env file: --env-file, $VLM4SCENARIO_ENV_FILE, or .env in the repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def env_file_path(env_file: Path | None = None) -> Path:
    return Path(env_file or os.environ.get("VLM4SCENARIO_ENV_FILE") or DEFAULT_ENV_FILE)


class ProposerError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    """python.org builds on macOS ship without system CAs; use certifi's bundle when present."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def parse_json_text(s: str):
    """Parse a JSON object that may be wrapped in ``` fences or surrounded by prose."""
    t = s.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    lo, hi = t.find("{"), t.rfind("}")
    if lo >= 0 and hi > lo:
        t = t[lo:hi + 1]
    return json.loads(t)


class MockProposer:
    """Replays a proposal file: a cached answer (mock) or one written in a chat session (session)."""

    def __init__(self, path: Path, name: str = "mock", model: str | None = None):
        self.path, self.name = Path(path), name
        self.model = model or f"cached:{self.path.name}"

    def propose(self, query: str, image: bytes, system_prompt: str) -> tuple[str, dict]:
        return self.path.read_text(), {"backend": self.name, "model": self.model, "file": self.path.name}


def load_env(path: Path) -> dict:
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class OpenAIProposer:
    """A vision model behind an OpenAI-compatible /chat/completions endpoint.

    Reads OPENAI_API_KEY, OPENAI_MODEL and OPENAI_BASE_URL (default: the OpenAI API) from the
    environment or the env file; `model` and `base_url` override them. Any server that speaks the
    same protocol works, e.g. a self-hosted model server."""
    name = "openai"

    def __init__(self, env_file: Path | None = None, model: str | None = None, base_url: str | None = None,
                 cache_dir: Path | None = None, max_tokens: int = 6000, timeout_s: float = 300.0,
                 max_retries: int = 3, json_mode: bool = True):
        env = load_env(env_file_path(env_file))
        get = lambda k: os.environ.get(k) or env.get(k)  # noqa: E731
        self.api_key = get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProposerError("OPENAI_API_KEY not found; set it in the environment, in .env, or pass --env-file")
        self.model = model or get("OPENAI_MODEL")
        if not self.model:
            raise ProposerError("no model given; pass --model or set OPENAI_MODEL (the model must accept images)")
        self.base_url = (base_url or get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_tokens, self.timeout_s, self.max_retries, self.json_mode = max_tokens, timeout_s, max_retries, json_mode

    def payload(self, query: str, image: bytes, system_prompt: str) -> dict:
        p = {"model": self.model,
             "messages": [{"role": "system", "content": system_prompt},
                          {"role": "user", "content": [
                              {"type": "text", "text": query},
                              {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}}]}],
             "max_completion_tokens": self.max_tokens}
        if self.json_mode:
            p["response_format"] = {"type": "json_object"}
        return p

    def _cache_key(self, query, image, system_prompt) -> str:
        h = hashlib.sha256()
        for part in ("chat-completions-v1", self.base_url, self.model, query, system_prompt, str(self.max_tokens), str(self.json_mode)):
            h.update(part.encode() + b"\x00")
        h.update(image)
        return h.hexdigest()[:20]

    def propose(self, query: str, image: bytes, system_prompt: str) -> tuple[str, dict]:
        key = self._cache_key(query, image, system_prompt)
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache and cache.exists():
            raw, hit = json.loads(cache.read_text()), True
        else:
            raw, hit = self._post(self.payload(query, image, system_prompt)), False
        text = response_text(raw)
        if cache and not hit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=1))
        usage = raw.get("usage") or {}
        return text, {"backend": "openai", "model": raw.get("model") or self.model, "cache_key": key, "cache_hit": hit,
                      "tokens": usage.get("total_tokens"), "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason")}

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=_ssl_context()) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProposerError(f"API HTTP {e.code}: {text[:300]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries and not isinstance(getattr(e, "reason", None), ssl.SSLError):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProposerError(f"API network error: {e}") from e
        raise ProposerError("API: retries exhausted")


def response_text(raw: dict) -> str:
    """The model's text from a Chat Completions response (content as a string or a list of parts)."""
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProposerError(f"no choices[0].message.content in the response (keys {sorted(raw)})") from None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise ProposerError("the model returned an empty answer")
    return content
