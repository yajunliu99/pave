"""One vision-language model request through an OpenAI-compatible Chat Completions endpoint.

This module is the one model client of PAVE's two apps; it imports nothing from either package, so the same file
serves both (read: hybrid/inference.StageClient; edit: pave_edit.vlm).

Settings come from the environment or an env file:

  OPENAI_API_KEY    the key, sent as a bearer token
  OPENAI_BASE_URL   the API root (default https://api.openai.com/v1)
  OPENAI_MODEL      a model that accepts images

Any server speaking the same protocol works the same way: a model gateway (whose model ids may carry a provider
prefix, e.g. vendor/model-name), OpenRouter, or a self-hosted model server. Always send a system prompt: some
gateways put a long default one in front of a request that has none.

`complete` returns the Chat Completions body. `query_vision` returns it as an answer envelope,
{"response": text, "metadata": {"model_details": ..., "usage_metric": ...}}, the form read stores and replays.
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROTOCOL = "openai-chat-completions-v1"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
RETRY_STATUS = (429, 500, 502, 503, 504)


class VisionError(RuntimeError):
    """The request could not be made, or the answer is unusable (missing, empty or from another model)."""


def read_env_file(path) -> dict:
    """KEY=value lines of an env file; an absent file gives nothing."""
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def environment(*env_files) -> dict:
    """Settings from the env files (the first one that sets a key wins), with the process environment on top
    (an empty variable does not hide a file's value)."""
    merged = {}
    for f in env_files:
        for k, v in read_env_file(f).items():
            merged.setdefault(k, v)
    return {**merged, **{k: v for k, v in os.environ.items() if v}}


def settings(env: dict, model: str | None = None, base_url: str | None = None, need_key: bool = True) -> dict:
    """api_key, model and base_url for a client; `model` and `base_url` override the environment. Without
    `need_key`, a missing key is allowed (enough to name the model, e.g. to replay cached answers)."""
    api_key = env.get("OPENAI_API_KEY")
    if need_key and not api_key:
        raise VisionError("OPENAI_API_KEY not found; set it in the environment or in the env file")
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise VisionError("no model given; set OPENAI_MODEL or pass a model (it must accept images)")
    return {"api_key": api_key, "model": model, "base_url": (base_url or env.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")}


def identity(model: str, base_url: str = DEFAULT_BASE_URL) -> tuple[str, str]:
    """(model, provider) recorded with every answer: a gateway id "vendor/model-name" gives ("model-name", "vendor");
    a plain id gives the API host as the provider."""
    if "/" in model:
        provider, name = model.split("/", 1)
        return name, provider
    return model, urlparse(base_url).netloc or base_url


def ssl_context() -> ssl.SSLContext:
    """python.org builds on macOS ship without system CAs; use certifi's bundle when present."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class VisionClient:
    """Sends one prompt and one image per request.

    `before_request` is called before every HTTP request, retries included; it may raise to stop the call (a budget).
    `opener` replaces urllib.request.urlopen (tests)."""

    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 300.0,
                 max_retries: int = 3, json_mode: bool = True, before_request=None, opener=None):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.timeout_s, self.max_retries, self.json_mode = timeout_s, max_retries, json_mode
        self.before_request = before_request
        self.opener = opener or urllib.request.urlopen

    def payload(self, prompt: str, image: bytes, system_prompt: str | None = None, mime_type: str = "image/png",
                max_tokens: int = 6000, temperature: float | None = None) -> dict:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64," + base64.b64encode(image).decode()}}]})
        p = {"model": self.model, "messages": messages, "max_completion_tokens": max_tokens}
        if temperature is not None:
            p["temperature"] = temperature
        if self.json_mode:
            p["response_format"] = {"type": "json_object"}
        return p

    def complete(self, prompt: str, image: bytes, **kw) -> dict:
        """The Chat Completions body of one request, checked (see checked)."""
        return checked(self.post(self.payload(prompt, image, **kw)), self.model)

    def query_vision(self, prompt: str, image: bytes, system_prompt: str | None = None, mime_type: str = "image/png",
                     max_tokens: int = 6000, temperature: float | None = None) -> dict:
        """One request as an answer envelope (see envelope)."""
        body = self.complete(prompt, image, system_prompt=system_prompt, mime_type=mime_type, max_tokens=max_tokens,
                             temperature=temperature)
        return envelope(body, self.model, self.base_url)

    def post(self, payload: dict) -> dict:
        """The JSON body of the answer to a prepared payload (see payload), unchecked."""
        data = json.dumps(payload).encode()
        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            if self.before_request:
                self.before_request()
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with self.opener(req, timeout=self.timeout_s, context=ssl_context()) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                e.close()
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise VisionError(f"API HTTP {e.code}: {text[:300]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries and not isinstance(getattr(e, "reason", None), ssl.SSLError):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise VisionError(f"API network error: {e}") from e
        raise VisionError("API: retries exhausted")


def checked(body: dict, model: str) -> dict:
    """The body, once it holds a non-empty answer from the requested model (a dated variant of it counts)."""
    response_text(body)
    answered = body.get("model")
    if answered and answered != model and not answered.startswith(model + "-"):
        raise VisionError(f"requested model {model}, the response came from {answered}")
    return body


def response_text(body: dict) -> str:
    """The model's text in a Chat Completions body (content as a string or a list of parts)."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        keys = sorted(body) if isinstance(body, dict) else type(body).__name__
        raise VisionError(f"no choices[0].message.content in the response ({keys})") from None
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise VisionError("the model returned an empty answer")
    return content


def envelope(body: dict, model: str, base_url: str = DEFAULT_BASE_URL) -> dict:
    """A Chat Completions body as an answer envelope: the text under "response"; under "metadata", the model
    (model_details: requested model and provider, and the model the API reported) and the token counts and costs
    (usage_metric; a gateway's own figures are kept beside the standard ones)."""
    name, provider = identity(model, base_url)
    choice = (body.get("choices") or [{}])[0]
    usage = {"usage": body.get("usage")}
    gateway = (body.get("metadata") or {}).get("usage_metric") if isinstance(body.get("metadata"), dict) else None
    if gateway:
        usage["gateway"] = gateway
    return {"response": response_text(body), "finish_reason": choice.get("finish_reason"), "completion_id": body.get("id"),
            "protocol": PROTOCOL,
            "metadata": {"model_details": {"inference_provider": provider, "inference_model": name,
                                           "response_model": body.get("model")},
                         "usage_metric": usage}}


def validate_identity(answer: dict, model: str, provider: str) -> dict:
    """The model_details of an answer envelope, checked against the expected (model, provider)."""
    metadata = answer.get("metadata") if isinstance(answer, dict) else None
    details = metadata.get("model_details") if isinstance(metadata, dict) else None
    if not isinstance(details, dict):
        raise VisionError("the answer is missing metadata.model_details")
    actual = (details.get("inference_provider"), details.get("inference_model"))
    if actual != (provider, model):
        raise VisionError(f"model mismatch: requested {provider}/{model}, received {actual[0]}/{actual[1]}")
    return details


def load_json_lenient(s: str):
    """Parse JSON that may be wrapped in ``` fences or have prose around it."""
    t = s.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    lo, hi = t.find("{"), t.rfind("}")
    if lo >= 0 and hi > lo:
        t = t[lo:hi + 1]
    return json.loads(t)


def extract_payload(answer: dict):
    """The model's JSON body inside an answer envelope (or a gateway's own response wrapper)."""
    if not isinstance(answer, dict):
        return answer
    for key in ("response", "result", "output", "data", "content", "answer", "message"):
        if key in answer:
            v = answer[key]
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                try:
                    return load_json_lenient(v)
                except Exception:
                    continue
    return answer
