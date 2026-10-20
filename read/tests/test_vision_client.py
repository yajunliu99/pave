"""The OpenAI-compatible vision client, and StageClient answering through it."""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pave_read import vision_client
from pave_read.hybrid.common import ValidationError
from pave_read.hybrid.inference import StageClient


class Reply:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def completion(content, model="vendor/vision-model"):
    return {"id": "c1", "model": model, "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}


class Opener:
    """Answers each request with the next reply (a body, or an HTTP status to fail with) and keeps the requests."""
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, req, timeout=None, context=None):
        self.requests.append(json.loads(req.data.decode()))
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(req.full_url, reply, "error", {}, io.BytesIO(b"busy"))
        return Reply(reply)


class ClientTests(unittest.TestCase):
    def test_identity_names_the_provider(self):
        self.assertEqual(vision_client.identity("vendor/vision-model", "https://gateway.example/v1"), ("vision-model", "vendor"))
        self.assertEqual(vision_client.identity("gpt-test", "https://api.example.com/v1"), ("gpt-test", "api.example.com"))

    def test_settings_come_from_the_environment(self):
        env = {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "vendor/vision-model", "OPENAI_BASE_URL": "https://gateway.example/v1/"}
        self.assertEqual(vision_client.settings(env), {"api_key": "k", "model": "vendor/vision-model", "base_url": "https://gateway.example/v1"})
        self.assertEqual(vision_client.settings({"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"})["base_url"], vision_client.DEFAULT_BASE_URL)
        with self.assertRaises(vision_client.VisionError):
            vision_client.settings({"OPENAI_MODEL": "m"})
        self.assertIsNone(vision_client.settings({"OPENAI_MODEL": "m"}, need_key=False)["api_key"])

    def test_request_and_answer_envelope(self):
        opener = Opener(completion('{"lanes": 2}'))
        client = vision_client.VisionClient("k", "vendor/vision-model", "https://gateway.example/v1", opener=opener)
        answer = client.query_vision("count the lanes", b"png-bytes", system_prompt="be careful", mime_type="image/png",
                                     max_tokens=100, temperature=0.0)
        sent = opener.requests[0]
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertTrue(sent["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual((sent["max_completion_tokens"], sent["temperature"], sent["response_format"]), (100, 0.0, {"type": "json_object"}))
        self.assertEqual(vision_client.extract_payload(answer), {"lanes": 2})
        self.assertEqual(vision_client.validate_identity(answer, "vision-model", "vendor")["response_model"], "vendor/vision-model")
        self.assertEqual(answer["metadata"]["usage_metric"]["usage"]["total_tokens"], 13)

    def test_answer_from_another_model_is_refused(self):
        client = vision_client.VisionClient("k", "vendor/vision-model", opener=Opener(completion("{}", model="vendor/other-model")))
        with self.assertRaises(vision_client.VisionError):
            client.query_vision("q", b"x")

    def test_busy_server_is_retried_and_every_attempt_counted(self):
        counted = []
        with patch("time.sleep"):
            client = vision_client.VisionClient("k", "m", opener=Opener(503, completion("{}", model="m")), max_retries=1,
                                                before_request=lambda: counted.append(1))
            client.query_vision("q", b"x")
        self.assertEqual(len(counted), 2)

    def test_empty_answer_is_an_error(self):
        with self.assertRaises(vision_client.VisionError):
            vision_client.VisionClient("k", "m", opener=Opener(completion("  ", model="m"))).query_vision("q", b"x")


class StageClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".env").write_text("OPENAI_API_KEY=test-only\nOPENAI_MODEL=vendor/vision-model\n"
                                        "OPENAI_BASE_URL=https://gateway.example/v1\n")
        (self.root / "image.png").write_bytes(b"png-bytes")
        self.clean = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "VLM_"))}, clear=True)
        self.clean.start()

    def tearDown(self):
        self.clean.stop()
        self.tmp.cleanup()

    def test_stage_answers_through_the_openai_compatible_api(self):
        opener = Opener(completion('{"ok": true}'))
        with patch("urllib.request.urlopen", opener):
            client = StageClient(self.root, self.root / "run", {})
            result = client.run("stage", self.root / "image.png", "prompt", lambda v: v)
        self.assertEqual(result, {"ok": True})
        self.assertEqual((client.model, client.provider, client.calls), ("vision-model", "vendor", 1))
        # the answer is cached under the request's signature and replayed without a request
        replay = StageClient(self.root, self.root / "run", {}, cache_only=True)
        self.assertEqual(replay.run("stage", self.root / "image.png", "prompt", lambda v: v), {"ok": True})
        self.assertEqual(replay.calls, 0)

    def test_call_budget_stops_before_the_request(self):
        client = StageClient(self.root, self.root / "run", {"max_api_calls": 1})
        client.calls = 1
        with patch("urllib.request.urlopen", Opener(completion("{}"))) as opener:
            with self.assertRaises(ValidationError):
                client.run("stage", self.root / "image.png", "prompt", lambda v: v)
        self.assertEqual(opener.requests, [])


if __name__ == "__main__":
    unittest.main()
