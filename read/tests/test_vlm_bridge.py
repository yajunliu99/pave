import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image

from pave_read import vlm_bridge
from pave_read.hybrid.inference import StageClient


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = mock.patch.dict(os.environ, {"PAVE_VLM_BACKEND": "bridge", "PAVE_VLM_BRIDGE_DIR": str(self.tmp / "queue"),
                                                "PAVE_BRIDGE_MODEL": "test-model", "OPENAI_MODEL": "vendor/api-model"})
        self.env.start()
        self.image = self.tmp / "crop.png"
        Image.new("RGB", (40, 30), (90, 90, 90)).save(self.image)

    def tearDown(self):
        self.env.stop()

    def worker(self, reply, seen):
        def loop():
            for _ in range(200):
                claimed = vlm_bridge.claim(1, "w1")
                if claimed:
                    seen.append(Path(claimed[0]["prompt_file"]).read_text())
                    vlm_bridge.answer(claimed[0]["id"], reply)
                    return
                time.sleep(0.02)
        thread = threading.Thread(target=loop)
        thread.start()
        return thread

    def test_stage_client_gets_the_worker_answer_and_caches_it(self):
        seen = []
        thread = self.worker('```json\n{"lanes": 3}\n```', seen)
        client = StageClient(ROOT, self.tmp / "run", {"max_api_calls": 5})
        result = client.run("lane_check_NB", self.image, "How many lanes?", lambda x: x)
        thread.join()
        self.assertEqual(result, {"lanes": 3})
        self.assertEqual((client.model, client.provider), ("test-model", vlm_bridge.PROVIDER))
        self.assertIn("How many lanes?", seen[0])
        self.assertIn(str(self.image.resolve()), seen[0])
        latest = json.loads((self.tmp / "run" / "vlm" / "lane_check_NB" / "latest.json").read_text())
        self.assertEqual(latest["actual_model"]["inference_model"], "test-model")
        self.assertEqual(vlm_bridge.status(), {"pending": 0, "claimed": 0, "answers": 1, "done": 1})
        again = StageClient(ROOT, self.tmp / "run", {"max_api_calls": 5}, cache_only=True)
        self.assertEqual(again.run("lane_check_NB", self.image, "How many lanes?", lambda x: x), {"lanes": 3})

    def test_another_provider_never_reads_an_agents_cached_answer(self):
        thread = self.worker('{"lanes": 3}', [])
        StageClient(ROOT, self.tmp / "run", {"max_api_calls": 5}).run("lane_check_NB", self.image, "How many lanes?", lambda x: x)
        thread.join()
        with mock.patch.dict(os.environ, {"PAVE_BRIDGE_PROVIDER": "codex_subagent", "PAVE_BRIDGE_REUSE_IDENTICAL": "0"}):
            self.assertEqual(vlm_bridge.identity(), ("test-model", "codex_subagent"))
            other = StageClient(ROOT, self.tmp / "run", {"max_api_calls": 5}, cache_only=True)
            self.assertEqual(other.provider, "codex_subagent")
            with self.assertRaises(Exception):
                other.run("lane_check_NB", self.image, "How many lanes?", lambda x: x)
        with mock.patch.dict(os.environ, {"PAVE_BRIDGE_PROVIDER": " "}):
            self.assertEqual(vlm_bridge.identity()[1], vlm_bridge.PROVIDER)

    def api_answer(self, prior, name, prompt):
        """An answer by the API model to this exact request, cached in an earlier run `prior`."""
        from pave_read.hybrid.common import file_hash
        client = StageClient(ROOT, prior, {})
        signature = client._signature(file_hash(self.image), prompt, *client.api_identity)
        folder = prior / "vlm" / name
        folder.mkdir(parents=True)
        (folder / f"{signature}.raw.json").write_text(json.dumps({"response": {"lanes": 2}, "metadata": {"model_details": {
            "inference_provider": client.api_identity[1], "inference_model": client.api_identity[0]}}}))

    def test_identical_request_keeps_the_api_answer(self):
        prior = self.tmp / "prior"
        self.api_answer(prior, "surface_in_EB", "Classify.")
        client = StageClient(ROOT, self.tmp / "run", {}, reuse_cache_from=[prior])
        self.assertEqual(client.run("surface_in_EB", self.image, "Classify.", lambda x: x), {"lanes": 2})
        self.assertEqual(vlm_bridge.status()["pending"], 0)
        latest = json.loads((self.tmp / "run" / "vlm" / "surface_in_EB" / "latest.json").read_text())
        self.assertEqual(latest["actual_model"]["inference_model"], client.api_identity[0])
        again = StageClient(ROOT, self.tmp / "run", {}, cache_only=True)
        self.assertEqual(again.run("surface_in_EB", self.image, "Classify.", lambda x: x), {"lanes": 2})

    def test_reuse_of_api_answers_can_be_turned_off(self):
        prior = self.tmp / "prior"
        self.api_answer(prior, "surface_in_EB", "Classify.")
        seen = []
        thread = self.worker('{"lanes": 3}', seen)
        with mock.patch.dict(os.environ, {"PAVE_BRIDGE_REUSE_IDENTICAL": "0"}):
            client = StageClient(ROOT, self.tmp / "run", {}, reuse_cache_from=[prior])
            self.assertEqual(client.run("surface_in_EB", self.image, "Classify.", lambda x: x), {"lanes": 3})
        thread.join()

    def test_claim_is_exclusive_and_release_requeues(self):
        folder = vlm_bridge.queue_dir()
        (folder / "pending" / "a.json").write_text(json.dumps({"id": "a"}))
        self.assertEqual(len(vlm_bridge.claim(5, "w1")), 1)
        self.assertEqual(vlm_bridge.claim(5, "w2"), [])
        os.utime(folder / "claimed" / "a.json", (time.time() - 4000, time.time() - 4000))
        self.assertEqual(vlm_bridge.release(1800), ["a"])
        self.assertEqual(vlm_bridge.status()["pending"], 1)

    def test_claim_by_stage_keeps_readers_and_reviewers_apart(self):
        folder = vlm_bridge.queue_dir()
        for name in ("20261009_100000_reading_WB_aaaaaaaa", "20261009_100001_recheck_review_in_WB_r1_bbbbbbbb"):
            (folder / "pending" / f"{name}.json").write_text(json.dumps({"id": name}))
        self.assertEqual([r["id"] for r in vlm_bridge.claim(5, "reviewer", skip=["reading_"])], ["20261009_100001_recheck_review_in_WB_r1_bbbbbbbb"])
        self.assertEqual([r["id"] for r in vlm_bridge.claim(5, "reader", only=["reading_"])], ["20261009_100000_reading_WB_aaaaaaaa"])

    def test_answer_rejects_text_that_is_not_json(self):
        folder = vlm_bridge.queue_dir()
        (folder / "claimed" / "b.json").write_text(json.dumps({"id": "b"}))
        with self.assertRaises(json.JSONDecodeError):
            vlm_bridge.answer("b", "three lanes")
        self.assertTrue((folder / "claimed" / "b.json").exists())

    def test_default_backend_is_the_api(self):
        with mock.patch.dict(os.environ, {"PAVE_VLM_BACKEND": "", "OPENAI_MODEL": "vendor/vision-model"}):
            client = StageClient(ROOT, self.tmp / "run2", {})
        self.assertEqual((client.backend, client.model, client.provider), ("api", "vision-model", "vendor"))

    def test_queue_needs_no_api_settings(self):
        with mock.patch.dict(os.environ, {"PAVE_VLM_BACKEND": "bridge", "OPENAI_MODEL": ""}), \
                mock.patch("pave_read.vision_client.read_env_file", return_value={}):
            client = StageClient(ROOT, self.tmp / "run3", {})
        self.assertEqual((client.backend, client.api_identity), ("bridge", None))


if __name__ == "__main__":
    unittest.main()
