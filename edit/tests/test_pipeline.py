"""python -m unittest discover -s tests   (run from the repository root)"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from pave_edit.checks import run_checks
from pave_edit.georef import Georef
from pave_edit.gmns import Network
from pave_edit.pipeline import run
from pave_edit.proposal import validate
from pave_edit.render import view_extent
from pave_edit.standards import evaluate
from pave_edit.templates import TEMPLATES, baseline_spec
from pave_edit.vlm import MockProposer, OpenAIProposer, ProposerError

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "sr347_i10_ddi"
GOOD = json.loads((SCEN / "mock_proposal.json").read_text())
BAD = json.loads((SCEN / "mock_proposal_bad.json").read_text())


def quiet_run(proposer, **kw):
    tmp = tempfile.TemporaryDirectory()
    with contextlib.redirect_stdout(io.StringIO()):
        code = run(SCEN, Path(tmp.name), proposer, **kw)
    state = json.loads((Path(tmp.name) / "run_state.json").read_text())
    tmp.cleanup()
    return code, state


class ScriptedProposer:
    """Stands in for a live model: returns the scripted answers in order, records prompts."""
    name = "scripted"
    model = "scripted"

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def propose(self, query, image, system_prompt):
        self.prompts.append(query)
        return json.dumps(self.answers.pop(0)), {"backend": "scripted"}


class PipelineTests(unittest.TestCase):
    def test_cached_proposal_builds_ddi(self):
        code, state = quiet_run(MockProposer(SCEN / "mock_proposal.json"))
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "accepted")
        self.assertEqual(state["changes"]["node"]["added"], 4)
        self.assertEqual(state["changes"]["link"]["added"], 4)
        self.assertTrue(all(c["passed"] for c in state["checks"]))

    def test_opposing_carriageway_binding_is_rejected(self):
        code, state = quiet_run(MockProposer(SCEN / "mock_proposal_bad.json"))
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "rejected")
        self.assertTrue(all(d["gate"] == "P4 role fit" for d in state["defects"]))

    def test_one_repair_pass_recovers(self):
        p = ScriptedProposer(BAD, GOOD)
        code, state = quiet_run(p, max_repairs=1)
        self.assertEqual(code, 0)
        self.assertEqual(len(state["attempts"]), 2)
        self.assertIn("THE CHECKS REJECTED IT", p.prompts[1])
        self.assertIn("P4 role fit", p.prompts[1])

    def test_session_answer_with_hidden_names(self):
        answer = SCEN / "proposals" / "blind_mapbox_nonames.json"
        code, state = quiet_run(MockProposer(answer, name="session", model="chat session"), hide_names=True)
        self.assertEqual(code, 0)
        self.assertEqual(state["backend"], "session")
        self.assertEqual(state["status"], "accepted")
        self.assertTrue(all(c["passed"] for c in state["checks"]))

    def test_prepare_only_stops_before_proposing(self):
        tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            code = run(SCEN, Path(tmp.name), ScriptedProposer(), prepare_only=True, hide_names=True)
        files = {f.name for f in Path(tmp.name).iterdir()}
        tmp.cleanup()
        self.assertEqual(code, 0)
        self.assertEqual(files, {"ai_input.png", "ai_prompt.txt"})

    def test_repair_limit_is_respected(self):
        code, state = quiet_run(ScriptedProposer(BAD, BAD), max_repairs=1)
        self.assertEqual(code, 1)
        self.assertEqual(len(state["attempts"]), 2)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.net = Network.load(SCEN / "base")
        self.extent = view_extent(self.net)

    def gates(self, mutate):
        p = copy.deepcopy(GOOD)
        mutate(p)
        return {d["gate"] for d in validate(p, self.net, self.extent)[0]}

    def test_good_passes(self):
        self.assertEqual(self.gates(lambda p: None), set())

    def test_hallucinated_id(self):
        def m(p):
            p["terminals"][0]["bindings"]["approach_in"]["link_id"] = 999
        self.assertEqual(self.gates(m), {"P2 id exists"})

    def test_echo_mismatch(self):
        def m(p):
            p["terminals"][1]["bindings"]["onramp_out"]["to_node_id"] = 13
        self.assertEqual(self.gates(m), {"P3 echo neighbours"})

    def test_wrong_class(self):
        def m(p):
            p["facility_class"] = "spui"
            p["operations"][0]["from_class"] = "spui"
        self.assertIn("P7 class cross-check", self.gates(m))

    def test_anchor_on_wrong_terminal(self):
        def m(p):
            p["terminals"][1]["anchor_cell"] = "D2"
        self.assertEqual(self.gates(m), {"P6 anchor"})

    def test_operation_not_implemented(self):
        def m(p):
            p["operations"].append({"op": "add_lane", "from_class": "diamond", "to_class": "diamond", "terminals": ["SW"]})
        self.assertEqual(self.gates(m), {"P8 operations"})

    def test_unknown_catalog_label(self):
        def m(p):
            p["facility_class"] = "dumbbell"
        self.assertEqual(self.gates(m), {"P1 schema"})


class TemplateTests(unittest.TestCase):
    """Each template passes its own checks, and the checks reject the wrong network."""

    def setUp(self):
        self.base = Network.load(SCEN / "base")
        self.base.rebuild_movements()
        _, self.terms = validate(GOOD, self.base, view_extent(self.base))

    def build(self, target):
        return TEMPLATES[("replace_facility", "diamond", target)](self.base, self.terms, f"after_{target}")

    def failed(self, after, spec):
        return {c["check"].split()[0] for c in run_checks(self.base, after, self.terms, spec) if not c["passed"]}

    def test_every_target_through_the_pipeline(self):
        for target in ("ddi", "spui", "roundabout_diamond"):
            with self.subTest(target=target):
                code, state = quiet_run(MockProposer(SCEN / "mock_proposal.json"), target=target)
                self.assertEqual(code, 0, state.get("error"))
                self.assertEqual(state["target_override"], target)
                self.assertTrue(all(c["passed"] for c in state["checks"]))
                self.assertNotIn("not_met", {c["status"] for c in state["standards"]})

    def test_ddi_checks_reject_the_plain_diamond(self):
        _, _, spec = self.build("ddi")
        self.assertTrue({"DDI-1", "DDI-2", "DDI-3"} <= self.failed(self.base, spec))

    def test_spui_movement_rules_reject_the_ddi(self):
        ddi, _, _ = self.build("ddi")
        _, _, spui_spec = self.build("spui")
        self.assertIn("C3", self.failed(ddi, spui_spec))

    def test_clockwise_ring_is_caught(self):
        after, _, spec = self.build("roundabout_diamond")
        for l in after.links.values():
            if l.get("control_status") == "circulating":
                l["from_node_id"], l["to_node_id"] = l["to_node_id"], l["from_node_id"]
                l["geometry"] = l["geometry"][::-1]
        self.assertIn("RBT-1", self.failed(after, spec))

    def test_left_turn_conflict_is_measured(self):
        spec = baseline_spec(self.terms)
        crit = {c["id"]: c for c in evaluate("diamond", self.base, self.base, self.terms, spec, [])}
        self.assertEqual(crit["DIA-2"]["value"], 2)          # one conflict per terminal in the diamond
        ddi, _, dspec = self.build("ddi")
        checks = run_checks(self.base, ddi, self.terms, dspec)
        crit = {c["id"]: c for c in evaluate("ddi", self.base, ddi, self.terms, dspec, checks)}
        self.assertEqual(crit["DDI-S2"]["status"], "met")
        self.assertEqual(crit["DDI-S4"]["value"], 47.5)       # crossover angle on the schematic


class GeorefTests(unittest.TestCase):
    def setUp(self):
        self.base = Network.load(SCEN / "base")
        _, self.terms = validate(GOOD, self.base, view_extent(self.base))
        self.geo = Georef(SCEN, self.base)

    def test_base_is_drawn_where_it_was_measured(self):
        nodes, links = self.geo.network(self.base)
        for nid, xy in self.geo.node_xy.items():
            self.assertEqual(tuple(nodes[nid]), xy)
        self.assertEqual(links[301], self.geo.link_geom[301])

    def test_ddi_crossovers_land_inside_their_terminals(self):
        ddi, _, _ = TEMPLATES[("replace_facility", "diamond", "ddi")](self.base, self.terms, "after_ddi")
        nodes, _ = self.geo.network(ddi)
        for nid, n in ddi.nodes.items():
            if n["node_type"] == "crossover":
                ports = [self.geo.node_xy[p] for t in self.terms if t.group == n["control_group"] for p in t.port_nodes()]
                xs, ys = zip(*ports)
                self.assertTrue(min(xs) <= nodes[nid][0] <= max(xs) and min(ys) <= nodes[nid][1] <= max(ys), (nid, nodes[nid]))


class OpenAIProposerTests(unittest.TestCase):
    """Replays Chat Completions responses; the network call itself is stubbed."""

    def setUp(self):                      # the developer's own OPENAI_* variables must not leak in
        clean = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
        self.env = mock.patch.dict(os.environ, clean, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def make(self, content=None, **kw):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tmp.write("OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-vision-model\nOPENAI_BASE_URL=https://example.invalid/v1/\n")
        tmp.close()
        p = OpenAIProposer(Path(tmp.name), **kw)
        Path(tmp.name).unlink()
        body = content if content is not None else ("I'll read panel A first.\n```json\n" + json.dumps(GOOD) + "\n```")
        p.sent = []
        p._post = lambda payload: p.sent.append(payload) or {
            "model": "test-vision-model-2026", "usage": {"total_tokens": 9000},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": body}}]}
        return p

    def test_answer_with_prose_and_fences(self):
        p = self.make()
        code, state = quiet_run(p)
        self.assertEqual(code, 0)
        self.assertEqual(state["attempts"][0]["meta"]["tokens"], 9000)
        self.assertEqual(p.url, "https://example.invalid/v1/chat/completions")
        sent = p.sent[0]
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        user = sent["messages"][1]["content"]
        self.assertEqual([part["type"] for part in user], ["text", "image_url"])
        self.assertTrue(user[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_content_as_list_of_parts(self):
        code, _ = quiet_run(self.make(content=[{"type": "text", "text": json.dumps(GOOD)}]))
        self.assertEqual(code, 0)

    def test_arguments_override_the_env_file(self):
        p = self.make(model="other-model", base_url="http://localhost:8000/v1")
        self.assertEqual((p.model, p.url), ("other-model", "http://localhost:8000/v1/chat/completions"))

    def test_missing_key_is_reported(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tmp.write("OPENAI_MODEL=test-vision-model\n")
        tmp.close()
        with self.assertRaises(ProposerError):
            OpenAIProposer(Path(tmp.name))
        Path(tmp.name).unlink()

    def test_non_json_reply_goes_to_repair(self):
        code, state = quiet_run(self.make(content="I cannot tell from this image."), max_repairs=0)
        self.assertEqual(code, 1)
        self.assertEqual(state["defects"][0]["gate"], "P0 parse")


if __name__ == "__main__":
    unittest.main()
