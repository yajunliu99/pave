"""Review-loop control flow and the site runner, with scripted model responses.

These check that the loop terminates, records why, and applies what a review
asked for. They say nothing about whether a real model traces boundaries well.
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from movement_fixer.fusion.coordinates import Viewport
from movement_fixer.hybrid.common import ValidationError, read_json, write_json
from movement_fixer.autoloop.controller import run_review_loop
from movement_fixer.autoloop.evaluate import compare, evaluate_run
from movement_fixer.autoloop.geometry_stage import run_site, validate_draft, validate_review
from movement_fixer.autoloop.strips import Frame, section_regions


def loop(script, issues=lambda s: [], apply=None, **options):
    """Toy loop over integer states; `script` yields one review response per round."""
    responses = iter(script)
    sent = []

    def review(state, found, number, rounds):
        sent.append([i["key"] for i in found])
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value
    result = run_review_loop(0, check=lambda s: [{"key": k} for k in issues(s)], review=review,
                             apply=apply or (lambda s, edits: (s + sum(edits), edits, [])), signature=lambda s: s,
                             magnitude=lambda a, b: abs(a - b), **options)
    return result, sent


def accept(**extra):
    return {"verdict": "accept", "edits": [], "issue_responses": [], **extra}


def revise(*edits, **extra):
    return {"verdict": "revise", "edits": list(edits), "issue_responses": [], **extra}


class ControllerTests(unittest.TestCase):
    def test_accepted_on_first_review(self):
        result, _ = loop([accept()])
        self.assertEqual((result["status"], result["stop_reason"], len(result["rounds"]), result["state"]),
                         ("accepted", "accepted", 1, 0))

    def test_revision_then_acceptance(self):
        result, _ = loop([revise(10), accept()])
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("accepted_after_revision", "accepted", 10))
        self.assertEqual(result["rounds"][0]["change_px"], 10)

    def test_small_change_counts_as_converged(self):
        result, _ = loop([revise(10), revise(1)])
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("accepted_after_revision", "converged", 11))

    def test_returning_to_an_earlier_state_stops_the_loop(self):
        result, _ = loop([revise(10), revise(-10), accept()], policy={"max_rounds": 5})
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("unresolved", "oscillation", 10))
        self.assertTrue(result["rounds"][1]["returned_to_earlier_state"])
        self.assertEqual(len(result["rounds"]), 2)

    def test_round_limit(self):
        result, _ = loop([revise(10), revise(10), revise(10), accept()], issues=lambda s: ["still-open"])
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("unresolved", "round_limit", 30))
        self.assertEqual(result["open_issues"], ["still-open"])

    def test_budget_is_checked_before_every_review(self):
        calls = []
        result, sent = loop([revise(10), accept()], budget_left=lambda: len(calls) < 1 and not calls.append(1))
        self.assertEqual((result["stop_reason"], result["state"], len(sent)), ("budget_exhausted", 10, 1))
        result, sent = loop([accept()], budget_left=lambda: False)
        self.assertEqual((result["status"], result["stop_reason"], sent), ("unresolved", "budget_exhausted", []))

    def test_edits_that_cannot_be_applied(self):
        result, _ = loop([revise(10)], apply=lambda s, edits: (s, [], [{"edit": e, "reason": "x"} for e in edits]))
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("unresolved", "no_applicable_edits", 0))

    def test_dismissed_issue_is_not_raised_again(self):
        script = [revise(10, issue_responses=[{"issue": "narrow", "decision": "dismiss", "reason": "bicycle strip"}]), accept()]
        result, sent = loop(script, issues=lambda s: ["narrow"])
        self.assertEqual(sent, [["narrow"], []])
        self.assertEqual((result["status"], result["dismissed"]), ("accepted_after_revision", {"narrow": "bicycle strip"}))

    def test_acceptance_with_an_unresolved_issue_is_not_settled(self):
        script = [accept(issue_responses=[{"issue": "shadow", "decision": "unresolved", "reason": "hidden by trees"}])]
        result, _ = loop(script, issues=lambda s: ["shadow"])
        self.assertEqual((result["status"], result["stop_reason"], result["open_issues"]), ("unresolved", "accepted", ["shadow"]))

    def test_failed_review_keeps_the_last_good_state(self):
        result, _ = loop([revise(10), ValidationError("bad json")])
        self.assertEqual((result["status"], result["stop_reason"], result["state"]), ("unresolved", "review_failed", 10))
        self.assertEqual(result["rounds"][1]["error"], "bad json")


class ValidatorTests(unittest.TestCase):
    def spec(self):
        return {"id": "in_NB", "kind": "inbound_stopbar", "stations": [10., 60., 110.], "frame": Frame((500, 500), 0, 200, 120)}

    def boundary(self, x, **extra):
        return {"xs": [x] * 3, "kind": "lane_line", "visibility": "clear", "interpolated_stations": [], "evidence": "paint", **extra}

    def test_draft_is_sorted_and_checked(self):
        draft = validate_draft({"boundaries": [self.boundary(120), self.boundary(20), self.boundary(70, interpolated_stations=["S2"])]}, self.spec())
        self.assertEqual([e[0] for e in draft["strips"]["edges"]], [20, 70, 120])
        self.assertEqual(draft["strips"]["meta"][1]["interpolated_stations"], ["S2"])
        for bad in ([self.boundary(20)], [self.boundary(20), self.boundary(22)], [self.boundary(20), self.boundary(260)],
                    [self.boundary(20), self.boundary(70, kind="motor_lane")], [self.boundary(20), {**self.boundary(70), "xs": [70, 70]}],
                    [self.boundary(20), self.boundary(70, interpolated_stations=["S9"])]):
            with self.assertRaises(ValidationError):
                validate_draft({"boundaries": bad}, self.spec())

    def test_review_must_answer_every_key_and_cite_evidence(self):
        edit = {"op": "split_region", "region": "r1", "evidence": "dashed line inside"}
        good = {"verdict": "revise", "edits": [edit], "issue_responses": [{"issue": "k1", "decision": "edit", "reason": "line visible"}]}
        self.assertEqual(validate_review(good, ["k1"])["edits"], [edit])
        bad = [{**good, "issue_responses": []}, {**good, "verdict": "accept"}, {**good, "edits": [{"op": "split_region", "region": "r1"}]},
               {**good, "edits": []}, {**good, "edits": [{"op": "repaint", "evidence": "x"}]},
               {"verdict": "accept", "edits": [], "issue_responses": [{"issue": "k1", "decision": "edit", "reason": "x"}]},
               {"verdict": "accept", "edits": [], "issue_responses": [{"issue": "k1", "decision": "dismiss", "reason": " "}]}]
        for value in bad:
            with self.assertRaises(ValidationError):
                validate_review(value, ["k1"])
        validate_review({"verdict": "accept", "edits": [], "issue_responses": [{"issue": "k1", "decision": "dismiss", "reason": "bicycle strip"}]}, ["k1"])


class ScriptedVLM:
    """Stands in for StageClient: one scripted response per stage name."""
    provider, model = "scripted", "offline"

    def __init__(self, respond):
        self.respond, self.calls, self.hits, self.stages, self.prompts = respond, 0, 0, [], {}

    def run(self, name, image, prompt, validator):
        assert Path(image).is_file(), image
        self.calls += 1
        self.stages.append(name)
        self.prompts[name] = prompt
        return validator(self.respond(name, prompt))


def boundaries(*xs, stations=5):
    return {"boundaries": [{"xs": [x] * stations, "kind": "lane_line", "visibility": "clear", "interpolated_stations": [],
                            "evidence": "painted line"} for x in xs]}


def keys(prompt):
    return re.findall(r"- key (\S+) \|", prompt)


TRUE = (34, 70, 106, 142, 178)


def misses_one_line(name, prompt):
    """The draft leaves out the line at x=106; the first review restores it."""
    if name.startswith("draft_"):
        return boundaries(34, 70, 142, 178)
    found = keys(prompt)
    if name.endswith("_r1"):
        return {"verdict": "revise", "edits": [{"op": "split_region", "region": "r2", "evidence": "dashed separator at x=106"}],
                "issue_responses": [{"issue": k, "decision": "edit", "reason": "separator visible inside"} for k in found]}
    return {"verdict": "accept", "edits": [], "issue_responses": [{"issue": k, "decision": "dismiss", "reason": "matches paint"} for k in found]}


class SiteTests(unittest.TestCase):
    def site(self, temp, **policy):
        root = Path(temp)
        Image.new("RGB", (1200, 1200), (88, 88, 88)).save(root / "sat.png")
        view = Viewport(1200, 1200, (-12461000.0, 3951400.0, -12460880.0, 3951520.0), "synthetic test viewport")

        def wkt(*pixels):
            return "LINESTRING (" + ", ".join("%.8f %.8f" % view.to_lonlat(*p) for p in pixels) + ")"
        c = (600, 600)
        ends = {"S": (600, 1150), "N": (600, 50), "W": (50, 600), "E": (1150, 600)}
        rows = ["link_id,from_node_id,to_node_id,geometry,lanes,allowed_uses,from_biway"]
        for i, end in enumerate(ends.values()):
            rows.append(f'{i + 1},9,351,"{wkt(end, c)}",2,auto,1')
            rows.append(f'{i + 5},351,9,"{wkt(c, end)}",2,auto,1')
        (root / "link.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (root / "node.csv").write_text("node_id,x_coord,y_coord\n351,%.8f,%.8f\n" % view.to_lonlat(*c), encoding="utf-8")
        config = {"schema": "autoloop-site-1", "site_id": "t", "node_id": 351,
                  "satellite": {"path": "sat.png", "georeference": {"kind": "viewport", "ground_mpp": .1, "viewport": view.serialize()}},
                  "network": {"node_csv": "node.csv", "link_csv": "link.csv"}, "policy": {"anchor": False, **policy}}
        write_json(root / "site.json", config)
        return root

    def test_street_view_is_shown_while_tracing_its_approach(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            view = Viewport(1200, 1200, (-12461000.0, 3951400.0, -12460880.0, 3951520.0), "synthetic test viewport")
            Image.new("RGB", (320, 320), (120, 120, 120)).save(root / "gsv.png")
            lon, lat = view.to_lonlat(636, 760)   # on the NB lanes, 16 m south of the centre, facing north
            write_json(root / "base_config.json", {"sources": {"gsv_NB_near_forward": {
                "kind": "image", "role": "gsv", "direction": "NB", "reverse_view": False, "sampling_position": "near", "path": "gsv.png",
                "actual_lat": lat, "actual_lon": lon, "compass_heading_deg": 0.0, "capture_date": "2025-10",
                "view_settings": {"pitch": -12, "fov": 90, "size": [320, 320]}}}})
            vlm = ScriptedVLM(misses_one_line)
            run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertIn("Panel S is a street view from camera C", vlm.prompts["draft_in_NB"])
            self.assertIn("(S) the street view taken from camera C (camera inside the crop)", vlm.prompts["review_in_NB_r1"])
            self.assertNotIn("Panel S", vlm.prompts["draft_in_SB"])       # no street view of that approach
            self.assertNotIn("Panel S", vlm.prompts["draft_out_NB"])      # exits are traced from the satellite crop alone
            panels = read_json(root / "run/geometry/in_NB/round_1/input.panels.json")
            self.assertEqual([p["image_id"] for p in panels], ["boundaries", "regions", "S"])
            nb = next(s for s in read_json(root / "run/geometry/sections.json") if s["id"] == "in_NB")
            self.assertEqual(nb["street_view"]["id"], "gsv_NB_near_forward")

    def test_a_later_finding_resumes_the_review_from_the_final_strips(self):
        from movement_fixer.autoloop.geometry_stage import plan_site, run_section
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            vlm = ScriptedVLM(misses_one_line)
            run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            record = next(h for h in read_json(root / "run/geometry/history.json") if h["section_id"] == "in_NB")
            image, specs, policy, _ = plan_site(read_json(root / "site.json"), root)
            spec = {**next(s for s in specs if s["id"] == "in_NB"), "frame": Frame(tuple(record["frame"]["center"]), 0.0,
                                                                                  record["frame"]["width"], record["frame"]["height"])}
            finding = {"key": "street_view_count:5:4", "code": "street_view_count", "target": "in_NB", "hint": "street view shows five", "detail": {}}
            before = vlm.calls
            _, rerun = run_section(image, spec, vlm, vlm, root / "run/geometry", policy, lambda: True, start=record["final"], extra_issues=[finding])
            self.assertNotIn("draft_in_NB", vlm.stages[before:])                 # no new draft
            self.assertIn("recheck_review_in_NB_r1", vlm.stages[before:])
            self.assertIn("- key street_view_count:5:4 | in_NB | street view shows five", vlm.prompts["recheck_review_in_NB_r1"])
            self.assertTrue((root / "run/geometry/in_NB/recheck/round_1/review.json").is_file())
            self.assertEqual(len(rerun["final"]["edges"]), 6)                     # the scripted reviewer split r2 once more

    def test_anchoring_places_the_window_at_the_stop_line(self):
        def respond(name, prompt):
            if name == "anchor_in_NB":  # stop bar 15 m south of the centre; NB lanes from the axis to 7.2 m east of it
                return {"stop_line": {"y": 400, "kind": "stop_bar", "evidence": "white bar"},   # probe starts 250 px past the node
                        "cross_sections": [{"y": 400, "left_x": 200, "right_x": 272, "evidence": "yellow line to kerb"},
                                           {"y": 650, "left_x": 200, "right_x": 272, "evidence": "yellow line to kerb"}],
                        "confidence": "high", "notes": ""}
            if name.startswith("anchor_"):
                return {"stop_line": {"y": None, "kind": "not_visible", "evidence": "plain grey"},
                        "cross_sections": [{"y": 400, "left_x": 200, "right_x": 272, "evidence": "-"},
                                           {"y": 650 if "in_" in name else 150, "left_x": 200, "right_x": 272, "evidence": "-"}],
                        "confidence": "low", "notes": "nothing visible"}
            return misses_one_line(name, prompt)
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp, anchor=True)
            vlm = ScriptedVLM(respond)
            summary = run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertEqual(vlm.calls, 8 + 24)
            self.assertEqual(summary["anchoring"]["in_NB"], "anchored")
            # the exit on in_NB's arm takes in_NB's stop line (same crosswalk); elsewhere no line was seen, so the window
            # follows the reported carriageway but keeps the network start
            self.assertEqual(summary["anchoring"]["out_SB"], "from_arm_partner")
            self.assertEqual({k for k, v in summary["anchoring"].items() if v == "aligned_only"}, set(summary["anchoring"]) - {"in_NB", "out_SB"})
            exit_sb = next(a for a in read_json(root / "run/geometry/anchors.json") if a["section_id"] == "out_SB")
            self.assertAlmostEqual(exit_sb["start_m"], 15.0, delta=0.2)
            record = next(h for h in read_json(root / "run/geometry/history.json") if h["section_id"] == "in_NB")
            # 1 m beyond the stop line (y=750), 28 m long: centre 15 m further south, on the lanes' middle 3.6 m east of the axis
            for got, want in zip(record["frame"]["center"], (636.0, 900.0)):
                self.assertAlmostEqual(got, want, delta=0.05)
            self.assertEqual((record["frame"]["heading_deg"], record["frame"]["width"]), (0.0, 212))
            frame = Frame(tuple(record["frame"]["center"]), 0.0, record["frame"]["width"], record["frame"]["height"])
            self.assertAlmostEqual(frame.to_source(0, record["stations"][0])[1], 760.0)
            nb = next(s for s in read_json(root / "run/geometry/sections.json") if s["id"] == "in_NB")
            self.assertEqual(nb["window_anchor"]["line_kind"], "stop_bar")

    def test_plan_only_contacts_no_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            result = run_site(root / "site.json", root, root / "run", plan_only=True, clients=(None, None))
            self.assertEqual((result["state"], len(result["sections"])), ("planned", 8))
            self.assertEqual(result["budget"], {"sections": 8, "anchoring_calls": 0, "typical_calls": [16, 32], "stages_upper_bound": 40,
                                                "with_one_repair_per_stage": 80, "hard_cap": 48})
            self.assertTrue((root / "run/geometry/plan.png").is_file())
            self.assertFalse((root / "run/geometry/sections.json").exists())

    def test_review_restores_a_missed_boundary_in_every_section(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            vlm = ScriptedVLM(misses_one_line)
            summary = run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertEqual(summary["status_counts"], {"accepted_after_revision": 8})
            self.assertEqual((summary["vlm_calls_this_run"], vlm.calls), (24, 24))  # draft + two reviews, eight sections
            self.assertEqual(set(summary["stop_reasons"].values()), {"accepted"})
            sections = read_json(root / "run/geometry/sections.json")
            for section in sections:
                self.assertEqual((len(section["regions"]), section["geometry_source"]), (4, "vlm_draft_review_loop"))
                self.assertTrue(all(0 <= x < 1200 and 0 <= y < 1200 for r in section["regions"] for x, y in r["polygon"]))
            nb = next(s for s in sections if s["id"] == "in_NB")
            # Crop is 212 px wide, centred 36 px right of the axis at x=600: crop x=34 is source x=564.
            self.assertEqual([r["polygon"][0][0] for r in nb["regions"]], [564.0, 600.0, 636.0, 672.0])
            self.assertEqual(nb["rotation_ccw"], 0)
            record = next(h for h in read_json(root / "run/geometry/history.json") if h["section_id"] == "in_NB")
            self.assertEqual((len(record["draft"]["edges"]), len(record["final"]["edges"]), len(record["rounds"])), (4, 5, 2))
            self.assertEqual(record["final"]["meta"][2]["origin"], "review_r1")
            self.assertIn("possible_merged_lanes:r2", record["rounds"][0]["issues"][0])
            self.assertEqual(record["rounds"][1]["issues"], [])
            for name in ("draft/input.jpg", "round_1/input.jpg", "round_1/review.json", "round_2/strips.json", "result.json"):
                self.assertTrue((root / "run/geometry/in_NB" / name).is_file(), name)
            # Earlier changes are described by position: IDs are renumbered after a split.
            self.assertIn("round 1: added a boundary at x≈106", vlm.prompts["review_in_NB_r2"])
            self.assertNotIn('"region": "r2"', vlm.prompts["review_in_NB_r2"].split("Changes already made")[1].split("Answer every")[0])
            self.assertEqual(record["rounds"][0]["edits_applied"][0]["effect"], "added a boundary at x≈106")
            self.assertFalse(read_json(root / "run/input_manifest.json")["reference_used_in_inference"])

    def test_loop_effect_is_measured_after_the_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            vlm = ScriptedVLM(misses_one_line)
            run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            history = read_json(root / "run/geometry/history.json")
            reference = {"sections": []}
            for record in history:
                frame = Frame(tuple(record["frame"]["center"]), record["frame"]["heading_deg"], record["frame"]["width"], record["frame"]["height"])
                truth = {"stations": record["stations"], "edges": [[float(x)] * 5 for x in TRUE]}
                kind = "inbound_stopbar" if record["section_id"].startswith("in_") else "outbound_receiving"
                reference["sections"].append({"id": record["section_id"], "regions": section_regions(truth, frame, kind)})
            write_json(root / "reference.json", reference)
            result = evaluate_run(root / "run", root / "reference.json", .1)
            self.assertEqual(result["totals"]["draft"], {"reference_regions": 32, "matched": 16, "candidate_regions": 24,
                                                         "traced_without_counterpart": 8})
            self.assertEqual(result["totals"]["final"], {"reference_regions": 32, "matched": 32, "candidate_regions": 32,
                                                         "traced_without_counterpart": 0})
            self.assertEqual((len(result["loop_effect"]["sections_improved"]), result["loop_effect"]["sections_worsened"],
                              result["loop_effect"]["sections_unchanged"]), (8, [], 0))
            self.assertEqual(result["sections"][0]["final"]["mean_edge_error_m"], 0)

    def test_budget_stops_remaining_sections_with_a_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp, max_vlm_calls=7)
            vlm = ScriptedVLM(misses_one_line)
            summary = run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertEqual(vlm.calls, 7)
            reasons = list(summary["stop_reasons"].values())
            self.assertEqual(reasons.count("accepted"), 2)
            self.assertEqual(reasons.count("budget_exhausted"), 6)
            self.assertEqual(summary["status_counts"], {"accepted_after_revision": 2, "unresolved": 6})

    def test_border_contact_widens_the_crop_and_redrafts_once(self):
        def respond(name, prompt):
            if name.startswith("draft_"):
                return boundaries(34, 70, 106, 248) if name.endswith("_w1") else boundaries(34, 70, 106, 210)
            return {"verdict": "accept", "edits": [], "issue_responses": [{"issue": k, "decision": "dismiss", "reason": "kerb"} for k in keys(prompt)]}
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            vlm = ScriptedVLM(respond)
            run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertEqual(vlm.stages[:3], ["draft_in_EB", "draft_in_EB_w1", "review_in_EB_r1"])
            record = read_json(root / "run/geometry/history.json")[0]
            self.assertEqual((record["frame"]["width"], record["events"][0]["event"]), (212 + 2 * 60, "crop_widened"))
            self.assertEqual(record["events"][0]["sides"], ["right"])

    def test_unusable_draft_is_reported_not_invented(self):
        def respond(name, prompt):
            return boundaries(34) if name == "draft_in_NB" else misses_one_line(name, prompt)
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            vlm = ScriptedVLM(respond)
            summary = run_site(root / "site.json", root, root / "run", clients=(vlm, vlm))
            self.assertEqual(summary["stop_reasons"]["in_NB"], "draft_failed")
            nb = next(s for s in read_json(root / "run/geometry/sections.json") if s["id"] == "in_NB")
            self.assertEqual((nb["regions"], nb["review"]["status"]), ([], "unresolved"))
            self.assertEqual(summary["status_counts"], {"accepted_after_revision": 7, "unresolved": 1})

    def test_changed_inputs_need_a_new_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.site(temp)
            run_site(root / "site.json", root, root / "run", plan_only=True)
            config = read_json(root / "site.json")
            config["policy"] = {"max_rounds": 1}
            write_json(root / "site.json", config)
            with self.assertRaises(ValidationError):
                run_site(root / "site.json", root, root / "run", plan_only=True)


class CompareTests(unittest.TestCase):
    def test_identical_and_merged(self):
        frame = Frame((500, 500), 90, 200, 400)
        truth = {"stations": [20., 200., 380.], "edges": [[x] * 3 for x in (30., 80., 130., 180.)]}
        reference = section_regions(truth, frame, "inbound_stopbar")
        same = compare(truth, frame, reference, .1)
        self.assertEqual((same["matched"], same["mean_edge_error_px"], same["count_difference"]), (3, 0, 0))
        merged = compare({"stations": truth["stations"], "edges": [[x] * 3 for x in (30., 130., 184.)]}, frame, reference, .1)
        # The double-width region overlaps r1 and r2 at 0.5 each and matches neither.
        self.assertEqual((merged["matched"], merged["count_difference"], merged["unmatched_reference"]), (1, -1, ["r1", "r2"]))
        self.assertEqual(merged["matches"], [{"candidate": "r2", "reference": "r3", "iou": .926, "edge_error_px": 2.0}])
        self.assertEqual(merged["unmatched_candidate"], ["r1"])

    def test_sections_on_different_road_lengths_are_not_scored(self):
        frame = Frame((500, 500), 0, 200, 400)
        truth = {"stations": [20., 200., 380.], "edges": [[30.] * 3, [80.] * 3]}
        far = section_regions(truth, Frame((500, 2000), 0, 200, 400), "inbound_stopbar")
        self.assertFalse(compare(truth, frame, far)["comparable"])


if __name__ == "__main__":
    unittest.main()
