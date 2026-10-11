import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from movement_fixer.autoloop.frames import road_arms
from movement_fixer.hybrid import legs
from movement_fixer.hybrid.common import ValidationError
from movement_fixer.hybrid.evidence_reasoning import evidence_movement_prompt
from movement_fixer.hybrid.geometry_checks import audit_geometry
from movement_fixer.hybrid.validation import DIRECTIONS, compass_receiving, validate_movement_output
from test_hybrid_pipeline import lanes


def section(sid, heading, arm=None):
    kind = "inbound_stopbar" if sid.startswith("in_") else "outbound_receiving"
    s = {"id": sid, "kind": kind, "direction": sid.split("_", 1)[1], "heading_deg": heading}
    return {**s, "arm": arm} if arm else s


def tee():
    """West, east and south arms; the south arm is the stem."""
    return [section("in_EB", 90, "W"), section("out_WB", 270, "W"), section("in_WB", 270, "E"), section("out_EB", 90, "E"),
            section("in_NB", 0, "S"), section("out_SB", 180, "S")]


def skewed():
    """A cross whose north-east leg leaves 35 degrees off straight ahead of northbound traffic."""
    return [section("in_NB", 0, "S"), section("out_SB", 180, "S"), section("out_NEB", 35, "NE"), section("in_SWB", 215, "NE"),
            section("out_WB", 270, "W"), section("in_EB", 90, "W")]


def decision(turn, to):
    return {"turn": turn, "to_direction": to, "status": "unresolved", "lane_pairs": [], "basis": "inferred", "confidence": "low",
            "evidence_refs": [], "assumptions": [], "reason": "fixture"}


def outbound(*directions):
    return [lanes("out_" + d, ["motor_vehicle_lane"]) for d in directions]


class FourWayTests(unittest.TestCase):
    def setUp(self):
        # Traced headings are never exactly cardinal.
        heading = {"NB": 359.4, "EB": 89.1, "SB": 180.6, "WB": 270.9}
        self.sections = [section(p + d, h) for d, h in heading.items() for p in ("in_", "out_")]

    def test_same_options_as_the_compass_rotation(self):
        for a in legs.approaches(self.sections):
            options = legs.receiving(self.sections, a)
            self.assertEqual(options, compass_receiving(a["direction"]))
            self.assertEqual(list(options), list(DIRECTIONS))

    def test_prompt_text_is_unchanged_for_a_plain_cross(self):
        # Cached model answers of four-way junctions stay valid only if the prompt is byte-identical.
        byid = {s["id"]: s for s in self.sections}
        for d in DIRECTIONS:
            options = legs.receiving(self.sections, byid["in_" + d])
            new = evidence_movement_prompt(d, [], {}, [], {"NB": {}}, options, legs.describe(self.sections, byid["in_" + d], options))
            self.assertEqual(new, evidence_movement_prompt(d, [], {}, [], {"NB": {}}))
            self.assertIn("Report all four turns once.", new)

    def test_without_headings_compass_names_still_work(self):
        plain = [{"id": p + d, "kind": k, "direction": d} for d in DIRECTIONS for p, k in (("in_", "inbound_stopbar"), ("out_", "outbound_receiving"))]
        self.assertEqual(legs.receiving(plain, plain[0]), compass_receiving("NB"))
        with self.assertRaises(ValidationError):
            legs.heading({"id": "in_link_4", "direction": "link_4"})


class TeeTests(unittest.TestCase):
    def test_stem_has_no_through_movement(self):
        sections = tee()
        stem = next(s for s in sections if s["id"] == "in_NB")
        options = legs.receiving(sections, stem)
        self.assertEqual(options, {"EB": ["right"], "SB": ["u_turn"], "WB": ["left"]})
        self.assertEqual(legs.fixed_turns(options), {"left": "WB", "right": "EB", "u_turn": "SB"})
        prompt = evidence_movement_prompt("NB", [], {}, [], {}, options, legs.describe(sections, stem, options))
        self.assertIn("Report all three turns once.", prompt)
        self.assertIn('"to_direction":"EB|SB|WB"', prompt)

    def test_validator_takes_the_three_decisions(self):
        options = legs.receiving(tee(), tee()[4])
        ss = [lanes("in_NB", ["motor_vehicle_lane"])] + outbound("EB", "SB", "WB")
        good = {"movements": [decision("left", "WB"), decision("right", "EB"), decision("u_turn", "SB")]}
        self.assertEqual(len(validate_movement_output(good, "NB", ss, enforce_turn_constraints=False, receiving=options)), 3)
        for bad in ([decision("left", "WB"), decision("right", "EB")],                          # one receiving section left out
                    [decision("left", "WB"), decision("right", "EB"), decision("through", "SB")],  # no through on the stem
                    [decision("left", "WB"), decision("left", "WB"), decision("u_turn", "SB")]):
            with self.assertRaises(ValidationError):
                validate_movement_output({"movements": bad}, "NB", ss, enforce_turn_constraints=False, receiving=options)

    def test_through_road_pairs_straight_ahead_only(self):
        sections = tee()
        for s in sections:
            s.update(source_id="sat", regions=[{"id": "r1", "polygon": [[0, 0], [10, 0], [10, 40], [0, 40]], "gate_point": [5, 0]}])
        audit = audit_geometry({"sections": sections, "sources": {"sat": {"role": "satellite"}},
                                "geometry_checks": {"reference_regions": ["in_EB:r1"]}})
        self.assertEqual(audit["straight_pairs"]["EB"]["out_section_id"], "out_EB")
        self.assertFalse(audit["straight_pairs"]["NB"]["comparable"])


class SkewTests(unittest.TestCase):
    def test_model_chooses_the_turn_of_a_skewed_leg(self):
        sections = skewed()
        approach = sections[0]
        options = legs.receiving(sections, approach)
        self.assertEqual(options["NEB"], ["through", "right"])
        self.assertIsNone(legs.fixed_turns(options))
        rows = legs.describe(sections, approach, options)
        self.assertEqual(next(r for r in rows if r["to_direction"] == "NEB")["heading_change_deg"], 35)
        prompt = evidence_movement_prompt("NB", [], {}, [], {}, options, rows)
        self.assertIn("Report one decision per receiving section", prompt)
        self.assertIn('"candidate_turns": ["through", "right"]', prompt)
        ss = [lanes("in_NB", ["motor_vehicle_lane"])] + outbound("NEB", "SB", "WB")
        for turn in ("through", "right"):
            chosen = {"movements": [decision(turn, "NEB"), decision("u_turn", "SB"), decision("left", "WB")]}
            validate_movement_output(chosen, "NB", ss, enforce_turn_constraints=False, receiving=options)
        with self.assertRaises(ValidationError):
            validate_movement_output({"movements": [decision("left", "NEB"), decision("u_turn", "SB"), decision("left", "WB")]},
                                     "NB", ss, enforce_turn_constraints=False, receiving=options)

    def test_known_arms_settle_u_turns(self):
        approach = section("in_NB", 0, "S")
        sharp = section("out_SEB", 150, "SE")
        self.assertEqual(legs.receiving([approach, sharp], approach), {"SEB": ["right"]})
        self.assertEqual(legs.receiving([section("in_NB", 0), section("out_SEB", 150)], section("in_NB", 0)), {"SEB": ["right", "u_turn"]})

    def test_look_back_and_upstream(self):
        sections = skewed()
        self.assertEqual(legs.opposing_approach(sections, sections[2])["id"], "in_SWB")
        no_arms = [{k: v for k, v in s.items() if k != "arm"} for s in tee()]
        self.assertEqual(legs.opposing_approach(no_arms, no_arms[1])["id"], "in_EB")
        self.assertTrue(legs.is_upstream(section("in_NEB", 45), -10, -10))
        self.assertFalse(legs.is_upstream(section("in_NEB", 45), 10, 10))


class TolerantFormatTests(unittest.TestCase):
    def test_structured_notes_and_overview_citations_are_kept_not_fatal(self):
        options = legs.receiving(tee(), tee()[4])
        ss = [lanes("in_NB", ["motor_vehicle_lane"])] + outbound("EB", "SB", "WB")
        left = {**decision("left", "WB"), "status": "candidate", "lane_pairs": [{"in_region_id": "r1", "out_region_id": "r1"}],
                "evidence_refs": ["full_satellite:left_arrows", "in_NB:r1"]}
        value = {"movements": [left, decision("right", "EB"), decision("u_turn", "SB")], "gsv_alignment_notes": [{"view": "near", "note": "shifted"}]}
        result = validate_movement_output(value, "NB", ss, enforce_turn_constraints=False, receiving=options)
        self.assertEqual(result[0]["evidence_refs"], ["in_NB:r1"])
        self.assertEqual(result[0]["unlisted_evidence_refs"], ["full_satellite:left_arrows"])
        self.assertIn("unlisted_evidence_reference", result[0]["validation_flags"])
        self.assertIsInstance(result[0]["gsv_alignment_notes"][0], str)
        only_overview = {**left, "evidence_refs": ["full_satellite:left_arrows"]}
        with self.assertRaises(ValidationError):  # the overview alone cannot support a candidate
            validate_movement_output({"movements": [only_overview, decision("right", "EB"), decision("u_turn", "SB")]}, "NB", ss,
                                     enforce_turn_constraints=False, receiving=options)


class UntracedReceivingTests(unittest.TestCase):
    def test_decision_toward_an_untraced_exit_is_dropped_not_fatal(self):
        options = {"WB": ["left"], "SB": ["u_turn"]}  # out_EB could not be traced, so the right turn is not offered
        ss = [lanes("in_NB", ["motor_vehicle_lane"])] + outbound("SB", "WB")
        value = {"movements": [decision("left", "WB"), decision("u_turn", "SB"), decision("right", "EB")]}
        result = validate_movement_output(value, "NB", ss, enforce_turn_constraints=False, receiving=options)
        self.assertEqual([m["to_direction"] for m in result], ["WB", "SB"])
        self.assertEqual(result[0]["dropped_decisions"][0]["to_direction"], "EB")
        paired = {**decision("right", "EB"), "status": "candidate", "lane_pairs": [{"in_region_id": "r1", "out_region_id": "r1"}]}
        with self.assertRaises(ValidationError):  # pairing lanes with a section that is not there is still an error
            validate_movement_output({"movements": [decision("left", "WB"), decision("u_turn", "SB"), paired]}, "NB", ss,
                                     enforce_turn_constraints=False, receiving=options)
        prompt = evidence_movement_prompt("NB", [], {}, [], {}, options, [], ["out_EB"])
        self.assertIn("out_EB exist in the road network but could not be traced", prompt)


class LegCheckTests(unittest.TestCase):
    def test_sections_must_match_network_legs(self):
        tee_legs = {"EB": {"in_link_id": 1, "out_link_id": 2}, "WB": {"in_link_id": 3, "out_link_id": 4},
                    "NB": {"in_link_id": 5}, "SB": {"out_link_id": 6}}
        legs.check_sections(tee(), tee_legs)
        with self.assertRaises(ValidationError):
            legs.check_sections(tee(), {**tee_legs, "NB": {"in_link_id": 5, "out_link_id": 7}})  # no out_NB section
        with self.assertRaises(ValidationError):
            legs.check_sections(tee(), {k: v for k, v in tee_legs.items() if k != "SB"})  # out_SB has no link


class ArmTests(unittest.TestCase):
    def test_legs_group_into_named_arms(self):
        self.assertEqual(road_arms([2, 358, 90, 180, 183, 270]),
                         [("N", 0.0), ("N", 0.0), ("E", 90), ("S", 181.5), ("S", 181.5), ("W", 270)])
        self.assertEqual([a for a, _ in road_arms([23, 66])], ["NE1", "NE2"])


if __name__ == "__main__":
    unittest.main()
