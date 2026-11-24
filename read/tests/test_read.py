"""Lean Read pipeline: lane regions from a trace answer, and the checks on the trace and lane-use answers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pave_read.autoloop.strips import Frame
from pave_read.hybrid.common import ValidationError
from pave_read.harness import geometry, stages


def window(inbound=True, outbound=True):
    sections = {}
    if inbound:
        sections["inbound"] = {"section_id": "in_NB", "direction": "NB", "travel_heading_deg": 0.0, "link_id": 1}
    if outbound:
        sections["outbound"] = {"section_id": "out_SB", "direction": "SB", "travel_heading_deg": 180.0, "link_id": 2}
    return {"arm": "S", "frame": Frame((500.0, 800.0), 0.0, 400, 600), "stations": [100.0, 200.0, 300.0, 400.0],
            "stations_m": [10, 20, 30, 40], "mpp": 0.1, "sections": sections, "driving_side": "right"}


def trace(**kw):
    # left curb | out lane | out lane | centre | in lane (pocket from S2) | in lane | curb
    value = {"stop_line": {"y_centre": 50.0, "y_curb": 50.0}, "exit_start_y": 60.0,
             "boundaries": [{"kind": "curb_or_pavement_edge", "xs": [20, 20, 20, 20]},
                            {"kind": "lane_line", "xs": [80, 80, 80, 80]},
                            {"kind": "centre_line", "xs": [140, 140, 140, 140]},
                            {"kind": "lane_line", "xs": [200, 200, None, None]},
                            {"kind": "lane_line", "xs": [260, 260, 260, 260]},
                            {"kind": "curb_or_pavement_edge", "xs": [320, 320, 320, 320]}],
             "strips": [{"carriageway": "outbound", "surface_type": "motor_vehicle_lane", "confidence": "high"},
                        {"carriageway": "outbound", "surface_type": "motor_vehicle_lane", "confidence": "high"},
                        {"carriageway": "inbound", "surface_type": "motor_vehicle_lane", "confidence": "high"},
                        {"carriageway": "inbound", "surface_type": "motor_vehicle_lane", "confidence": "high"},
                        {"carriageway": "inbound", "surface_type": "bicycle_lane", "confidence": "high"}]}
    value.update(kw)
    return value


class Regions(unittest.TestCase):
    def test_inbound_numbered_from_the_centre_outbound_from_the_centre_too(self):
        found = geometry.regions(trace(), window())
        inbound = geometry.motor_regions(found, "inbound")
        outbound = geometry.motor_regions(found, "outbound")
        self.assertEqual([r["strip"] for r in inbound], [2, 3])      # x 140-200 then 200-260: left to right
        self.assertEqual([r["strip"] for r in outbound], [1, 0])     # moving down: centre side (x 80-140) first
        self.assertEqual([r["region_id"] for r in inbound], ["r1", "r2"])

    def test_a_pocket_counts_where_it_reaches_the_stop_line(self):
        found = geometry.regions(trace(), window())
        pocket = next(r for r in found["inbound"] if r["strip"] == 2)
        self.assertTrue(pocket["at_gate"])
        self.assertEqual(pocket["rows"], [0, 1])

    def test_a_lane_that_ends_before_the_stop_line_is_not_counted(self):
        t = trace()
        t["boundaries"][3]["xs"] = [None, None, 200, 200]  # the line exists only upstream
        found = geometry.regions(t, window())
        self.assertEqual(len(geometry.motor_regions(found, "inbound")), 1)

    def test_exit_read_where_its_lines_start(self):
        t = trace()
        t["boundaries"][1]["xs"] = [None, None, 80, 80]  # the exit lane line starts downstream
        found = geometry.regions(t, window())
        self.assertEqual(len(geometry.motor_regions(found, "outbound")), 2)

    def test_no_region_joins_the_two_carriageways(self):
        t = trace()
        t["boundaries"][2]["xs"] = [None, None, 140, 140]  # centre line missing near the junction
        t["boundaries"][1]["xs"] = [None, None, 80, 80]
        found = geometry.regions(t, window())
        self.assertTrue(all(r["width_m"] < 10 for side in ("inbound", "outbound") for r in found[side]))

    def test_skewed_stop_line_keeps_the_lanes_whose_stop_bar_lies_further_back(self):
        # The stop bar meets the road-centre side at y=150 and the curb at y=50: the centre-side lane lines start only
        # past y=150 (station 2), the curbside ones past y=50. A single gate at y=50 would drop the centre-side lane.
        t = trace(stop_line={"y_centre": 150.0, "y_curb": 50.0})
        t["boundaries"][2]["xs"] = [None, 140, 140, 140]  # centre line
        t["boundaries"][3]["xs"] = [None, 200, 200, 200]
        t["boundaries"][1]["xs"] = [None, 80, 80, 80]
        t["boundaries"][0]["xs"] = [None, 20, 20, 20]
        found = geometry.regions(t, window())
        inbound = geometry.motor_regions(found, "inbound")
        self.assertEqual([r["strip"] for r in inbound], [2, 3])
        self.assertTrue(all(2.5 <= r["width_m"] <= 7 for r in inbound))

    def test_one_stop_line_y_reads_as_a_square_stop_bar(self):
        old = trace()
        old.pop("stop_line")
        old["stop_line_y"] = 50.0
        self.assertEqual([r["strip"] for r in geometry.motor_regions(geometry.regions(old, window()), "inbound")],
                         [r["strip"] for r in geometry.motor_regions(geometry.regions(trace(), window()), "inbound")])

    def test_bicycle_lane_is_not_a_motor_region(self):
        found = geometry.regions(trace(), window())
        self.assertTrue(all(r["surface_type"] != "bicycle_lane" for r in geometry.motor_regions(found, "inbound")))

    def test_bicycle_lane_shared_by_turning_traffic_is_a_motor_region(self):
        t = trace()
        t["strips"][4]["surface_type"] = "shared_bicycle_motor_lane"
        inbound = geometry.motor_regions(geometry.regions(t, window()), "inbound")
        self.assertEqual([r["region_id"] for r in inbound], ["r1", "r2", "r3"])
        self.assertEqual(inbound[-1]["surface_type"], "shared_bicycle_motor_lane")

    def test_cross_section_names_the_strips_between_lanes(self):
        text = stages.cross_section(geometry.regions(trace(), window()))
        self.assertEqual(text.split(" | ")[0].split()[0], "r1")
        self.assertIn("bicycle lane 6 m (no r#)", text)


class Issues(unittest.TestCase):
    def test_low_confidence_and_wide_lanes_are_flagged(self):
        t = trace()
        t["strips"][3]["confidence"] = "low"
        found = geometry.regions(t, window())
        issues = geometry.trace_issues(found, window()["sections"], 2.5, 5.0)
        self.assertTrue(any("low-confidence" in i for i in issues))
        t["boundaries"][4]["xs"] = [300, 300, 300, 300]
        t["boundaries"][5]["xs"] = [330, 330, 330, 330]
        found = geometry.regions(t, window())
        self.assertTrue(any("wider than one lane" in i for i in geometry.width_issues(found, window()["sections"], 2.5, 5.0)))


class TraceCheck(unittest.TestCase):
    def test_valid(self):
        stages.validate_trace(trace(), window())

    def test_crossing_boundaries_rejected(self):
        t = trace()
        t["boundaries"][1]["xs"] = [150, 80, 80, 80]
        with self.assertRaises(ValidationError):
            stages.validate_trace(t, window())

    def test_strip_count_must_match(self):
        t = trace()
        t["strips"] = t["strips"][:-1]
        with self.assertRaises(ValidationError):
            stages.validate_trace(t, window())

    def test_stop_line_needs_both_ends(self):
        with self.assertRaises(ValidationError):
            stages.validate_trace(trace(stop_line={"y_centre": 50.0}), window())

    def test_trace_recorded_with_kerb_names_still_reads(self):
        # An answer may name the curb end y_kerb and the curb edge kerb_or_pavement_edge.
        t = trace(stop_line={"y_centre": 150.0, "y_kerb": 50.0})
        for b in t["boundaries"]:
            if b["kind"] == "curb_or_pavement_edge":
                b["kind"] = "kerb_or_pavement_edge"
        self.assertEqual(geometry.stop_line_ends(t), (150.0, 50.0))
        stages.validate_trace(t, window())
        self.assertEqual(t["stop_line"], {"y_centre": 150.0, "y_curb": 50.0})
        self.assertEqual(t["boundaries"][0]["kind"], "curb_or_pavement_edge")

    def test_no_exit_start_on_an_arm_without_outbound(self):
        t = trace()
        with self.assertRaises(ValidationError):
            stages.validate_trace(t, window(outbound=False))


class UseCheck(unittest.TestCase):
    lanes = [{"region_id": "r1"}, {"region_id": "r2"}]
    exits = [{"section_id": "out_NB", "regions": ["r1", "r2"]}, {"section_id": "out_EB", "regions": ["r1"]}]

    def answer(self, **kw):
        value = {"lanes": [{"region_id": "r1", "exists": "yes", "turns": ["left"], "basis": "observed", "confidence": "high", "evidence": ""},
                           {"region_id": "r2", "exists": "yes", "turns": ["through", "right"], "basis": "inferred", "confidence": "medium", "evidence": ""}],
                 "movements": [{"turn": "left", "to": "out_EB", "status": "candidate", "lane_pairs": [["r1", "r1"]]},
                               {"turn": "through", "to": "out_NB", "status": "candidate", "lane_pairs": [["r2", "r1"]]},
                               {"turn": "right", "to": None, "status": "unresolved", "lane_pairs": []},
                               {"turn": "u_turn", "to": None, "status": "not_proposed", "lane_pairs": []}],
                 "detail_requests": [], "temporary_control": {"seen": "no", "lanes": [], "effect": "", "evidence": ""}}
        value.update(kw)
        return value

    def test_valid(self):
        stages.validate_use(self.answer(), self.lanes, self.exits, {"approach"})

    def test_lane_allowing_a_turn_must_feed_it(self):
        a = self.answer()
        a["lanes"][0]["turns"] = ["left", "through"]
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"})

    def test_pair_from_a_lane_that_does_not_allow_the_turn(self):
        a = self.answer()
        a["movements"][1]["lane_pairs"] = [["r1", "r1"], ["r2", "r1"]]
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"})

    def test_a_lane_that_does_not_exist_allows_no_turn(self):
        a = self.answer()
        a["lanes"][1]["exists"] = "no"
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"})

    def test_no_detail_requests_on_the_second_look(self):
        a = self.answer(detail_requests=[{"source": "approach", "bbox_xyxy": [0, 0, .5, .5]}])
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"}, allow_details=False)

    def test_missing_lanes_need_where_and_evidence(self):
        stages.validate_use(self.answer(missing_lanes=[{"where": "between r2 and the curb", "evidence": "a car queues there"}]),
                            self.lanes, self.exits, {"approach"})
        with self.assertRaises(ValidationError):
            stages.validate_use(self.answer(missing_lanes=[{"where": "", "evidence": "x"}]), self.lanes, self.exits, {"approach"})
        with self.assertRaises(ValidationError):
            stages.validate_use(self.answer(missing_lanes="none"), self.lanes, self.exits, {"approach"})

    def test_missing_lane_report_goes_into_the_retrace_prompt(self):
        w = window()
        found = geometry.regions(trace(), w)
        prompt = stages.retrace_prompt(w, False, "Test", trace(), found, None, missing=[("between r2 and the curb", "a car queues there")])
        self.assertIn("between r2 and the curb: a car queues there", prompt)
        self.assertNotIn("separate reading", prompt)

    def test_trace_prompt_says_rails_in_a_lane_are_not_boundaries_and_asks_for_a_stop_line(self):
        prompt = stages.trace_prompt(window(), False, "Test")
        self.assertIn("rails of a streetcar or light-rail track laid in a traffic lane are not boundaries", prompt)
        self.assertIn("one shared_rail_motor_lane strip", prompt)
        self.assertIn("Give it whenever this arm has an inbound carriageway", prompt)
        self.assertNotIn("rail edges.", prompt)

    def test_narrow_lane_issue_names_a_traced_line_that_is_not_a_lane_line(self):
        w = window()
        t = trace()
        t["boundaries"].insert(2, {"kind": "lane_line", "xs": [x + 5 if x is not None else None for x in t["boundaries"][1]["xs"]],
                                   "interpolated": [], "visibility": "clear"})
        t["strips"].insert(1, {"carriageway": "inbound", "surface_type": "shared_rail_motor_lane", "confidence": "high", "evidence": "x"})
        issues = geometry.width_issues(geometry.regions(t, w), w["sections"])
        self.assertTrue(any("such as a rail of a streetcar track" in i for i in issues), issues)

    def test_curb_note_only_for_a_lane_as_wide_as_a_lane_beyond_a_bicycle_lane(self):
        found = {"inbound": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "width_m": 3.3},
                             {"region_id": None, "surface_type": "bicycle_lane", "width_m": 1.6},
                             {"region_id": "r2", "surface_type": "motor_vehicle_lane", "width_m": 3.18}]}
        self.assertEqual([r["region_id"] for r in stages.curb_wide(found)], ["r2"])
        found["inbound"][2]["width_m"] = 2.3
        self.assertEqual(stages.curb_wide(found), [])
        no_bike = {"inbound": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "width_m": 3.3},
                               {"region_id": "r2", "surface_type": "motor_vehicle_lane", "width_m": 3.4}]}
        self.assertEqual(stages.curb_wide(no_bike), [])

    def test_use_prompt_unchanged_without_the_new_notes(self):
        lanes = [{"region_id": "r1", "motor_lane_index": 1, "width_m": 3.3, "surface_type": "motor_vehicle_lane"}]
        exits = [{"section_id": "out_WB", "compass": "west", "turn_angle": -90.0, "geometric_hint": "to the left", "regions": ["r1"]}]
        found = {"inbound": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "width_m": 3.3}]}
        plain = stages.use_prompt("NB", "Test", lanes, exits, None, ["C1"], False, found=found)
        self.assertNotIn("Curbside parking ends", plain)
        self.assertNotIn("Look again at the leftmost lane", plain)
        self.assertIn("No separate street-view reading is available.\n\n", plain)
        wide = {"inbound": found["inbound"] + [{"region_id": None, "surface_type": "bicycle_lane", "width_m": 1.6},
                                               {"region_id": "r2", "surface_type": "motor_vehicle_lane", "width_m": 3.2}]}
        curb = stages.use_prompt("NB", "Test", lanes, exits, None, ["C1"], False, found=wide)
        self.assertIn("r2 (3.2 m) lies between a bicycle lane and the curb", curb)
        left = stages.use_prompt("NB", "Test", lanes, exits, None, ["C1"], False, found=found, left_note=exits)
        self.assertIn("An exit lies to the left of this approach (out_WB, heading west)", left)

    def test_wide_bicycle_note_only_between_two_lanes(self):
        mid = {"inbound": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "width_m": 3.3},
                           {"region_id": None, "surface_type": "bicycle_lane", "width_m": 3.2},
                           {"region_id": "r2", "surface_type": "motor_vehicle_lane", "width_m": 3.3}]}
        self.assertEqual(stages.wide_bicycle(mid), [(3.2, "r1", "r2")])
        edge = {"inbound": mid["inbound"][:2]}
        self.assertEqual(stages.wide_bicycle(edge), [])
        mid["inbound"][1]["width_m"] = 1.8
        self.assertEqual(stages.wide_bicycle(mid), [])

    def test_wide_note_names_the_lane_and_its_width(self):
        lanes = [{"region_id": "r1", "motor_lane_index": 1, "width_m": 3.7, "surface_type": "motor_vehicle_lane"}]
        exits = [{"section_id": "out_WB", "compass": "west", "turn_angle": -90.0, "geometric_hint": "to the left", "regions": ["r1"]}]
        found = {"inbound": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "width_m": 3.7}]}
        plain = stages.use_prompt("NB", "Test", lanes, exits, None, ["C1"], False, found=found)
        self.assertNotIn("as wide as a motor lane at the stop line", plain)
        self.assertNotIn("wider than a bicycle lane usually is", plain)
        note = stages.use_prompt("NB", "Test", lanes, exits, None, ["C1"], False, found=found, wide_note=[("r1", 3.68)])
        self.assertIn("An earlier answer judged r1 (3.68 m) not to be a general motor lane, but it is as wide as a motor lane", note)

    def test_shared_inferred_flags_an_inferred_left_through_lane_beside_another_through_lane(self):
        lane = lambda rid, turns, basis="inferred", exists="yes": {"region_id": rid, "turns": turns, "basis": basis, "exists": exists}
        self.assertEqual(stages.shared_inferred({"lanes": [lane("r1", ["left", "through"]), lane("r2", ["through", "right"])]}), ["r1"])
        self.assertEqual(stages.shared_inferred({"lanes": [lane("r1", ["left", "through"], "observed"), lane("r2", ["through"])]}), [])
        self.assertEqual(stages.shared_inferred({"lanes": [lane("r1", ["left", "through"]), lane("r2", ["right"])]}), [])
        self.assertEqual(stages.shared_inferred({"lanes": [lane("r1", ["left", "through"]), lane("r2", ["through"], exists="no")]}), [])

    def test_wide_bicycle_strip_between_lanes_is_a_measured_issue(self):
        w = window()
        t = trace()
        found = geometry.regions(t, w)
        regs = found["inbound"]
        bike = {"region_id": None, "surface_type": "bicycle_lane", "width_m": 3.2}
        found["inbound"] = regs[:1] + [bike] + regs[1:]
        issues = geometry.width_issues(found, w["sections"])
        if len([r for r in regs if r.get("region_id")]) >= 2:
            self.assertTrue(any("traced as a bicycle lane" in i for i in issues), issues)
        bike["width_m"] = 1.8
        self.assertFalse(any("traced as a bicycle lane" in i for i in geometry.width_issues(found, w["sections"])))

    def test_no_left_goes_into_the_retrace_prompt(self):
        w = window()
        found = geometry.regions(trace(), w)
        left = {"section_id": "out_WB", "compass": "west", "turn_angle": -90.0}
        prompt = stages.retrace_prompt(w, False, "Test", trace(), found, None, no_left=[left])
        self.assertIn("exit to the left of this approach (out_WB, heading west, turning -90 deg)", prompt)
        self.assertIn("found no inbound lane that allows a left turn", prompt)
        self.assertNotIn("separate reading", prompt)
        plain = stages.retrace_prompt(w, False, "Test", trace(), found, None)
        self.assertNotIn("exit to the left", plain)

    def test_a_kept_lane_must_exist_and_allow_a_turn(self):
        a = self.answer()
        a["lanes"][1].update(exists="no", turns=[])
        a["movements"][1].update(status="not_proposed", to=None, lane_pairs=[])
        stages.validate_use(a, self.lanes, self.exits, {"approach"})  # without the check a lane may be judged absent
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"}, must_exist=["r2"])
        b = self.answer()
        b["lanes"][1]["exists"] = "uncertain"
        stages.validate_use(b, self.lanes, self.exits, {"approach"}, must_exist=["r2"])
        c = self.answer()
        c["lanes"][1].update(exists="uncertain", turns=[])
        c["movements"][1].update(status="not_proposed", to=None, lane_pairs=[])
        stages.validate_use(c, self.lanes, self.exits, {"approach"})
        with self.assertRaises(ValidationError):
            stages.validate_use(c, self.lanes, self.exits, {"approach"}, must_exist=["r2"])

    def test_keep_all_forbids_dropping_any_lane(self):
        a = self.answer()
        a["lanes"][0].update(exists="no", turns=[])
        a["movements"][0].update(status="not_proposed", to=None, lane_pairs=[])
        stages.validate_use(a, self.lanes, self.exits, {"approach"}, must_exist=["r2"])
        with self.assertRaises(ValidationError):
            stages.validate_use(a, self.lanes, self.exits, {"approach"}, must_exist=["r2"], keep_all=True)

    def test_temporary_control_is_required_and_names_known_lanes(self):
        base = self.answer()
        del base["temporary_control"]
        with self.assertRaises(ValidationError):
            stages.validate_use(base, self.lanes, self.exits, {"approach"})
        ok = self.answer(temporary_control={"seen": "yes", "lanes": ["r2"], "effect": "closed for now", "evidence": "barrels in C2"})
        stages.validate_use(ok, self.lanes, self.exits, {"approach"})
        for bad in ({"seen": "maybe", "lanes": [], "effect": "", "evidence": ""},
                    {"seen": "yes", "lanes": ["r9"], "effect": "", "evidence": ""}, "none"):
            with self.assertRaises(ValidationError):
                stages.validate_use(self.answer(temporary_control=bad), self.lanes, self.exits, {"approach"})

    def test_use_prompt_reads_the_permanent_layout_and_names_kept_lanes_only_when_there_are_some(self):
        w = window()
        found = geometry.regions(trace(), w)
        lanes = geometry.motor_regions(found, "inbound")
        exits = [{"section_id": "out_SB", "compass": "south", "turn_angle": 180.0, "geometric_hint": "back the way the approach came",
                  "regions": ["r1"]}]
        args = ("NB", "Test", lanes, exits, None, [], False)
        plain = stages.use_prompt(*args, found=found)
        self.assertIn(stages.TEMPORARY, plain)
        self.assertIn('"temporary_control"', plain)
        self.assertNotIn("An earlier answer", plain)
        self.assertEqual(plain, stages.use_prompt(*args, found=found, kept=()))
        why = "The separate reading above counted 3 general motor lane(s) at the stop bar, more than the 1 that would remain without r2."
        kept = stages.use_prompt(*args, found=found, kept=["r2"], kept_why=why)
        self.assertIn("An earlier answer judged r2 not to be", kept)
        self.assertIn(why + " So nothing independent", kept)
        self.assertEqual(kept.replace(kept[kept.index("\nAn earlier answer"):kept.index("\n\n1. For every lane")], ""), plain)

    def test_geometric_hint_and_turn_angle(self):
        self.assertEqual(stages.turn_angle(0, 90), 90)
        self.assertEqual(stages.turn_angle(0, 270), -90)
        self.assertEqual(stages.geometric_hint(5), "about straight ahead")
        self.assertEqual(stages.geometric_hint(-90), "to the left")


if __name__ == "__main__":
    unittest.main()
