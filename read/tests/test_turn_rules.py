import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop import anchor, render
from movement_fixer.autoloop.strips import Frame
from movement_fixer.hybrid.common import ValidationError
from movement_fixer.hybrid.turn_rules import RULE, apply_kerb_turn_default

SECTIONS = [{"id": f"{p}_{d}", "kind": k, "direction": d, "heading_deg": h}
            for p, k in (("in", "inbound_stopbar"), ("out", "outbound_receiving"))
            for d, h in (("NB", 0), ("EB", 90), ("SB", 180), ("WB", 270))]


def lanes(arrows_on_kerb=()):
    def section(sid, n, kerb_arrows=()):
        regions = [{"region_id": f"r{i}", "motor_lane_index": i, "observed_arrows": [], "lane_use_prediction": {}} for i in range(1, n + 1)]
        regions[-1]["observed_arrows"] = list(kerb_arrows)
        regions.append({"region_id": f"r{n + 1}", "motor_lane_index": None, "surface_type": "bicycle_strip"})
        return {"section_id": sid, "regions": regions}
    return [section("in_NB", 3, arrows_on_kerb), section("out_EB", 2), section("out_NB", 2), section("out_WB", 2),
            section("in_EB", 2), section("in_SB", 2), section("in_WB", 2), section("out_SB", 2)]


def decision(turn, to, status="unresolved", pairs=()):
    return {"turn": turn, "to_direction": to, "from_direction": "NB", "in_section_id": "in_NB", "status": status,
            "lane_pairs": list(pairs), "basis": "inferred", "confidence": "low", "reason": "no arrow seen", "validation_flags": []}


class KerbTurnTests(unittest.TestCase):
    def test_unresolved_right_turn_gets_the_kerbside_lane(self):
        out, notes = apply_kerb_turn_default([decision("right", "EB")], lanes(), SECTIONS)
        right = next(m for m in out if m["from_direction"] == "NB" and m["turn"] == "right")
        self.assertEqual((right["status"], right["default_rule"]), ("candidate", RULE))
        self.assertEqual([(p["ib_lane"], p["ob_lane"]) for p in right["lane_pairs"]], [(3, 2)])  # kerb lane to kerb lane
        self.assertEqual(right["model_decision"]["status"], "unresolved")
        self.assertIn(RULE, right["validation_flags"])

    def test_missing_decision_is_added_when_geometry_names_one_exit(self):
        out, _ = apply_kerb_turn_default([], lanes(), SECTIONS)
        self.assertEqual({(m["from_direction"], m["to_direction"]) for m in out if m["turn"] == "right"},
                         {("NB", "EB"), ("EB", "SB"), ("SB", "WB"), ("WB", "NB")})

    def test_a_through_only_arrow_excludes_the_lane(self):
        out, notes = apply_kerb_turn_default([decision("right", "EB")], lanes(["through"]), SECTIONS)
        self.assertEqual(next(m for m in out if m["from_direction"] == "NB")["status"], "unresolved")
        self.assertIn("through only", next(n for n in notes if n["approach"] == "NB")["why"])

    def test_a_candidate_from_another_lane_gains_the_kerbside_lane(self):
        pair = {"in_region_id": "r2", "out_region_id": "r2", "ib_lane": 2, "ob_lane": 2}
        out, _ = apply_kerb_turn_default([decision("right", "EB", "candidate", [pair])], lanes(), SECTIONS)
        right = next(m for m in out if m["from_direction"] == "NB")
        self.assertEqual([p["ib_lane"] for p in right["lane_pairs"]], [2, 3])

    def test_exit_judged_another_turn_is_left_alone(self):
        out, notes = apply_kerb_turn_default([decision("through", "EB")], lanes(), SECTIONS)
        self.assertFalse(any(m["turn"] == "right" and m["from_direction"] == "NB" for m in out))
        self.assertIn("as through", next(n for n in notes if n["approach"] == "NB")["why"])


class AnchorTests(unittest.TestCase):
    spec = {"id": "in_NB", "kind": "inbound_stopbar", "ground_mpp": 0.1, "node_px": [600, 600], "away_unit": [0, 1],
            "carriageway_m": 7.2, "frame": Frame((600, 900), 0.0, 212, 320)}

    def reading(self, **change):
        value = {"stop_line": {"y": 400, "kind": "stop_bar", "evidence": "bar"},   # probe starts 25 m (250 px) past the node
                 "cross_sections": [{"y": 400, "left_x": 200, "right_x": 272, "evidence": "x"},
                                    {"y": 650, "left_x": 200, "right_x": 272, "evidence": "x"}],
                 "confidence": "medium", "notes": ""}
        value.update(change)
        return value

    def test_readings_are_checked(self):
        probe = anchor.probe_frame(self.spec)
        anchor.validate_anchor(self.reading(), self.spec, probe)
        wide = [{"y": 400, "left_x": 0, "right_x": 399, "evidence": ""}, {"y": 650, "left_x": 200, "right_x": 272, "evidence": ""}]
        above = [{"y": 400, "left_x": 200, "right_x": 272, "evidence": ""}, {"y": 300, "left_x": 200, "right_x": 272, "evidence": ""}]
        for bad in (self.reading(cross_sections=wide), self.reading(cross_sections=above),
                    self.reading(stop_line={"y": None, "kind": "stop_bar", "evidence": ""})):
            with self.assertRaises(ValidationError):
                anchor.validate_anchor(bad, self.spec, probe)

    def test_window_follows_the_carriageway_direction(self):
        probe = anchor.probe_frame(self.spec)
        skew = [{"y": 400, "left_x": 200, "right_x": 272, "evidence": ""}, {"y": 650, "left_x": 250, "right_x": 322, "evidence": ""}]
        frame, info = anchor.anchored_frame(self.spec, probe, self.reading(cross_sections=skew),
                                            {"section_length_m": 28.0, "end_pad_m": 2.0, "lateral_margin_m": 7.0})
        self.assertEqual(info["placed"], "anchored")
        self.assertAlmostEqual(frame.heading_deg, 360 - np.degrees(np.arctan2(50, 250)), places=2)  # travel up and to the left
        policy = {"section_length_m": 28.0, "end_pad_m": 2.0, "lateral_margin_m": 7.0}
        frame, info = anchor.anchored_frame(self.spec, probe, self.reading(confidence="low", cross_sections=skew), policy)
        self.assertEqual(info["placed"], "aligned_only")
        # direction from the reading; start kept at the network window's first station (y = 900 - 160 + 20 = 760)
        self.assertAlmostEqual(frame.heading_deg, 360 - np.degrees(np.arctan2(50, 250)), places=2)
        self.assertAlmostEqual(info["start_px"][1], 760, delta=8)


    def test_one_arm_agrees_on_its_crosswalk(self):
        exit_spec = {**self.spec, "id": "out_SB", "kind": "outbound_receiving", "arm": "S"}
        approach = {**self.spec, "arm": "S"}
        p_in, p_out = anchor.probe_frame(approach), anchor.probe_frame(exit_spec)
        # outbound probe: junction toward the bottom; 15 m from the node is 150 px above the node row
        out_node_y = p_out.height - 250
        exit_line = lambda y, conf: {"stop_line": {"y": y, "kind": "crosswalk_edge", "evidence": ""}, "confidence": conf, "notes": "",
                                     "cross_sections": [{"y": y, "left_x": 128, "right_x": 200, "evidence": ""},
                                                        {"y": y - 250, "left_x": 128, "right_x": 200, "evidence": ""}]}
        # the exit's reading disagrees by 20 m at the same confidence: the approach's stop bar wins
        starts = anchor.harmonise([(approach, p_in, self.reading()), (exit_spec, p_out, exit_line(out_node_y - 350, "medium"))])
        self.assertEqual(starts["out_SB"][1], "from_arm_partner")
        self.assertAlmostEqual(starts["out_SB"][0], 15.0, delta=0.2)
        # a higher-confidence exit line wins instead
        starts = anchor.harmonise([(approach, p_in, self.reading()), (exit_spec, p_out, exit_line(out_node_y - 350, "high"))])
        self.assertEqual((starts["out_SB"][1], starts["in_NB"][1]), ("anchored", "from_arm_partner"))
        self.assertAlmostEqual(starts["in_NB"][0], 35.0, delta=0.2)


class ShadeTests(unittest.TestCase):
    def test_only_substantial_shadow_gets_a_lightened_copy(self):
        sunny = Image.new("RGB", (200, 200), (150, 150, 150))
        self.assertIsNone(render.shade_lifted(sunny))
        half = np.full((200, 200, 3), 150, np.uint8)
        half[:, 100:] = 60
        lifted = render.shade_lifted(Image.fromarray(half))
        self.assertGreater(np.asarray(lifted)[100, 180].mean(), 100)
        self.assertEqual(np.asarray(lifted)[100, 20].tolist(), [150, 150, 150])


if __name__ == "__main__":
    unittest.main()
