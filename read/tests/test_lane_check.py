import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.fusion.lane_check import (blind_sheet, check_prompt, disagreements, layout, reference_lanes, reference_name,
                                              validate_reading, verdict)
from movement_fixer.hybrid.common import ValidationError


def reading(*turns, settled=True, confidence="medium", **extra):
    return {"lanes": [{"lane": i + 1, "turns": list(t), "evidence": "arrow"} for i, t in enumerate(turns)],
            "count_settled": settled, "count_confidence": confidence, **extra}


class LaneCheckTests(unittest.TestCase):
    def test_reference_lanes_are_read_left_to_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            with open(path, "w", newline="") as stream:
                w = csv.DictWriter(stream, fieldnames=["utdf_intid", "utdf_mvmt", "ib_link_id", "type", "start_ib_lane", "end_ib_lane"])
                w.writeheader()
                w.writerows([{"utdf_intid": 1, "utdf_mvmt": "NBL", "ib_link_id": 7, "type": "left", "start_ib_lane": 1, "end_ib_lane": 2},
                             {"utdf_intid": 1, "utdf_mvmt": "NBT", "ib_link_id": 7, "type": "thru", "start_ib_lane": 3, "end_ib_lane": 3},
                             {"utdf_intid": 1, "utdf_mvmt": "NBR", "ib_link_id": 7, "type": "right", "start_ib_lane": 3, "end_ib_lane": 3},
                             {"utdf_intid": 1, "utdf_mvmt": "NWR", "ib_link_id": 7, "type": "right", "start_ib_lane": 1, "end_ib_lane": 1},
                             {"utdf_intid": 1, "utdf_mvmt": "SBT", "ib_link_id": "", "type": "thru", "start_ib_lane": 1, "end_ib_lane": 1}])
            self.assertEqual(reference_lanes(path), {"7": {"NB": ["L", "L", "TR"], "NW": ["R"]}})

    def test_only_differing_approaches_are_checked(self):
        config = {"legs": {"NB": {"in_link_id": 7}, "SB": {"in_link_id": 8}}}
        lanes = [{"section_id": "in_NB", "count": {"model_motor_count": 2}}, {"section_id": "in_SB", "count": {"model_motor_count": 3}}]
        found = disagreements(config, lanes, {"7": {"NB": ["L", "L", "TR"]}, "8": {"SB": ["L", "T", "T"]}})
        self.assertEqual([(d["approach"], d["vlm_count"], d["reference_count"]) for d in found], [("NB", 2, 3)])

    def test_shared_link_compares_the_approach_of_the_same_name(self):
        config = {"legs": {"SB": {"in_link_id": 8}, "EB": {"in_link_id": 9}}}
        lanes = [{"section_id": "in_SB", "count": {"model_motor_count": 2}}, {"section_id": "in_EB", "count": {"model_motor_count": 1}}]
        found = disagreements(config, lanes, {"8": {"SB": ["LT", "T"], "SW": ["LR"]}, "9": {"NE": ["L", "T"]}})
        self.assertEqual([(d["approach"], d["reference_name"], d["reference_also_on_link"]) for d in found], [("EB", "NE", [])])
        self.assertEqual(reference_name(["SW", "SB"], "SB"), "SB")
        self.assertIsNone(reference_name([], "SB"))

    def test_prompt_and_sheet_carry_no_answer(self):
        prompt = check_prompt(["mid: 50 m from the centre, 2025-11"])
        for leak in ("UTDF", "reference", "model found", "VLM"):
            self.assertNotIn(leak, prompt)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("sat.png", "v.jpg"):
                Image.new("RGB", (300, 400), "gray").save(Path(tmp) / name)
            path = blind_sheet(Path(tmp) / "sat.png", [(Path(tmp) / "v.jpg", "mid: 50 m")], Path(tmp) / "out" / "NB.jpg")
            self.assertTrue(path.exists())

    def test_reading_validation(self):
        validate_reading(reading(["left"], ["through"], ["through", "right"]))
        with self.assertRaises(ValidationError):
            validate_reading({**reading(["left"]), "lanes": [{"lane": 2, "turns": ["left"], "evidence": ""}]})
        with self.assertRaises(ValidationError):
            validate_reading({**reading(["left"]), "count_settled": "yes"})
        with self.assertRaises(ValidationError):
            validate_reading(reading(["L"]))

    def test_verdicts(self):
        utdf = ["L", "T", "T", "R"]
        self.assertEqual(verdict(reading(["left"], ["through"], ["through"], ["right"]), 3, utdf), ("utdf", "model_missed_right_lane"))
        self.assertEqual(verdict(reading(["left"], ["through"], ["through"]), 3, utdf), ("model", "utdf_differs_from_imagery"))
        self.assertEqual(verdict(reading(["left"], ["through"]), 3, utdf), ("neither", "imagery_differs_from_both"))
        self.assertEqual(verdict(reading(["left"], ["through"], ["through"], ["right"], settled=False), 3, utdf)[0], "unclear")
        self.assertEqual(verdict(reading(["left"], ["through"], ["through"], ["right"], confidence="low"), 3, utdf)[0], "unclear")
        self.assertEqual(layout(reading(["left"], [], bicycle_lanes=1)), "L ? +1 bike")


if __name__ == "__main__":
    unittest.main()
