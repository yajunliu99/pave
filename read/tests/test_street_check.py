import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.autoloop.street_check import issue, mismatches, reading_issue, reading_mismatches, street_count

SOURCES = {f"gsv_NB_{p}_forward": {"sampling_position": p, "distance_to_center_m": d, "capture_date": "2025-10"}
           for p, d in (("near", 26), ("mid", 51), ("far", 105))}


def audit(**counts):
    return {"street_view_counts": [{"source_id": f"gsv_NB_{p}_forward", "motor_lanes": n, "settled": s, "unmatched": "r1 is opposing"}
                                   for p, (n, s) in counts.items()]}


class StreetCheckTests(unittest.TestCase):
    def test_count_prefers_a_settled_near_view(self):
        self.assertEqual(street_count(audit(near=(2, True), mid=(3, True)), SOURCES)[:2], (2, "near view, settled"))
        self.assertEqual(street_count(audit(near=(4, False), mid=(4, False)), SOURCES)[:2], (4, "near and mid views agree"))
        self.assertEqual(street_count(audit(near=(4, False), mid=(3, True)), SOURCES)[:2], (3, "mid view, settled"))
        self.assertIsNone(street_count(audit(near=(4, False), mid=(3, False), far=(2, True)), SOURCES)[0])  # far views are left out

    def test_only_differing_approaches_are_flagged(self):
        lanes = [{"section_id": "in_NB", "kind": "inbound_stopbar", "direction": "NB", "count": {"model_motor_count": 1},
                  "regions": [{"region_id": "r1", "surface_type": "motor_vehicle_lane", "existence": "supported"}]},
                 {"section_id": "out_NB", "kind": "outbound_receiving", "direction": "NB", "count": {"model_motor_count": 1}, "regions": []}]
        found = mismatches(lanes, {"NB": audit(near=(2, True))}, {"sources": SOURCES})
        self.assertEqual([(m["section_id"], m["street_view_lanes"], m["traced_motor_lanes"]) for m in found], [("in_NB", 2, 1)])
        self.assertEqual(mismatches(lanes, {"NB": audit(near=(1, True))}, {"sources": SOURCES}), [])
        made = issue(found[0])
        self.assertEqual((made["key"], made["target"], made["code"]), ("street_view_count:2:1", "in_NB", "street_view_count"))
        self.assertIn("a lane with no strip of its own", made["hint"])
        self.assertEqual(made["detail"]["street_views"]["near"]["unmatched"], "r1 is opposing")

    def test_settled_independent_reading_that_differs_is_flagged(self):
        lanes = [{"section_id": "in_WB", "kind": "inbound_stopbar", "direction": "WB", "count": {"model_motor_count": 6},
                  "regions": [{"region_id": f"r{i}", "surface_type": "motor_vehicle_lane", "existence": "supported"} for i in range(1, 7)]}]
        def reading(n, settled=True, confidence="medium"):
            return [{"approach": "WB", "layout_seen": "L T T R +1 bike", "reading": {
                "lanes": [{"lane": i + 1, "turns": [], "evidence": f"lane {i + 1}"} for i in range(n)], "bicycle_lanes": 1,
                "count_settled": settled, "count_confidence": confidence, "limitations": "shadow"}}]
        found = reading_mismatches(lanes, reading(4))
        self.assertEqual([(m["section_id"], m["reading_lanes"], m["traced_motor_lanes"]) for m in found], [("in_WB", 4, 6)])
        for quiet in (reading(6), reading(4, settled=False), reading(4, confidence="low"), []):
            self.assertEqual(reading_mismatches(lanes, quiet), [])
        made = reading_issue(found[0])
        self.assertEqual((made["key"], made["code"], made["target"]), ("independent_reading_count:4:6", "independent_reading_count", "in_WB"))
        self.assertIn("two strips over one lane", made["hint"])
        self.assertIn("L T T R +1 bike", made["hint"])
        self.assertEqual(made["detail"]["reading_other_facilities"]["bicycle_lanes"], 1)
        self.assertNotIn("UTDF", made["hint"])


if __name__ == "__main__":
    unittest.main()
