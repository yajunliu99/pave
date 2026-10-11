import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.fusion.presence import apply_presence,region_role


class PresenceTests(unittest.TestCase):
    def region(self,surface,existence,confidence="high"):
        return apply_presence({"id":"arbitrary_region","model_surface_type":surface,"model_existence":existence,"model_confidence":confidence,"model_use":"unknown"})

    def test_high_confidence_nonroad_is_not_a_lane(self):
        value=self.region("non_road","rejected")
        self.assertEqual(value["graph_role"],"rejected_region")
        self.assertFalse(value["is_motor_lane"])
        self.assertFalse(value["movement_endpoint_eligible"])
        self.assertFalse(value["motor_use_applicable"])
        self.assertEqual(value["model_use"],"unknown")  # Preserve original model output separately from UI applicability.

    def test_confidence_is_not_a_positive_existence_vote(self):
        for confidence in ("high","medium","low"):
            self.assertEqual(self.region("non_road","rejected",confidence)["graph_role"],"rejected_region")

    def test_supported_motor_and_shared_rail_are_lane_candidates(self):
        for surface in ("motor_vehicle_lane","shared_rail_motor_lane"):
            self.assertTrue(self.region(surface,"supported")["movement_endpoint_eligible"])

    def test_uncertain_motor_does_not_become_supported_lane(self):
        value=self.region("motor_vehicle_lane","uncertain")
        self.assertEqual(value["graph_role"],"unresolved_candidate")
        self.assertFalse(value["movement_endpoint_eligible"])
        self.assertIsNone(value["motor_use_applicable"])

    def test_bicycle_and_rail_facilities_are_separate(self):
        for surface in ("bicycle_strip","rail_area"):
            value=self.region(surface,"supported")
            self.assertEqual(value["graph_role"],"non_motor_facility")
            self.assertFalse(value["is_motor_lane"])

    def test_rejection_overrides_motor_like_surface_label(self):
        value=self.region("motor_vehicle_lane","rejected")
        self.assertEqual(value["graph_role"],"rejected_region")
        self.assertFalse(value["movement_endpoint_eligible"])

    def test_buffer_and_parking_are_not_travel_lanes(self):
        for surface in ("median_or_buffer","shoulder_or_parking"):
            self.assertEqual(self.region(surface,"supported")["graph_role"],"rejected_region")

    def test_unknown_stays_a_candidate_not_a_rejection(self):
        self.assertEqual(region_role("unknown","uncertain"),"unresolved_candidate")

    def test_uncertain_nonroad_guess_is_not_silently_rejected(self):
        self.assertEqual(region_role("non_road","uncertain"),"unresolved_candidate")


if __name__=="__main__":unittest.main()
