"""A clear satellite arrow survives an inconclusive street-view check; real contrary evidence still wins."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from movement_fixer.hybrid.evidence_reasoning import fuse_surface_usage, keep_clear_arrow

USE = {"region_id": "r2", "surface_type": "motor_vehicle_lane", "existence": "supported", "surface_reason": "asphalt",
       "use": "unknown", "allowed_turns": [], "basis": "unknown", "confidence": "low", "binding": "crop too worn.", "visual_refs": []}
SEEN = {"surface_type": "motor_vehicle_lane", "existence": "supported", "evidence": "arrow", "observed_arrows": ["left"], "visibility": "clear"}


class KeepClearArrowTests(unittest.TestCase):
    def test_inconclusive_view_does_not_erase_a_clear_arrow(self):
        u = keep_clear_arrow(SEEN, USE)
        self.assertEqual((u["use"], u["allowed_turns"], u["use_audit_prediction"]), ("left_only", ["left"], "unknown"))
        self.assertIn("inconclusive, not contrary", u["binding"])

    def test_contrary_or_weak_evidence_is_left_alone(self):
        for initial, use in ((SEEN, {**USE, "use": "right_only", "allowed_turns": ["right"]}),   # the use audit read something else
                             ({**SEEN, "visibility": "partial"}, USE),                            # the satellite arrow was not clear
                             ({**SEEN, "observed_arrows": []}, USE),                              # nothing was seen
                             (SEEN, {**USE, "surface_type": "bicycle_strip"}),                    # not a motor lane after all
                             (SEEN, {**USE, "existence": "uncertain"})):
            self.assertEqual(keep_clear_arrow(initial, use), use)

    def test_two_arrows_become_shared(self):
        u = keep_clear_arrow({**SEEN, "observed_arrows": ["through", "left"]}, USE)
        self.assertEqual((u["use"], u["allowed_turns"]), ("shared", ["left", "through"]))

    def test_fusion_applies_the_rule_and_numbers_lanes(self):
        section = {"regions": [{"region_id": "r2", "surface_type": "motor_vehicle_lane", "existence": "supported", "evidence": "arrow",
                                "observed_arrows": ["left"], "visibility": "clear"}], "count": {}}
        fused = fuse_surface_usage(section, {"regions": [USE]})
        r = fused["regions"][0]
        self.assertEqual((r["lane_use_prediction"]["use"], r["motor_lane_index"], r["initial_surface_audit"]["observed_arrows"]), ("left_only", 1, ["left"]))


if __name__ == "__main__":
    unittest.main()
