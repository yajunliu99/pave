import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pave_read.fusion.control_acquisition import StreetViewClient  # noqa: E402
from pave_read.fusion.control_sampling import policy_with  # noqa: E402
from pave_read.fusion.controls import detail_pose, needs_resampling, validate_controls  # noqa: E402
from pave_read.hybrid.common import ValidationError, digest, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("run_signals", ROOT / "scripts/pipeline/run_signals.py")
run_signals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_signals)
EXAMPLE = ROOT / "examples/broadway_ames_2021"


def control(**k):
    c = {"id": "c1", "kind": "lane_use_sign", "observed_text": "ONLY", "symbol": "left arrow", "description": "sign over r1",
         "face_status": "front", "readability": "readable", "confidence": "high", "applies_to": "lane_candidates",
         "lane_candidates": ["r1"], "binding_status": "candidate", "binding_reason": "mounted over the left lane",
         "visual_refs": [{"source_id": "gsv_a", "bbox_xyxy": [.1, .1, .3, .3], "finding": "arrow and ONLY"}]}
    return {**c, **k}


class SignalsTest(unittest.TestCase):
    def test_controls_are_validated(self):
        out = validate_controls({"controls": [control()]}, ["r1", "r2"], {"gsv_a"})
        self.assertFalse(out["movement_constraints_applied"])
        with self.assertRaises(ValidationError):   # a lane the trace does not have
            validate_controls({"controls": [control(lane_candidates=["r3"])]}, ["r1", "r2"], {"gsv_a"})
        with self.assertRaises(ValidationError):   # lanes named for a sign that applies to the whole approach
            validate_controls({"controls": [control(applies_to="approach")]}, ["r1", "r2"], {"gsv_a"})
        with self.assertRaises(ValidationError):   # evidence in an image that was not sent
            validate_controls({"controls": [control(visual_refs=[{"source_id": "other", "bbox_xyxy": [0, 0, 1, 1]}])]}, ["r1"], {"gsv_a"})

    def test_resampling(self):
        self.assertTrue(needs_resampling({"controls": []}))   # nothing found is not a verified absence
        self.assertFalse(needs_resampling({"controls": [control()]}))
        self.assertTrue(needs_resampling({"controls": [control(readability="partial")]}))

    def test_detail_pose_aims_at_the_box(self):
        view = {"image_size": [640, 640], "hfov_deg": 100, "pitch_deg": 0, "compass_heading_deg": 90}
        pose = detail_pose(view, [.45, .45, .55, .55])
        self.assertAlmostEqual(pose["compass_heading_deg"], 90, places=3)
        self.assertAlmostEqual(pose["pitch_deg"], 0, places=3)
        self.assertGreater(detail_pose(view, [.8, .45, .9, .55])["compass_heading_deg"], 90)   # right of centre turns right
        self.assertEqual(detail_pose(view, [.49, .49, .51, .51])["hfov_deg"], 15.0)   # a tiny box gets the narrowest view
        self.assertEqual(detail_pose(view, [0, 0, 1, 1])["hfov_deg"], 55.0)          # and a large one the widest

    def test_scene_turns_travel_up(self):
        self.assertAlmostEqual(run_signals.up_angle(0, -1), 0)
        self.assertAlmostEqual(run_signals.up_angle(1, 0), 90)
        self.assertAlmostEqual(run_signals.up_angle(-1, 0), -90)
        self.assertAlmostEqual(abs(run_signals.up_angle(0, 1)), 180)

    def test_seed_cache_is_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed, cache = Path(tmp) / "seed", Path(tmp) / "cache"
            params = {"pano": "abc"}
            write_json(seed / "metadata" / (digest(params) + ".json"), {"status": "OK", "pano_id": "abc"})
            client = StreetViewClient(ROOT, cache, policy_with({}), cache_only=True, seed=[seed])
            self.assertEqual(client.metadata({"pano_id": "abc"})["pano_id"], "abc")
            self.assertTrue((cache / "metadata" / (digest(params) + ".json")).exists())

    def test_profile_of_the_example(self):
        site = json.loads((EXAMPLE / "imagery/site.json").read_text())
        lanes = json.loads((EXAMPLE / "run/lanes.json").read_text())
        profile = run_signals.build_profile(site, lanes, {}, EXAMPLE / "run")
        approaches = {a["label"]: a for a in profile["approaches"]}
        self.assertEqual(sorted(approaches), ["EB", "NB", "WB"])
        self.assertEqual(approaches["EB"]["region_ids"], ["r1", "r2"])
        self.assertEqual(approaches["WB"]["section_id"], "in_WB")
        self.assertFalse(Path(profile["geometry_provenance"]["node_csv"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
