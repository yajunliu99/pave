import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from movement_fixer.fusion.downstream import camera_context,acquisition_summary


class DownstreamTests(unittest.TestCase):
    def test_same_side_does_not_receive_opposite_camera_instruction(self):
        for role in ("away","toward"):
            c=camera_context({"carriageway_status":"nominal_outgoing_side","view_role":role})
            self.assertEqual(c["camera_role"],"outbound_primary_"+role)
            self.assertIn("do not assume the target is across",c["target_relation"])

    def test_opposite_and_unknown_stay_distinct(self):
        opposite=camera_context({"carriageway_status":"opposite_carriageway_auxiliary","view_role":"away"})
        unknown=camera_context({"carriageway_status":"opposite_or_unresolved","view_role":"away"})
        self.assertIn("opposing carriageway",opposite["target_relation"])
        self.assertIn("unresolved",unknown["camera_role"])

    def test_successes_and_pano_groups_not_view_counts(self):
        result=acquisition_summary({"acquisition_mode":"official_location_query","metadata_attempts":[{"status":"OK"},{"status":"ZERO_RESULTS"}],
            "views":[{"pano_id":"a","carriageway_status":"nominal_outgoing_side"},{"pano_id":"a","carriageway_status":"nominal_outgoing_side"},{"pano_id":"b","carriageway_status":"opposite_carriageway_auxiliary"}]})
        self.assertEqual(result["new_panorama_queries_succeeded"],1)
        self.assertEqual(result["unique_panoramas"],2)
        self.assertEqual(result["nominal_outgoing_views"],2)
        self.assertEqual(result["opposing_auxiliary_views"],1)


if __name__=="__main__":unittest.main()
