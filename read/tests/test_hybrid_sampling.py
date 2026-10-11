import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.hybrid.common import read_json,ValidationError
from movement_fixer.hybrid.sampling import audit_sampling
from movement_fixer.hybrid.validation import validate_context,validate_movement_output
from movement_fixer.hybrid.prompts import movement_prompt
from test_hybrid_pipeline import lanes,decisions


class SamplingTests(unittest.TestCase):
    def setUp(self): self.cfg=read_json(ROOT/"configs/pilots/university_mill_351.json")

    def test_all_four_approaches_have_three_distinct_positions(self):
        audit=audit_sampling(self.cfg)
        self.assertEqual(len(audit["positions"]),12)
        self.assertEqual(audit["unique_panoramas"],12)
        self.assertTrue(all(x>10 for pairs in audit["position_separation_m"].values() for x in pairs.values()))
        self.assertEqual(len({v["source_id"] for v in self.cfg["context_views"]+self.cfg["gsv_views"]}),15)

    def test_missing_distance_role_is_rejected(self):
        self.cfg["context_views"].pop()
        with self.assertRaises(ValidationError): audit_sampling(self.cfg)

    def test_duplicate_panorama_cannot_fill_both_roles(self):
        a,b=self.cfg["context_views"][:2]
        self.cfg["sources"][b["source_id"]]["pano_id"]=self.cfg["sources"][a["source_id"]]["pano_id"]
        with self.assertRaises(ValidationError): audit_sampling(self.cfg)

    def test_actual_coordinates_override_distance_labels(self):
        source=self.cfg["sources"][self.cfg["context_views"][0]["source_id"]]
        source["actual_lat"]+=.001
        with self.assertRaises(ValidationError): audit_sampling(self.cfg)

    def test_context_keeps_distance_and_panorama_for_movement(self):
        view=self.cfg["context_views"][0]; source=self.cfg["sources"][view["source_id"]]
        result=validate_context({"observations":[],"notes":[]},view,source)
        self.assertEqual(result["sampling_position"],view["sampling_position"])
        self.assertEqual(result["pano_id"],source["pano_id"])
        self.assertEqual(result["distance_to_center_m"],source["distance_to_center_m"])

    def test_movement_preserves_comparison_and_all_three_evidence_refs(self):
        sections=[lanes("in_NB",["motor_vehicle_lane"])]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]
        for position in ("near","mid","far"):
            sections.append({"section_id":"gsv_context_NB_"+position,"observations":[{"id":"o1"}]})
        value=decisions(); value["gsv_alignment_notes"]=["Mid and near show different occlusions."]
        value["movements"][0].update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["gsv_context_NB_near:o1","gsv_context_NB_mid:o1","gsv_context_NB_far:o1"])
        result=validate_movement_output(value,"NB",sections)
        self.assertEqual(result[0]["gsv_alignment_notes"],value["gsv_alignment_notes"])
        prompt=movement_prompt("NB",sections,self.cfg["legs"],{},{} )
        self.assertIn("gsv_context_NB_near",prompt); self.assertIn("gsv_context_NB_mid",prompt)
        self.assertIn("gsv_context_NB_far",prompt)

    def test_far_cannot_reuse_mid_panorama(self):
        near,mid,far=self.cfg["context_views"][:3]
        self.cfg["sources"][far["source_id"]]["pano_id"]=self.cfg["sources"][mid["source_id"]]["pano_id"]
        with self.assertRaises(ValidationError): audit_sampling(self.cfg)

    def test_wb_historical_far_view_is_explicitly_flagged(self):
        audit=audit_sampling(self.cfg)
        self.assertTrue(audit["date_comparisons"]["WB"]["review_required"])
        self.assertEqual(audit["date_comparisons"]["WB"]["dates"]["far"],"2024-06")
        self.assertFalse(audit["date_comparisons"]["NB"]["review_required"])

    def test_movement_flags_mixed_dates_without_inventing_arrows(self):
        sections=[lanes("in_NB",["motor_vehicle_lane"])]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]
        for position,date in (("near","2025-11"),("mid","2025-11"),("far","2024-06")):
            sections.append({"section_id":"gsv_context_NB_"+position,"kind":"gsv_forward_context","direction":"NB",
                "sampling_position":position,"capture_date":date,"observations":[]})
        results=validate_movement_output(decisions(),"NB",sections)
        self.assertTrue(all("gsv_capture_date_mismatch" in r["validation_flags"] for r in results))

    def test_archived_two_band_configuration_still_validates(self):
        cfg=read_json(ROOT/"configs/pilots/archive/university_mill_351.v3-two-positions.json")
        audit=audit_sampling(cfg)
        self.assertEqual(audit["position_order"],["near","mid"])
        self.assertEqual(audit["unique_panoramas"],8)


if __name__=="__main__": unittest.main()
