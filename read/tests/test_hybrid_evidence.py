import copy
import tempfile
import unittest
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image
from movement_fixer.hybrid.common import read_json,ValidationError
from movement_fixer.hybrid.evidence_config import build_evidence_config,assert_no_answer_fields
from movement_fixer.hybrid.evidence_reasoning import validate_surface,validate_usage,validate_refs,evidence_movement_prompt,fuse_surface_usage
from movement_fixer.hybrid.evidence_visuals import scene_views,detail_sheet
from movement_fixer.hybrid.validation import validate_movement_output
from test_hybrid_pipeline import lanes,decisions


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.old=read_json(ROOT/"configs/pilots/archive/university_mill_351.v4-three-positions.json")
        self.cfg=build_evidence_config(self.old)

    def test_default_restores_all_candidate_regions_without_answers(self):
        assert_no_answer_fields(self.cfg)
        for sid,count in (("out_EB",5),("out_WB",4)):
            self.assertEqual(len(next(s for s in self.cfg["sections"] if s["id"]==sid)["regions"]),count)
        self.assertEqual(self.cfg["inference_mode"],"visual_evidence")

    def test_review_canary_cannot_pass_allowlist(self):
        original=self.old["sections"][0]
        original["notes"]="CANARY_REFERENCE_847"
        original["regions"][0]["review_annotation"]={"intended_directions":["CANARY_REFERENCE_847"]}
        original["regions"][0]["quality"]="CANARY_REFERENCE_847"
        self.assertNotIn("CANARY_REFERENCE_847",str(build_evidence_config(self.old)))

    def test_runtime_rejects_injected_reference_keys(self):
        self.cfg["sections"][0]["regions"][0]["review_annotation"]={}
        with self.assertRaises(ValidationError):assert_no_answer_fields(self.cfg)

    def test_visual_exclusion_comes_from_model_output_not_region_id(self):
        s={"id":"out_EB","kind":"outbound_receiving","direction":"EB","source_id":"satellite","regions":[{"id":"r1"},{"id":"r4"}]}
        raw={"regions":[{"region_id":r,"surface_type":"motor_vehicle_lane","existence":"supported","observed_arrows":[],"confidence":"medium","evidence":"visible pavement","visual_refs":[]} for r in ("r1","r4")]}
        result=validate_surface(raw,s,{"scene_out_EB"})
        self.assertEqual(result["count"]["model_motor_count"],2)
        self.assertTrue(result["regions"][1]["endpoint_eligible"])
        raw["regions"][0].update(existence="rejected",surface_type="non_road")
        result=validate_surface(raw,s,{"scene_out_EB"})
        self.assertFalse(result["regions"][0]["endpoint_eligible"])
        self.assertTrue(result["regions"][1]["endpoint_eligible"])

    def test_visual_refs_require_valid_image_and_coordinates(self):
        with self.assertRaises(ValidationError):validate_refs([{"source_id":"foreign","bbox_xyxy":[0,0,1,1]}],{"gsv"})
        with self.assertRaises(ValidationError):validate_refs([{"source_id":"gsv","bbox_xyxy":[0,0,1.2,1]}],{"gsv"})

    def test_usage_output_is_not_forced_to_right_only(self):
        s={"regions":[{"region_id":"r5"}]}
        raw={"regions":[{"region_id":"r5","surface_type":"motor_vehicle_lane","existence":"supported","surface_reason":"joint image evidence","use":"through_only","allowed_turns":["through"],"binding":"paint evidence",
            "visual_refs":[{"source_id":"gsv","bbox_xyxy":[.1,.2,.3,.4]}]}],"detail_requests":[]}
        self.assertEqual(validate_usage(raw,s,{"gsv"})["regions"][0]["use"],"through_only")

    def test_joint_model_evidence_can_revise_initial_satellite_class(self):
        section=lanes("in_NB",["shoulder_or_parking"])
        region=section["regions"][0];region["existence"]="rejected"
        usage={"regions":[{"region_id":"r1","surface_type":"motor_vehicle_lane","existence":"supported",
            "surface_reason":"A visible travel arrow and curb correspondence support roadway.","use":"through_only","allowed_turns":["through"]}]}
        fuse_surface_usage(section,usage)
        self.assertTrue(region["endpoint_eligible"])
        self.assertEqual(region["initial_surface_audit"]["surface_type"],"shoulder_or_parking")
        self.assertEqual(region["lane_use_prediction"]["use"],"through_only")

    def test_evidence_movement_conflict_is_flagged_not_overwritten(self):
        ss=[lanes("in_NB",["motor_vehicle_lane"])]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]
        ss[0]["regions"][0]["lane_use_prediction"]={"use":"right_only","allowed_turns":["right"]}
        v=decisions();v["movements"][1].update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["in_NB:r1","out_NB:r1"])
        result=validate_movement_output(v,"NB",ss,enforce_turn_constraints=False)
        self.assertEqual(result[1]["status"],"candidate")
        self.assertIn("model_lane_use_disagreement:r1:through",result[1]["validation_flags"])

    def test_scene_preserves_external_context_pixels(self):
        image=Image.new("RGB",(600,600),"green")
        s={"id":"in_NB","regions":[{"id":"r1","polygon":[[250,250],[300,250],[300,350],[250,350]],"gate_point":[275,250]}]}
        with tempfile.TemporaryDirectory() as tmp:
            raw,ann,meta=scene_views(image,s,tmp)
            self.assertEqual(raw.getpixel((0,0)),(0,128,0))
            self.assertGreater(raw.width,50)
            self.assertEqual(list(meta["source_crop_xyxy"]),[90,90,460,510])

    def test_native_detail_crop_is_selected_from_model_coordinates(self):
        image=Image.new("RGB",(100,100),"white")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"details.png"
            detail_sheet([{"source_id":"gsv","bbox_xyxy":[.2,.3,.7,.9]}],{"gsv":image},image,p)
            mapping=read_json(p.with_suffix(".crops.json"))
            self.assertEqual(mapping[0]["source_bbox_xyxy"],[20,30,70,90])


if __name__=="__main__":unittest.main()
