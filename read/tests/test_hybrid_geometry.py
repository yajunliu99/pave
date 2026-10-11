import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image
from movement_fixer.hybrid.common import ValidationError,read_json,write_json
from movement_fixer.hybrid.geometry_checks import audit_geometry,width_samples,through_alignment,uncovered_width
from movement_fixer.hybrid.validation import validate_lanes,validate_movement_output,validate_context
from movement_fixer.hybrid.inference import StageClient
from test_hybrid_pipeline import lanes,decisions


class GeometryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/"configs/pilots/archive/university_mill_351.v5-lane-use-review.json")

    def test_original_double_width_and_missing_left_detected(self):
        old=read_json(ROOT/"configs/pilots/archive/university_mill_351.v1.json")
        old["geometry_checks"]=self.config["geometry_checks"]
        envelope=next(s for s in self.config["sections"] if s["id"]=="in_EB")["carriageway_envelope"]
        next(s for s in old["sections"] if s["id"]=="in_EB")["carriageway_envelope"]=envelope
        audit=audit_geometry(old)
        for rid in ("r1","r2"):
            self.assertIn("possible_merged_lanes",audit["sections"]["out_EB"]["regions"][rid]["flags"])
            self.assertGreater(audit["sections"]["out_EB"]["regions"][rid]["relative_width"],1.9)
        self.assertTrue(audit["sections"]["in_EB"]["envelope_coverage"]["missing_region_review"])

    def test_revised_geometry_regular_width_and_no_gap(self):
        audit=audit_geometry(self.config)
        for rid in ("r1","r2"):
            self.assertEqual(audit["sections"]["out_EB"]["regions"][rid]["flags"],[])
        self.assertFalse(audit["sections"]["in_EB"]["envelope_coverage"]["missing_region_review"])
        self.assertEqual(len(next(s for s in self.config["sections"] if s["id"]=="in_EB")["regions"]),5)

    def test_eb_straight_pairs_are_nearly_horizontal(self):
        checks=audit_geometry(self.config)["straight_pairs"]["EB"]["pairs"]
        for i,o in (("r2","r1"),("r3","r2")):
            pair=next(p for p in checks if p["in_region_id"]==i and p["out_region_id"]==o)
            self.assertLess(pair["lateral_in_lane_widths"],.15)
            self.assertFalse(pair["hard_violation"])
        wrong=next(p for p in checks if p["in_region_id"]=="r3" and p["out_region_id"]=="r1")
        self.assertTrue(wrong["hard_violation"])

    def test_rotated_strip_width_not_bbox_width(self):
        angle=math.radians(30); u=(math.cos(angle),math.sin(angle)); n=(-u[1],u[0])
        polygon=[[200+s*u[0]+t*n[0],200+s*u[1]+t*n[1]] for s,t in ((0,0),(600,0),(600,50),(0,50))]
        for width in width_samples(polygon,u): self.assertAlmostEqual(width,50,places=8)

    def test_frozen_reference_keeps_unchanged_sections_hints(self):
        first=audit_geometry(self.config)
        revised=copy.deepcopy(self.config)
        region=next(s for s in revised["sections"] if s["id"]=="in_SB")["regions"][1]  # a width reference
        cx=sum(x for x,_ in region["polygon"])/len(region["polygon"]); cy=sum(y for _,y in region["polygon"])/len(region["polygon"])
        region["polygon"]=[[cx+(x-cx)*1.4,cy+(y-cy)*1.4] for x,y in region["polygon"]]
        unfrozen=audit_geometry(revised)
        self.assertNotEqual(unfrozen["references"],first["references"])
        revised["geometry_checks"]["frozen_reference_width_px"]={sid:r["width_px"] for sid,r in first["references"].items()}
        frozen=audit_geometry(revised)
        sid=next(iter(first["references"]))
        self.assertEqual(frozen["references"][sid]["width_px"],first["references"][sid]["width_px"])
        self.assertTrue(frozen["references"][sid]["frozen"])
        self.assertEqual(frozen["sections"]["out_EB"]["regions"],first["sections"]["out_EB"]["regions"])

    def test_width_reference_not_applied_to_perspective_gsv(self):
        audit=audit_geometry(self.config)
        self.assertTrue(all(not key.startswith("gsv") for key in audit["sections"]))

    def test_uncovered_interval_union(self):
        self.assertEqual(uncovered_width((0,100),[(0,40),(30,60),(80,100)]),20)

    def test_large_straight_shift_rejected_but_turn_not_affected(self):
        ss=[lanes("in_NB",["motor_vehicle_lane"])]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]
        values=decisions(); through=values["movements"][1]
        through.update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["in_NB:r1","out_NB:r1"])
        audit={"straight_pairs":{"NB":{"pairs":[{"in_region_id":"r1","out_region_id":"r1",**through_alignment((0,500),(120,0),"NB",50)}]}}}
        with self.assertRaises(ValidationError): validate_movement_output(values,"NB",ss,geometry_audit=audit)
        values=decisions(); values["movements"][0].update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["in_NB:r1","out_WB:r1"])
        self.assertEqual(validate_movement_output(values,"NB",ss,geometry_audit=audit)[0]["status"],"candidate")

    def test_double_width_motor_region_is_not_endpoint(self):
        section={"id":"in_NB","kind":"inbound_stopbar","direction":"NB","source_id":"sat","regions":[{"id":"r1"}]}
        raw={"regions":[{"region_id":"r1","surface_type":"motor_vehicle_lane","observed_arrows":[],"visibility":"partial","basis":"inferred","confidence":"medium","evidence":"wide road"}]}
        normalized=validate_lanes(raw,section,{"regions":{"r1":{"single_motor_lane_width_supported":False,"flags":["possible_merged_lanes"]}}})
        self.assertFalse(normalized["regions"][0]["endpoint_eligible"])
        ss=[normalized]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]
        values=decisions(); values["movements"][0].update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["in_NB:r1","out_WB:r1"])
        with self.assertRaises(ValidationError): validate_movement_output(values,"NB",ss)

    def test_forward_primary_views_cover_four_directions(self):
        self.assertEqual({v["direction"] for v in self.config["context_views"]},{"NB","SB","EB","WB"})
        for v in self.config["context_views"]:
            source=self.config["sources"][v["source_id"]]
            self.assertFalse(source["reverse_view"])
            self.assertEqual(source["camera_role"],"forward_primary")
        self.assertEqual(sum(s.get("reverse_view",False) for s in self.config["sources"].values()),3)

    def test_user_review_does_not_become_observed_arrow(self):
        section={"id":"in_EB","kind":"inbound_stopbar","direction":"EB","source_id":"sat",
            "regions":[{"id":"r1","review_annotation":{"surface_type":"motor_vehicle_lane","intended_directions":["left"]}}]}
        raw={"regions":[{"region_id":"r1","surface_type":"motor_vehicle_lane","observed_arrows":[],"visibility":"unclear","basis":"inferred","confidence":"medium","evidence":"user review"}]}
        value=validate_lanes(raw,section)["regions"][0]
        self.assertEqual(value["observed_arrows"],[])
        self.assertEqual(value["reviewed_directions"],["left"])

    def test_cross_run_cache_reuse_and_changed_prompt_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/"input.png"; Image.new("RGB",(4,4)).save(image)
            old=StageClient(root,root/"old",{"preset":"default"})
            raw={"metadata":{"model_details":{"inference_provider":old.provider,"inference_model":old.model}},"response":{"answer":1}}
            with mock.patch.object(old,"_client",return_value=SimpleNamespace(query_vision=lambda *a,**k:raw)):
                self.assertEqual(old.run("test",image,"same",lambda x:x),{"answer":1})
            new=StageClient(root,root/"new",{"preset":"default"},cache_only=True,reuse_cache_from=[root/"old"])
            with mock.patch.object(new,"_client",side_effect=AssertionError("network access")):
                self.assertEqual(new.run("test",image,"same",lambda x:x),{"answer":1})
                with self.assertRaises(ValidationError): new.run("test",image,"changed",lambda x:x)
            self.assertEqual(new.calls,0)
            self.assertEqual(new.hits,1)

    def test_atomic_json_write_with_long_windows_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            filename="lane_gsv_context_NB_"+"a"*20+".raw.json"
            # A valid 235-character final path used to produce a >260-char temp path.
            padding=235-len(str(root))-len(filename)-2
            target=root/("d"*padding)/filename
            writes=[]
            original=Path.write_text
            def bounded_write(path,*args,**kwargs):
                self.assertLess(len(str(path)),260)
                writes.append(path)
                return original(path,*args,**kwargs)
            with mock.patch.object(Path,"write_text",new=bounded_write):
                write_json(target,{"value":1})
                write_json(target,{"value":2})
            self.assertEqual(read_json(target),{"value":2})
            self.assertTrue(all(not path.exists() for path in writes))


if __name__=="__main__": unittest.main()
