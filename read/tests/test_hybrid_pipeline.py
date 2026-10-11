import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image
from movement_fixer.hybrid.common import ValidationError, load_config, write_json, read_json
from movement_fixer.hybrid.geometry import polygon_mask, bbox_union_coverage, rotate, unrotate_point
from movement_fixer.hybrid.validation import receiving_direction, validate_lanes, validate_movement_output
from movement_fixer.hybrid.pipeline import network_context, write_csv, preserve_review_fields


def lanes(sid,types):
    section={"id":sid,"kind":"inbound_stopbar" if sid.startswith("in_") else "outbound_receiving",
             "direction":sid.split("_")[-1],"source_id":"sat","coverage":"candidate_full",
             "regions":[{"id":f"r{i+1}"} for i in range(len(types))]}
    value={"regions":[{"region_id":f"r{i+1}","surface_type":t,"observed_arrows":[],
                        "visibility":"not_visible","basis":"inferred","confidence":"medium","evidence":"fixture"} for i,t in enumerate(types)]}
    return validate_lanes(value,section)


def decisions():
    return {"movements":[{"turn":t,"to_direction":receiving_direction("NB",t),"status":"unresolved",
                          "lane_pairs":[],"basis":"inferred","confidence":"low","evidence_refs":[],"assumptions":[],"reason":"not enough evidence"}
                         for t in ("left","through","right","u_turn")]}


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.sections=[lanes("in_NB",["motor_vehicle_lane","bicycle_strip","motor_vehicle_lane"])]
        self.sections += [lanes("out_"+d,["motor_vehicle_lane","motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]

    def test_turn_targets(self):
        expected={"NB":("WB","NB","EB","SB"),"EB":("NB","EB","SB","WB"),
                  "SB":("EB","SB","WB","NB"),"WB":("SB","WB","NB","EB")}
        for d,outputs in expected.items():
            self.assertEqual(tuple(receiving_direction(d,t) for t in ("left","through","right","u_turn")),outputs)

    def test_lane_number_skips_bike(self):
        self.assertEqual([r["motor_lane_index"] for r in self.sections[0]["regions"]],[1,None,2])

    def test_unknown_count_range(self):
        value=lanes("out_NB",["motor_vehicle_lane","unknown","bicycle_strip"])
        self.assertEqual(value["count"]["model_count_range"],[1,2])
        self.assertFalse(value["count"]["human_verified"])

    def candidate(self):
        data=decisions(); data["movements"][0].update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],evidence_refs=["in_NB:r1","out_WB:r1"])
        return data

    def test_candidate_indices(self):
        value=validate_movement_output(self.candidate(),"NB",self.sections)
        self.assertEqual(value[0]["lane_pairs"][0]["ib_lane"],1)
        self.assertFalse(value[0]["auto_apply"])

    def test_reject_bicycle_movement(self):
        data=self.candidate(); data["movements"][0]["lane_pairs"][0]["in_region_id"]="r2"
        with self.assertRaises(ValidationError): validate_movement_output(data,"NB",self.sections)

    def test_reject_wrong_exit_direction(self):
        data=self.candidate(); data["movements"][0]["to_direction"]="EB"
        with self.assertRaises(ValidationError): validate_movement_output(data,"NB",self.sections)

    def test_reject_foreign_evidence(self):
        data=self.candidate(); data["movements"][0]["evidence_refs"]=["out_WB:r99"]
        with self.assertRaises(ValidationError): validate_movement_output(data,"NB",self.sections)

    def test_reject_unresolved_pair(self):
        data=self.candidate(); data["movements"][0]["status"]="unresolved"
        with self.assertRaises(ValidationError): validate_movement_output(data,"NB",self.sections)

    def test_crossing_and_merge_flags(self):
        data=self.candidate(); data["movements"][0]["lane_pairs"]=[{"in_region_id":"r1","out_region_id":"r2"},{"in_region_id":"r3","out_region_id":"r1"}]
        self.assertIn("crossing_lane_order",validate_movement_output(data,"NB",self.sections)[0]["validation_flags"])
        data["movements"][0]["lane_pairs"][0]["out_region_id"]="r1"
        self.assertIn("many_to_one_merge_requires_review",validate_movement_output(data,"NB",self.sections)[0]["validation_flags"])

    def test_missing_turn_rejected(self):
        data=decisions(); data["movements"].pop()
        with self.assertRaises(ValidationError): validate_movement_output(data,"NB",self.sections)

    def test_low_confidence_and_geometry_flags(self):
        self.sections[4]["regions"][0]["confidence"]="low"
        value=validate_movement_output(self.candidate(),"NB",self.sections,{"out_WB":{"r1":"provisional"}})
        self.assertIn("low_confidence_endpoint",value[0]["validation_flags"])
        self.assertIn("provisional_endpoint_geometry",value[0]["validation_flags"])

    def test_preserve_user_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"review.csv"
            write_csv(path,[{"candidate_id":"NBL","reason":"old","approved":"yes","reviewer":"tester"}],
                      ["candidate_id","reason","approved","reviewer"])
            rows=preserve_review_fields(path,[{"candidate_id":"NBL","reason":"new","approved":"","reviewer":""}],
                                       ["candidate_id"],["approved","reviewer"])
            self.assertEqual(rows[0]["reason"],"new")
            self.assertEqual(rows[0]["approved"],"yes")
            self.assertEqual(rows[0]["reviewer"],"tester")

    def test_json_unchanged_skips_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"result.json"; write_json(path,{"value":1})
            with mock.patch.object(Path,"replace",side_effect=AssertionError("unnecessary write")):
                write_json(path,{"value":1})

    def test_json_replace_retries_transient_lock(self):
        original=Path.replace
        calls=[]
        def replacement(path,target):
            calls.append(1)
            if len(calls)<3: raise PermissionError("temporary lock")
            return original(path,target)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"result.json"
            with mock.patch.object(Path,"replace",autospec=True,side_effect=replacement), mock.patch("movement_fixer.hybrid.common.time.sleep"):
                write_json(path,{"value":2})
            self.assertEqual(read_json(path),{"value":2})
            self.assertEqual(len(calls),3)

    def test_rectangle_union_not_double_counted(self):
        mask=polygon_mask((10,10),[[0,0],[9,0],[9,9],[0,9]])
        det={"bbox_xyxy":[0,0,4,9]}
        self.assertEqual(bbox_union_coverage(mask,[det,det]),.5)
        self.assertEqual(bbox_union_coverage(mask,[]),0)

    def test_degenerate_polygon_rejected(self):
        with self.assertRaises(ValidationError): polygon_mask((10,10),[[0,0],[1,1],[2,2]])

    def test_inverse_rotations(self):
        im=Image.new("I",(7,5)); im.putdata(range(35))
        for angle in (0,90,180,270):
            out=rotate(im,angle)
            for x in range(out.width):
                for y in range(out.height):
                    a,b=unrotate_point(x,y,7,5,angle)
                    self.assertEqual(out.getpixel((x,y)),b*7+a)

    @unittest.skipUnless((ROOT/"cache/mapbox_351").exists(), "needs the node 351 satellite crop (imagery is not distributed)")
    def test_pilot_config_and_network(self):
        cfg=load_config(ROOT/"configs/pilots/university_mill_351.json",ROOT)
        self.assertEqual(len(cfg["sections"]),8)
        context=network_context(cfg,ROOT)
        self.assertEqual(context["NB"]["outbound"]["link_id"],144)


if __name__=="__main__": unittest.main()
