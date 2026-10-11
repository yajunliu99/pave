import copy
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.hybrid.common import read_json,ValidationError
from movement_fixer.hybrid.geometry_checks import audit_geometry
from movement_fixer.hybrid.validation import validate_lanes,validate_movement_output,receiving_direction
from test_hybrid_pipeline import lanes,decisions


class LaneUseTests(unittest.TestCase):
    def setUp(self): self.cfg=read_json(ROOT/"configs/pilots/archive/university_mill_351.v5-lane-use-review.json")

    def make_sections(self,annotation=None,arrows=None):
        section={"id":"in_NB","kind":"inbound_stopbar","direction":"NB","source_id":"sat",
            "regions":[{"id":"r1","review_annotation":annotation} if annotation else {"id":"r1"}]}
        raw={"regions":[{"region_id":"r1","surface_type":"motor_vehicle_lane","observed_arrows":arrows or [],
            "visibility":"clear","basis":"observed" if arrows else "inferred","confidence":"high","evidence":"fixture"}]}
        return [validate_lanes(raw,section)]+[lanes("out_"+d,["motor_vehicle_lane"]) for d in ("NB","EB","SB","WB")]

    def proposed(self,turn):
        value=decisions();m=next(m for m in value["movements"] if m["turn"]==turn)
        m.update(status="candidate",lane_pairs=[{"in_region_id":"r1","out_region_id":"r1"}],
            evidence_refs=["in_NB:r1","out_"+receiving_direction("NB",turn)+":r1"],
            assumptions=["Geometry appears aligned, so assume shared use."])
        return value

    def test_removed_exit_regions_are_only_in_exclusion_audit(self):
        for sid,removed in (("out_EB",{"r4","r5"}),("out_WB",{"r4"})):
            s=next(s for s in self.cfg["sections"] if s["id"]==sid)
            self.assertEqual({r["id"] for r in s["regions"]},{"r1","r2","r3"})
            self.assertEqual({r["id"] for r in s["excluded_regions"]},removed)
            self.assertTrue(all(not r["endpoint_eligible"] for r in s["excluded_regions"]))

    def test_normal_width_does_not_make_outside_pavement_a_lane(self):
        s=next(s for s in self.cfg["sections"] if s["id"]=="out_EB")
        rejected=copy.deepcopy(next(r for r in s["excluded_regions"] if r["id"]=="r4"))
        rejected["width_exception_reason"]="Width looks plausible"
        s["regions"].append(rejected)
        check=audit_geometry(self.cfg)["sections"]["out_EB"]["regions"]["r4"]
        self.assertLess(check["relative_width"],1.2)
        self.assertIn("outside_reviewed_carriageway",check["flags"])
        self.assertFalse(check["single_motor_lane_width_supported"])

    def test_user_right_only_rejects_through_even_with_assumptions(self):
        ss=self.make_sections({"surface_type":"motor_vehicle_lane","intended_directions":["right"],"exclusive":True})
        with self.assertRaises(ValidationError): validate_movement_output(self.proposed("through"),"NB",ss)
        result=validate_movement_output(self.proposed("right"),"NB",ss)
        self.assertEqual(next(m for m in result if m["turn"]=="right")["status"],"candidate")
        self.assertEqual(ss[0]["regions"][0]["observed_arrows"],[])

    def test_observed_right_arrow_rejects_geometric_through(self):
        ss=self.make_sections(arrows=["right"])
        with self.assertRaises(ValidationError): validate_movement_output(self.proposed("through"),"NB",ss)

    def test_shared_pavement_arrow_set_permits_both_and_rejects_left(self):
        ss=self.make_sections(arrows=["through","right"])
        for turn in ("through","right"): validate_movement_output(self.proposed(turn),"NB",ss)
        with self.assertRaises(ValidationError): validate_movement_output(self.proposed("left"),"NB",ss)

    def test_unmarked_lane_can_still_have_geometric_through_candidate(self):
        validate_movement_output(self.proposed("through"),"NB",self.make_sections())

    def test_eb_and_wb_outer_lane_constraints_are_exclusive_right(self):
        for sid in ("in_EB","in_WB"):
            r=next(s for s in self.cfg["sections"] if s["id"]==sid)["regions"][-1]
            self.assertEqual(r["review_annotation"]["intended_directions"],["right"])
            self.assertTrue(r["review_annotation"]["exclusive"])

    def test_removed_region_id_cannot_be_referenced_in_movement(self):
        ss=self.make_sections();v=self.proposed("through")
        next(m for m in v["movements"] if m["turn"]=="through")["lane_pairs"][0]["out_region_id"]="r4"
        with self.assertRaises(ValidationError): validate_movement_output(v,"NB",ss)


if __name__=="__main__": unittest.main()
