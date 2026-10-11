import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movement_fixer.fusion.controls import detail_pose,validate_controls,validate_inventory
from movement_fixer.hybrid.common import ValidationError


class ControlTests(unittest.TestCase):
    def setUp(self):self.view={'image_size':[640,640],'hfov_deg':100,'pitch_deg':12,'compass_heading_deg':350}
    def test_center_retains_pose(self):
        p=detail_pose(self.view,[.45,.45,.55,.55]);self.assertAlmostEqual(p['compass_heading_deg'],350);self.assertAlmostEqual(p['pitch_deg'],12)
    def test_right_target_wraps_heading(self):self.assertLess(detail_pose(self.view,[.8,.4,.9,.6])['compass_heading_deg'],90)
    def test_top_target_raises_pitch(self):self.assertGreater(detail_pose(self.view,[.4,.1,.6,.2])['pitch_deg'],12)
    def test_detail_fov_preserves_tall_object_and_is_bounded(self):
        self.assertGreater(detail_pose(self.view,[.48,.2,.52,.8])['hfov_deg'],15)
        self.assertEqual(detail_pose(self.view,[.49,.49,.51,.51])['hfov_deg'],15)
    def test_inventory_cannot_request_unknown_source(self):
        with self.assertRaises(ValidationError):validate_inventory({'observations':[],'detail_requests':[{'source_id':'missing','bbox_xyxy':[0,0,1,1],'question':'read'}]},{'known'})
    def control(self):return {'id':'c1','kind':'turn_restriction_sign','confidence':'high','readability':'readable','face_status':'front','binding_status':'unresolved','applies_to':'unknown','lane_candidates':[],'observed_text':'','symbol':'unknown','description':'','binding_reason':'','visual_refs':[{'source_id':'s','bbox_xyxy':[0,0,1,1]}]}
    def test_unknown_binding_cannot_assert_lane(self):
        c=self.control();c['lane_candidates']=['r1']
        with self.assertRaises(ValidationError):validate_controls({'controls':[c]},['r1'],{'s'})
    def test_static_light_never_promotes_phase_or_constraint(self):
        result=validate_controls({'controls':[self.control()],'movement_constraints_applied':True,'signal_phase_plan':'invented'},['r1'],{'s'})
        self.assertFalse(result['movement_constraints_applied']);self.assertEqual(result['signal_phase_plan'],'not_observable_from_static_imagery')
    def test_control_requires_visual_support(self):
        c=self.control();c['visual_refs']=[]
        with self.assertRaises(ValidationError):validate_controls({'controls':[c]},['r1'],{'s'})


if __name__=='__main__':unittest.main()
