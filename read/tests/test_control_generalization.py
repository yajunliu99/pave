import copy
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.fusion.coordinates import local_lonlat
from movement_fixer.fusion.control_sampling import prepare_profile,initial_requests,request_at,assess_panorama,point_at,policy_with,safe_id
from movement_fixer.fusion.control_acquisition import StreetViewClient,AcquisitionUnavailable,acquire_initial,alternative_requests
from movement_fixer.fusion.controls import needs_resampling,reading_signature
from movement_fixer.hybrid.common import ValidationError


def config(angles=(0,90,180,270),side='right',width=10):
    origin=[-112.,33.]
    return {'schema':'control-site-1','site_id':'arbitrary-site','origin_lonlat':origin,'policy':{'driving_side':side},'approaches':[
        {'id':f'a{i}','centerline_lonlat':[list(local_lonlat(-120*math.sin(math.radians(a)),-120*math.cos(math.radians(a)),origin)),origin],
         'centerline_reference':'shared_road_axis','carriageway_width_m':width,'allow_interior':True} for i,a in enumerate(angles)]}


class GeometryTests(unittest.TestCase):
    def test_skew_angles_follow_geometry(self):
        angles=[13,107,203,288];p=prepare_profile(config(angles))
        for a,h in zip(p['approaches'],angles):self.assertAlmostEqual(a['heading_deg'],h,places=3)
    def test_three_and_five_arms_without_cardinal_names(self):
        for angles in [(27,125,278),(0,65,140,216,290)]:
            p=prepare_profile(config(angles));self.assertEqual(len(p['approaches']),len(angles));self.assertEqual(sum(len(initial_requests(p,a)) for a in p['approaches']),3*len(angles))
    def test_left_hand_traffic_mirrors_lateral_not_heading(self):
        r=prepare_profile(config(side='right'));l=prepare_profile(config(side='left'))
        a=initial_requests(r,r['approaches'][0])[0];b=initial_requests(l,l['approaches'][0])[0]
        self.assertAlmostEqual(a['requested_local_xy'][0],-b['requested_local_xy'][0]);self.assertAlmostEqual(a['nominal_heading_deg'],b['nominal_heading_deg'])
    def test_directed_carriageway_does_not_add_second_offset(self):
        c=config();c['approaches'][0]['centerline_reference']='directed_carriageway';p=prepare_profile(c)
        self.assertEqual(initial_requests(p,p['approaches'][0])[0]['requested_lateral_m'],0)
    def test_width_scales_sampling(self):
        n=prepare_profile(config(width=6));w=prepare_profile(config(width=18))
        self.assertLess(initial_requests(w,w['approaches'][0])[0]['requested_station_m'],initial_requests(n,n['approaches'][0])[0]['requested_station_m'])
    def test_t_junction_never_extends_camera_past_dead_end(self):
        c=config((0,90,270));c['approaches'][0]['allow_interior']=False;p=prepare_profile(c)
        self.assertTrue(all(r['requested_station_m']<0 for r in initial_requests(p,p['approaches'][0])))
    def test_stopline_geometry_overrides_envelope(self):
        c=config();o=c['origin_lonlat'];c['approaches'][0]['stopline']={'geometry_lonlat':[local_lonlat(-8,-24,o),local_lonlat(8,-24,o)],'provenance':'independent_annotation'}
        a=prepare_profile(c)['approaches'][0];self.assertAlmostEqual(a['reference_station_m'],-24,places=3);self.assertEqual(a['reference_kind'],'provided_stopline')
    def test_missing_stopline_is_labeled_estimate(self):self.assertEqual(prepare_profile(config())['approaches'][0]['reference_kind'],'estimated_junction_envelope')
    def test_stopline_requires_provenance(self):
        c=config();c['approaches'][0]['stopline']={'geometry_lonlat':[c['origin_lonlat']]}
        with self.assertRaises(ValidationError):prepare_profile(c)
    def test_curved_road_samples_polyline_not_straight_ray(self):
        c=config((0,));o=c['origin_lonlat'];c['approaches'][0]['centerline_lonlat']=[local_lonlat(x,y,o) for x,y in [(-50,-90),(-25,-40),(0,0)]];p=prepare_profile(c)
        r=request_at(p,p['approaches'][0],-70,'test');self.assertLess(r['requested_local_xy'][0],-15);self.assertGreater(r['nominal_heading_deg'],20)
    def test_short_geometry_clamps_requests(self):
        c=config((0,));c['approaches'][0]['centerline_lonlat'][0]=local_lonlat(0,-6,c['origin_lonlat']);p=prepare_profile(c)
        self.assertTrue(all(r['requested_station_m']>=-6 for r in initial_requests(p,p['approaches'][0])))
    def test_reversed_incoming_geometry_fails_closed(self):
        c=config();c['approaches'][0]['centerline_lonlat'].reverse()
        with self.assertRaises(ValidationError):prepare_profile(c)
    def test_nonfinite_coordinates_rejected(self):
        c=config();c['approaches'][0]['centerline_lonlat'][0][0]=float('nan')
        with self.assertRaises(ValidationError):prepare_profile(c)
    def test_path_ids_cannot_escape_workspace(self):self.assertNotIn('/',safe_id('../../evil'));self.assertNotEqual(safe_id('a/b'),safe_id('a?b'))
    def test_custom_origin(self):
        c=config((42,));o=[130.,-20.];c['origin_lonlat']=o;c['approaches'][0]['centerline_lonlat']=[local_lonlat(-60,-60,o),o]
        self.assertAlmostEqual(prepare_profile(c)['approaches'][0]['heading_deg'],45,places=3)
    def test_policy_rounds_and_fov_bounded(self):
        for override in ({'max_rounds':100},{'max_rounds':2.5},{'max_context_views_per_approach':2},{'max_image_requests':1.5},{'detail_min_fov_deg':60,'detail_max_fov_deg':15},{'image_size':[9999,640]}):
            with self.assertRaises(ValidationError):policy_with(override)


class PoseTests(unittest.TestCase):
    def setUp(self):self.p=prepare_profile(config());self.a=self.p['approaches'][0];self.r=request_at(self.p,self.a,-40,'test')
    def meta(self,x,y,date='2025-11'):return {'status':'OK','actual_lonlat':local_lonlat(x,y,self.p['origin_lonlat']),'capture_date':date,'pano_id':'p'}
    def test_wrong_carriageway_is_rejected(self):self.assertIn('opposing_carriageway_snap',assess_panorama(self.p,self.a,self.r,self.meta(-5,-40))['reasons'])
    def test_large_snap_is_rejected(self):self.assertFalse(assess_panorama(self.p,self.a,self.r,self.meta(50,-40))['usable'])
    def test_actual_location_not_request_is_used(self):self.assertAlmostEqual(assess_panorama(self.p,self.a,self.r,self.meta(5,-43))['actual_station_m'],-43,places=3)
    def test_dates_are_not_silently_current(self):
        self.p['policy']['preferred_date_window']=['2025-10','2025-11'];r=assess_panorama(self.p,self.a,self.r,self.meta(5,-40,'2024-06'))
        self.assertEqual(r['capture_relation'],'historical');self.assertTrue(r['usable'])
    def test_metadata_failure_preserves_status(self):self.assertEqual(assess_panorama(self.p,self.a,self.r,{'status':'ZERO_RESULTS'})['reasons'],['ZERO_RESULTS'])
    def test_adaptive_requests_different_from_existing(self):
        views=[{'view_role':'control_context','signed_station_m':-20}];r=alternative_requests(self.p,self.a,views,2)
        self.assertTrue(r);self.assertTrue(all(abs(v['requested_station_m']+20)>=3 for v in r))


class FakeResponse:
    status_code=200
    def __init__(self,body=None):
        self.body=body;buff=io.BytesIO();Image.new('RGB',(8,8)).save(buff,format='JPEG');self.content=buff.getvalue()
    def json(self):return self.body

class FakeSession:
    def __init__(self):self.calls=[]
    def get(self,url,params,timeout):
        self.calls.append((url,params))
        if url.endswith('metadata'):
            lat,lon=map(float,params['location'].split(','));return FakeResponse({'status':'OK','location':{'lng':lon,'lat':lat},'pano_id':'p','date':'2025-11'})
        return FakeResponse()


class CacheTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.s=FakeSession();self.p=prepare_profile(config());self.c=StreetViewClient(ROOT,self.tmp.name,self.p['policy'],session=self.s);self.c.key='test-only';self.r=initial_requests(self.p,self.p['approaches'][0])[0]
    def tearDown(self):self.tmp.cleanup()
    def test_metadata_cache_replay_makes_no_request(self):
        a=self.c.metadata(self.r);c=StreetViewClient(ROOT,self.tmp.name,self.p['policy'],cache_only=True,session=self.s);self.assertEqual(c.metadata(self.r),a);self.assertEqual(len(self.s.calls),1)
    def test_cache_miss_does_not_use_network(self):
        self.c.cache_only=True
        with self.assertRaises(AcquisitionUnavailable):self.c.metadata(self.r)
        self.assertEqual(self.s.calls,[])
    def test_metadata_budget_is_hard_limit(self):
        self.c.policy={**self.c.policy,'max_metadata_requests':1};self.c.metadata(self.r);r=copy.deepcopy(self.r);r['requested_lonlat'][0]+=.001
        with self.assertRaises(AcquisitionUnavailable):self.c.metadata(r)
        self.assertEqual(len(self.s.calls),1)
    def test_image_cache_key_includes_angles(self):
        v={'pano_id':'p','compass_heading_deg':0,'pitch_deg':10,'hfov_deg':80};a=self.c.image(v);b=self.c.image({**v,'compass_heading_deg':25});self.assertNotEqual(a['path'],b['path']);self.c.image(v);self.assertEqual(len(self.s.calls),2)
    def test_cache_record_never_contains_api_key(self):
        self.c.metadata(self.r)
        self.assertTrue(all('test-only' not in p.read_text(encoding='utf-8') for p in Path(self.tmp.name).rglob('*.json')))
    def test_duplicate_panos_dont_count_as_distinct_positions(self):
        m=acquire_initial(self.p,self.c);self.assertTrue(any(r['status']=='duplicate_panorama' for r in m['positions']));self.assertTrue(all(c['views']==1 for c in m['coverage']))
    def test_target_period_alternative_preferred_to_nearest_old_panorama(self):
        self.p['policy']['preferred_date_window']=['2025-10','2025-11'];original=self.s.get;seen=[]
        def dated_get(url,params,timeout):
            response=original(url,params,timeout)
            if url.endswith('metadata'):
                if not seen:response.body['date']='2013-04'
                seen.append(1)
            return response
        self.s.get=dated_get;m=acquire_initial(self.p,self.c)
        self.assertEqual(m['views'][0]['capture_date'],'2025-11')
        self.assertTrue(any(r['status']=='outside_preferred_date_window' for r in m['positions']))
    def test_known_panorama_fills_the_preferred_window(self):
        """A location search that only returns later captures is completed by a panorama named in the policy."""
        a=self.p['approaches'][0];r=self.r;lon,lat=r['requested_lonlat']
        self.p['policy'].update(preferred_date_window=['2020-01','2020-12'],date_search={'along_m':3,'across_m':3,'step_m':3,'radius_m':3},
                                known_panoramas={a['section_id']:['hidden2020']})
        original=self.s.get
        def get(url,params,timeout):
            if url.endswith('metadata') and 'pano' in params:
                self.s.calls.append((url,params))
                return FakeResponse({'status':'OK','location':{'lng':lon,'lat':lat},'pano_id':params['pano'],'date':'2020-11','copyright':'© Google'})
            return original(url,params,timeout)
        self.s.get=get;m=acquire_initial(self.p,self.c)
        first=[v for v in m['views'] if v['approach_id']==a['id']][0]
        self.assertEqual((first['pano_id'],first['capture_date'],first['capture_relation']),('hidden2020','2020-11','target_window'))
    def test_without_date_fallback_a_position_outside_the_window_stays_empty(self):
        self.p['policy'].update(preferred_date_window=['2020-01','2020-12'],date_search={'along_m':3,'across_m':3,'step_m':3,'radius_m':3},date_fallback=False)
        m=acquire_initial(self.p,self.c)
        self.assertEqual(m['views'],[])
        self.assertTrue(all(r['status']!='accepted' for r in m['positions']))
        self.p['policy']['date_fallback']=True
        self.assertTrue(acquire_initial(self.p,self.c)['views'])
    def test_known_panoramas_need_a_window_and_a_search(self):
        with self.assertRaises(ValidationError):policy_with({'known_panoramas':{'in_a0':['x']}})
        with self.assertRaises(ValidationError):policy_with({'known_panoramas':{'in_a0':'x'}})


class FeedbackTests(unittest.TestCase):
    def test_empty_inventory_does_not_establish_absence(self):self.assertTrue(needs_resampling({'controls':[]}))
    def control(self):return {'kind':'turn_restriction_sign','applies_to':'approach','readability':'readable','face_status':'front','symbol':'no U-turn','observed_text':'','binding_status':'unresolved'}
    def test_lane_binding_uncertainty_alone_does_not_trigger(self):self.assertFalse(needs_resampling({'controls':[self.control()]}))
    def test_unreadable_relevant_control_triggers(self):
        c=self.control();c['readability']='unreadable';self.assertTrue(needs_resampling({'controls':[c]}))
    def test_other_approach_ignored(self):
        c=self.control();c.update(readability='unreadable',applies_to='other_approach');self.assertFalse(needs_resampling({'controls':[c]}))
    def test_unknown_reading_is_not_progress(self):
        c=self.control();c.update(symbol='unknown',observed_text='[?]');self.assertFalse(reading_signature({'controls':[c]}))


if __name__=='__main__':unittest.main()
