"""Offline end-to-end checks use synthetic imagery and fake providers, not accuracy claims."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
from test_control_generalization import config
from movement_fixer.hybrid.common import read_json,write_json,file_hash,digest,ValidationError
from movement_fixer.hybrid.inference import StageClient
from movement_fixer.fusion.control_sampling import prepare_profile
from movement_fixer.fusion.control_acquisition import acquire_initial
spec=importlib.util.spec_from_file_location('generic_control_audit',ROOT/'scripts/pipeline/audit_traffic_controls.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SyntheticClient:
    def __init__(self,root,cache,policy,cache_only=False):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.policy=policy
        self.counts={'metadata_requests':0,'image_requests':0,'metadata_cache_hits':0,'image_cache_hits':0}
    def metadata(self,r):
        self.counts['metadata_requests']+=1
        return {'status':'OK','actual_lonlat':r['requested_lonlat'],'pano_id':digest(r['requested_lonlat'])[:12],'capture_date':'2025-11'}
    def image(self,v):
        self.counts['image_requests']+=1;p=self.cache/(digest([v['pano_id'],v['compass_heading_deg'],v['pitch_deg'],v['hfov_deg']])+'.jpg');Image.new('RGB',(16,16),'white').save(p)
        return {**v,'path':str(p),'image_size':[16,16],'image_sha256':file_hash(p)}


class SyntheticVLM:
    def __init__(self,*a,**k):self.calls=0;self.hits=0
    def run(self,name,image,prompt,validator):
        self.calls+=1;panels=read_json(Path(image).with_suffix('.panels.json'));source=panels[-1]['image_id']
        if name.startswith('inventory_'):return validator({'observations':['synthetic test'],'detail_requests':[{'source_id':source,'bbox_xyxy':[.4,.4,.6,.6],'question':'read'}]})
        improved=name.endswith('_r2')
        c={'id':'c1','kind':'regulatory_sign','observed_text':'SYNTHETIC READABLE' if improved else '', 'symbol':'unknown','description':'synthetic offline fixture',
           'face_status':'front','readability':'readable' if improved else 'unreadable','confidence':'medium','applies_to':'approach','lane_candidates':[],
           'binding_status':'unresolved','binding_reason':'no lane geometry in fixture','visual_refs':[{'source_id':source,'bbox_xyxy':[.4,.4,.6,.6],'finding':'synthetic'}]}
        return validator({'controls':[c],'unresolved':[] if improved else ['text unclear'],'notes':[],'resampling_requests':[]})


class PipelineTests(unittest.TestCase):
    def test_three_and_five_arm_adaptive_pipeline_without_satellite_or_compass_ids(self):
        for angles in [(20,140,270),(15,77,149,222,301)]:
            with self.subTest(arms=len(angles)),tempfile.TemporaryDirectory() as temp:
                t=Path(temp);c=config(angles);c['models']={'vlm':{},'yolo':{}};p=prepare_profile(c);client=SyntheticClient(ROOT,t/'cache',p['policy'])
                m=acquire_initial(p,client);write_json(t/'manifest.json',m)
                with patch.object(audit,'StreetViewClient',SyntheticClient),patch.object(audit,'StageClient',SyntheticVLM),patch.object(audit,'detect',lambda views,*args:{v['id']:{'detections':[]} for v in views}):
                    summary=audit.run_audit(t/'manifest.json',t/'audit')
                self.assertEqual(summary['approaches'],len(angles));self.assertEqual(summary['api_calls'],4*len(angles))
                self.assertTrue(all(v=='reading_sufficient' for v in summary['sampling_stop_reasons'].values()))
                self.assertTrue((t/'audit/index.html').exists());results=read_json(t/'audit/predictions.json')
                self.assertTrue(all(r['controls'][0]['observed_text']=='SYNTHETIC READABLE' and not r['movement_constraints_applied'] for r in results))
                self.assertTrue(all(len(h['rounds'])==2 for h in read_json(t/'audit/adaptive_history.json')))
    def test_vlm_http_budget_fails_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            t=Path(temp);(t/'.env').write_text('',encoding='utf-8');client=StageClient(t,t,{'max_api_calls':1});client.calls=1
            with patch('movement_fixer.hybrid.inference.CreateAIClient',side_effect=lambda **k:SimpleNamespace(session=k['session'])):
                with self.assertRaises(ValidationError):client._client().session.post('https://example.invalid')
            self.assertEqual(client.calls,1)


if __name__=='__main__':unittest.main()
