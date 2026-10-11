"""Fetch geometry-driven control views for a configured road-network junction."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.hybrid.common import read_json,write_json,file_hash,digest
from movement_fixer.fusion.control_sampling import prepare_profile,initial_requests
from movement_fixer.fusion.control_acquisition import StreetViewClient,acquire_initial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'cache/gsv_controls_shared');p.add_argument('--cache-only',action='store_true');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    profile=prepare_profile(read_json(a.config));signature=digest(profile);a.output.mkdir(parents=True,exist_ok=True);prior=a.output/'input_manifest.json'
    if prior.exists() and read_json(prior)['signature']!=signature:raise ValueError('Changed geometry or policy; use a new output directory')
    write_json(prior,{'config':str(a.config.resolve()),'config_sha256':file_hash(a.config),'signature':signature})
    write_json(a.output/'sampling_plan.json',{'profile':profile,'requests':[r for app in profile['approaches'] for r in initial_requests(profile,app)]})
    if a.plan_only:print({'status':'planned','approaches':len(profile['approaches'])});return
    client=StreetViewClient(ROOT,a.cache,profile['policy'],cache_only=a.cache_only);manifest=acquire_initial(profile,client);write_json(a.output/'manifest.json',manifest)
    print({'site_id':profile['site_id'],'views':len(manifest['views']),'coverage':manifest['coverage'],'counts':client.counts})


if __name__=='__main__':main()
