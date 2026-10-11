"""One entry point: network/profile -> cameras -> adaptive control audit -> optional atlas."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--config',type=Path);g.add_argument('--node-id')
    p.add_argument('--link-csv',type=Path);p.add_argument('--node-csv',type=Path);p.add_argument('--policy',type=Path);p.add_argument('--stoplines',type=Path);p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,default=ROOT/'cache/gsv_controls_shared');p.add_argument('--atlas-base',type=Path)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--cache-only',action='store_true');a=p.parse_args()
    for name in ('output','cache','config','link_csv','node_csv','policy','stoplines','baseline','atlas_base'):
        if getattr(a,name,None) is not None:setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    def run(script,args):subprocess.run([sys.executable,str(ROOT/'scripts/pipeline'/script),*map(str,args)],check=True,cwd=ROOT)
    config=a.config
    if a.node_id:
        config=a.output/'site.json';args=['--node-id',a.node_id,'--output',config]
        for name in ['link_csv','node_csv','policy','stoplines','baseline']:
            if getattr(a,name):args+=['--'+name.replace('_','-'),getattr(a,name)]
        run('build_control_profile.py',args)
    imagery=a.output/'imagery';audit=a.output/'audit';args=['--config',config,'--output',imagery,'--cache',a.cache]
    if a.cache_only:args+=['--cache-only']
    if a.plan_only:args+=['--plan-only']
    # Keep the original acquisition receipt stable during replay; cache-only audit verifies every used image.
    if not (a.cache_only and (imagery/'manifest.json').exists()):run('fetch_control_gsv.py',args)
    else:
        import json,hashlib
        sys.path.insert(0,str(ROOT/'src'))
        from movement_fixer.fusion.control_sampling import prepare_profile
        from movement_fixer.hybrid.common import read_json,digest
        if digest(prepare_profile(read_json(config)))!=read_json(imagery/'input_manifest.json')['signature']:raise ValueError('Replay config changed')
    if a.plan_only:return
    args=['--views',imagery/'manifest.json','--output',audit]
    if a.cache_only:args+=['--cache-only']
    run('audit_traffic_controls.py',args)
    if a.atlas_base:run('attach_control_evidence.py',['--base',a.atlas_base,'--audit',audit,'--output',a.output/'atlas'])
    run('validate_control_run.py',['--run',a.output])
    print('Control report:',audit/'index.html')


if __name__=='__main__':main()
