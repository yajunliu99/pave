"""Create a control-sampling profile from a network node, not compass slots."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.hybrid.common import read_json,write_json,resource_path,file_hash
from movement_fixer.fusion.control_sampling import from_network,prepare_profile,initial_requests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--node-csv',type=Path);p.add_argument('--link-csv',type=Path);p.add_argument('--node-id')
    p.add_argument('--baseline',type=Path,help='Optional lane images and aliases; never reads reviewed answers')
    p.add_argument('--policy',type=Path,help='JSON global policy overrides');p.add_argument('--stoplines',type=Path,help='JSON map: incoming link id -> geometry_lonlat + provenance')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=read_json(a.baseline/'inference_config.json') if a.baseline else None
    node_id=a.node_id or (str(cfg['node_id']) if cfg else None);link=a.link_csv or (resource_path(ROOT,cfg['network']['link_csv']) if cfg else None)
    if not node_id or not link:p.error('Supply --node-id and --link-csv, or --baseline')
    node=a.node_csv or link.parent/'node.csv';labels={};policy=read_json(a.policy) if a.policy else {};context=None
    if cfg and str(cfg['node_id'])==node_id:
        for s in cfg['sections']:
            if s['kind']!='inbound_stopbar':continue
            labels[str(cfg['legs'][s['direction']]['in_link_id'])]={'label':s['direction'],'section_id':s['id'],'region_ids':[r['id'] for r in s['regions']]}
        context={'baseline':str(a.baseline.resolve()),'config_sha256':file_hash(a.baseline/'inference_config.json')}
    profile=from_network(node,link,node_id,policy,labels,read_json(a.stoplines) if a.stoplines else None)
    if context:profile['lane_context']=context
    profile['models']={'vlm':cfg['vlm'] if cfg else {'preset':'default','max_tokens':6144,'temperature':0.},
        'yolo':cfg['yolo'] if cfg else {'weights':'~/.cache/net2cell-vlm/weights/yolo26n.pt','expected_sha256':'9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef','device':'auto','imgsz':1280,'confidence':.25,'occluder_classes':['traffic light','stop sign']}}
    prepared=prepare_profile(profile);write_json(a.output,profile)
    write_json(a.output.with_suffix('.plan.json'),{'profile':prepared,'requests':[r for app in prepared['approaches'] for r in initial_requests(prepared,app)]})
    for app in prepared['approaches']:print(app['id'],app['label'],round(app['heading_deg'],1),app['reference_kind'],round(app['reference_station_m'],1),flush=True)
    print({'site_id':node_id,'approaches':len(prepared['approaches']),'profile':str(a.output)})


if __name__=='__main__':main()
