"""Check a generic run's provenance and structural invariants, not semantic accuracy."""
import argparse
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.hybrid.common import read_json,write_json,file_hash
from movement_fixer.fusion.controls import validate_controls


def validate_run(run):
    run=Path(run).resolve();folder=run/'audit';m=read_json(folder/'views_manifest.json');p=m['profile'];policy=p['policy'];pred=read_json(folder/'predictions.json');summary=read_json(folder/'summary.json')
    views={v['id']:v for v in m['views']};approaches={a['id']:a for a in p['approaches']};checks={}
    def check(k,v):
        checks[k]=bool(v)
        if not v:raise ValueError('Run invariant failed: '+k)
    check('finished',read_json(folder/'status.json')['state']=='complete')
    check('one_result_per_approach',{s['approach_id'] for s in pred}==set(approaches) and len(pred)==len(approaches))
    check('unique_view_ids',len(views)==len(m['views']))
    check('image_hashes',all(file_hash(v['path'])==v['image_sha256'] for v in views.values()))
    check('valid_approach_membership',all(v['approach_id'] in approaches for v in views.values()))
    check('distinct_context_panoramas',all(len({v['pano_id'] for v in views.values() if v['approach_id']==aid and v['view_role']=='control_context'})==sum(v['approach_id']==aid and v['view_role']=='control_context' for v in views.values()) for aid in approaches))
    details=[v for v in views.values() if v['view_role']=='control_detail']
    check('detail_parents_same_panorama',all(v.get('parent_view_id') in views and views[v['parent_view_id']]['pano_id']==v['pano_id'] for v in details))
    check('bounded_detail_fov',all(policy['detail_min_fov_deg']-.01<=v['hfov_deg']<=policy['detail_max_fov_deg']+.01 for v in details))
    for s in pred:
        a=approaches[s['approach_id']];sources=set(views)|({'scene_'+a['section_id']} if p.get('lane_context') else set())
        validate_controls(copy.deepcopy(s),a['region_ids'],sources)
    check('no_inferred_movement_constraints',all(s['movement_constraints_applied'] is False for s in pred))
    history=read_json(folder/'adaptive_history.json') if (folder/'adaptive_history.json').exists() else []
    check('bounded_rounds',all(len(h['rounds'])<=policy['max_rounds'] for h in history))
    check('bounded_context_views',all(sum(v['approach_id']==aid and v['view_role']=='control_context' for v in views.values())<=policy['max_context_views_per_approach'] for aid in approaches))
    receipt=read_json(folder/'initial_execution.json') if (folder/'initial_execution.json').exists() else summary
    check('bounded_provider_calls',receipt['api_calls']<=policy['max_vlm_calls'] and receipt['acquisition_counts']['metadata_requests']<=policy['max_metadata_requests'] and receipt['acquisition_counts']['image_requests']<=policy['max_image_requests'])
    if (folder/'initial_predictions.json').exists():check('cache_replay_predictions_identical',read_json(folder/'initial_predictions.json')==pred)
    atlas=run/'atlas'
    if (atlas/'control_provenance.json').exists():
        pr=read_json(atlas/'control_provenance.json');base=Path(pr['base_atlas']);d=read_json(atlas/'atlas_data.json');b=read_json(base/'atlas_data.json');g=read_json(atlas/'lane_graph.json');bg=read_json(base/'lane_graph.json')
        check('base_source_unchanged',file_hash(base/'atlas_data.json')==pr['base_data_sha256'])
        check('lane_geometry_and_classes_unchanged',d['regions']==b['regions'] and d['lanes']==b['lanes'])
        check('movement_hypotheses_unchanged',g['baseline_movement_hypotheses']==bg['baseline_movement_hypotheses'])
        check('coverage_matrix_complete',len(d['coverage'])==len({r['section_id'] for r in d['regions']})*len(d['views']))
    report={'status':'passed','checks':checks,'site_id':p['site_id'],'counts':{'approaches':len(approaches),'views':len(views),'panoramas':len({v['pano_id'] for v in views.values()}),'control_hypotheses':sum(len(s['controls']) for s in pred)},
            'scope':'Structural/provenance checks only; not an accuracy score or proof all signs are visible.',
            'stop_reasons':summary['sampling_stop_reasons'],'reference_kinds':{a['id']:a['reference_kind'] for a in p['approaches']}}
    write_json(run/'validation.json',report)
    if atlas.exists():write_json(atlas/'validation.json',report)
    print({'status':'passed','checks':len(checks),'site_id':p['site_id']});return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();validate_run(a.run)
