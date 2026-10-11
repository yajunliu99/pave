"""Add traffic-control hypotheses to an immutable atlas snapshot without lane edits."""
import argparse
import math
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.hybrid.common import read_json,write_json,file_hash,resource_path
from movement_fixer.fusion.coordinates import Viewport,local_xy,local_lonlat,bearing_interval,sector_polygon,affine_points
from movement_fixer.fusion.graph import stable_id,time_relation,overlap_fraction
from movement_fixer.fusion.bundle import scene_to_primary,normalized_polygon


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    base=a.base.resolve();output=a.output.resolve()
    if base==output:raise ValueError('Use a separate atlas output')
    if read_json(a.audit/'status.json')['state']!='complete':raise ValueError('Control audit incomplete')
    # Older base atlases still carry a multi-file page; the page is now built only by dashboard/build_standalone_dashboard.py.
    shutil.copytree(base,output,dirs_exist_ok=True,ignore=lambda folder,names:{'build_manifest.json','validation.json'}|({'index.html','atlas.js','data.js'} if Path(folder)==base else set()))
    shutil.copytree(a.audit,output/'control_evidence',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.png'))
    data=read_json(output/'atlas_data.json');graph=read_json(output/'lane_graph.json');manifest=read_json(a.audit/'views_manifest.json');predictions=read_json(a.audit/'predictions.json')
    if data.get('controls'):raise ValueError('Use the atlas snapshot before control augmentation, preserving prior runs separately')
    site_id=str(manifest.get('site_id') or manifest.get('profile',{}).get('site_id','unknown'))
    baseline_cfg=base/'baseline/inference_config.json'
    if baseline_cfg.exists() and str(read_json(baseline_cfg)['node_id'])!=site_id:raise ValueError('Control site does not match atlas node')
    region_lookup={(r['section_id'],r['region_id']):r['id'] for r in data['regions']}
    window=tuple(data['target_window'].split('—')) if data.get('target_window') else None  # as the atlas recorded it
    atlas=Viewport(**data['georeference']['atlas_viewport']);origin=data['origin_lonlat']
    to_atlas=lambda points:[list(atlas.to_pixel(*local_lonlat(*p,origin))) for p in points]
    for v in manifest['views']:
        path=resource_path(ROOT,v['path']);dest=output/'assets'/path.name
        if file_hash(path)!=v['image_sha256']:raise ValueError('Changed control image')
        shutil.copy2(path,dest);xy=local_xy(*v['actual_lonlat'],origin);bearings=bearing_interval([0,0,1,1],v['image_size'],v['hfov_deg'],v['pitch_deg'],v['compass_heading_deg'])
        source={**v,'kind':'gsv','asset':'assets/'+dest.name,'source_path':str(path),'rendered_sha256':file_hash(dest),'source_group':'panorama/'+v['pano_id'],
                'time_relation':v.get('capture_relation',time_relation(v['capture_date'],window)),'valid_time':None,'local_xy':list(xy),'atlas_xy':list(atlas.to_pixel(*v['actual_lonlat'])),
                'in_baseline':False,'analysis_status':'provided_to_control_audit','horizontal_bearing_bounds_deg':list(bearings),'fov_polygon_atlas':to_atlas(sector_polygon(xy,*bearings,100)),
                'display_range_m':100,'pano_projection_heading_deg':None,'pano_yaw_offset_deg':None,'calibrated_height_m':None,'station_reference':manifest.get('coordinate_note','legacy_intersection_center'),'visibility_note':'Control view; signal/sign bearing has no depth. Nominal approach orientation does not establish lane applicability.'}
        if source['id'] in data['sources']:raise ValueError('Control source ID collides with base')
        data['sources'][source['id']]=source;data['views'].append(source)
    controls=[];control_obs=[];bindings=[]
    for section in predictions:
        for c in section['controls']:
            cid=f"control/{site_id}/{section['section_id']}/{c['id']}";item={**c,'id':cid,'section_id':section['section_id'],'direction':section['direction'],
                'object_status':'model_control_hypothesis','movement_constraints_applied':False,'observation_ids':[],'human_verified':False}
            for ref in c['visual_refs']:
                s=data['sources'][ref['source_id']];box=ref['bbox_xyxy'];oid=stable_id('obs/control/',[cid,ref])
                o={'id':oid,'control_id':cid,'section_id':section['section_id'],'source_id':s['id'],'source_group':s['source_group'],'stage':'traffic_control',
                   'pixel_bbox_normalized':box,'finding':ref['finding'],'claim':{k:c[k] for k in ['kind','observed_text','symbol','confidence','applies_to','binding_status']},
                   'capture_date':s['capture_date'],'time_relation':s['time_relation'],'valid_time':None,'actor':'control_vlm','map_location':None,'spatial_candidates':[],
                   'association_status':'control_observation_not_verified_lane_binding'}
                if s['kind']=='gsv':
                    bearings=bearing_interval(box,s['image_size'],s['hfov_deg'],s['pitch_deg'],s['compass_heading_deg'])
                    o.update(bearing_interval_deg=list(bearings),possible_ray_region_atlas=to_atlas(sector_polygon(s['local_xy'],*bearings,100)),location_status='bearing_only_no_depth')
                else:
                    primary=scene_to_primary(normalized_polygon(box,s['image_size']),s['projection'])
                    o.update(map_location=affine_points(primary,data['georeference']['primary_to_atlas']).tolist(),location_status='satellite_pixel_transform')
                control_obs.append(o);item['observation_ids'].append(oid)
            controls.append(item)
            bindings.append({'control_id':cid,'candidate_region_ids':[region_lookup[(section['section_id'],r)] for r in c['lane_candidates']],
                             'scope':c['applies_to'],'status':c['binding_status'],'reason':c['binding_reason'],'verified':False,'movement_constraints_applied':False})
    data['controls']=controls;data['control_bindings']=bindings;data['control_audit_summary']=read_json(a.audit/'summary.json')
    data['control_sampling_profile']=manifest.get('profile');data['control_sampling_history']=read_json(a.audit/'adaptive_history.json') if (a.audit/'adaptive_history.json').exists() else []
    data['control_unresolved']={s['section_id']:s.get('unresolved',[]) for s in predictions};data['observations'].extend(control_obs)
    for section in {r['section_id'] for r in data['regions']}:
        regions=[r for r in data['regions'] if r['section_id']==section]
        for v in data['views']:
            if v.get('sampling_domain')!='traffic_control':continue
            wedge=sector_polygon(v['local_xy'],*v['horizontal_bearing_bounds_deg'],100);overlap=max(overlap_fraction(r['polygon_local_m'],wedge) for r in regions)
            data['coverage'].append({'section_id':section,'view_id':v['id'],'horizontal_fov_possible':overlap>.02,'max_nominal_lane_overlap':round(overlap,4),
                'model_association_count':0,'actual_visibility_verified':False,'association_verified':False,'status':'fov_only' if overlap>.02 else 'not_in_nominal_fov',
                'pose_error_m':None,'date_relation':v['time_relation'],'new_view':True})
    graph.update(controls=controls,control_bindings=bindings,observations=data['observations'],signal_phase_plan='not_observable_from_static_imagery',control_constraints_applied=False)
    for filename,obj in [('atlas_data.json',data),('lane_graph.json',graph),('sources.json',data['sources']),('coverage_matrix.json',data['coverage'])]:write_json(output/filename,obj)
    geo=read_json(output/'atlas.geojson')
    for v in data['views']:
        if v.get('sampling_domain')=='traffic_control':geo['features'].append({'type':'Feature','id':v['id'],'geometry':{'type':'Point','coordinates':v['actual_lonlat']},'properties':{k:v.get(k) for k in ['pano_id','capture_date','compass_heading_deg','camera_role','camera_position_error_m']}})
    write_json(output/'atlas.geojson',geo)
    write_json(output/'control_provenance.json',{'base_atlas':str(base),'base_data_sha256':file_hash(base/'atlas_data.json'),'audit_manifest_sha256':file_hash(a.audit/'views_manifest.json'),
        'audit_predictions_sha256':file_hash(a.audit/'predictions.json'),'lane_and_movement_mutations':False,'reviewer_answers_used':False})
    report=(output/'report.md').read_text(encoding='utf-8')
    base_data=read_json(base/'atlas_data.json')
    report=report.replace(f"- {len(base_data['views'])} 张视图、{len({v['pano_id'] for v in base_data['views']})} 个全景拍摄点。",f"- {len(data['views'])} 张视图、{len({v['pano_id'] for v in data['views']})} 个全景拍摄点。")
    report=report.replace(f"- {len(base_data['observations'])} 条观测、{len(base_data['associations'])} 条模型关联。",f"- {len(data['observations'])} 条观测、{len(data['associations'])} 条车道模型关联；控制设施关联单独保存。")
    (output/'report.md').write_text(report,encoding='utf-8')
    with (output/'report.md').open('a',encoding='utf-8') as f:
        f.write('\n## 信号灯与标志近点复核\n\n'+f"新增 {len(manifest['views'])} 幅控制设施视图，{len(controls)} 项按方向分组的模型设施假设；不同方向可能重复看到同一设施，不能将其当作物理设施总数。\n\n")
        f.write('流程：近停止线及路口内部抬头取景 → YOLO26n 候选 → VLM 选择目标 → 同全景调整朝向/仰角/视场获取特写 → VLM 复核文字、符号、朝向和适用范围。每条结论保留原生图像框、拍摄日期和来源组。YOLO 标准类别只辅助发现交通灯和 stop sign，不能识别所有标志类别或读取文字。\n\n')
        f.write('静态灯色不代表信号配时或永久通行许可。标志和信号对车道的关联是待核假设，未自动改写车道用途或 movement。\n\n[Google 取景参数说明](https://developers.google.com/maps/documentation/streetview/request-streetview)；本地目标射线只用于调整取景，不推算设施地面坐标。\n')
    print({'control_hypotheses':len(controls),'new_views':len(manifest['views']),'control_observations':len(control_obs),'output':str(output)})


if __name__=='__main__':main()
