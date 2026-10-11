"""Generic bounded acquisition/recognition feedback loop for traffic controls."""
import argparse
import html
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from movement_fixer.hybrid.common import read_json,write_json,file_hash,digest,require
from movement_fixer.hybrid.evidence_visuals import sheet,transport_jpeg
from movement_fixer.hybrid.inference import StageClient
from movement_fixer.hybrid.yolo_aux import run_yolo
from movement_fixer.fusion.controls import detail_pose,validate_inventory,validate_controls,needs_resampling,reading_signature
from movement_fixer.fusion.control_sampling import safe_id
from movement_fixer.fusion.control_acquisition import StreetViewClient,AcquisitionUnavailable,acquire_context,alternative_requests
from movement_fixer.fusion.control_prompts import inventory_prompt,final_prompt


def detect(contexts,models,output):
    images={};sources=output/'sources';sources.mkdir(exist_ok=True)
    for v in contexts:
        path=Path(v['path']);require(file_hash(path)==v['image_sha256'],'Image changed')
        images[v['id']]=Image.open(path).convert('RGB');images[v['id']].save(sources/(v['id']+'.png'))
    config={'yolo':{**models['yolo'],'occluder_classes':['traffic light','stop sign']},'sources':{v['id']:v for v in contexts},'context_views':[],
            'gsv_views':[{'id':v['id'],'source_id':v['id'],'regions':[]} for v in contexts]}
    return run_yolo(config,ROOT,output,images,{},disabled=False)


def source_meta(views):
    return {v['id']:{k:v.get(k) for k in ['pano_id','actual_lonlat','capture_date','capture_relation','signed_station_m','compass_heading_deg','hfov_deg','pitch_deg','reference_kind','parent_view_id']} for v in views}


def entries_for(views,scene_id=None,scene_path=None):
    entries=[]
    if scene_id:entries.append((scene_id,'Candidate IDs only; travel UP; boundaries not verified',Image.open(scene_path).convert('RGB')))
    for v in views:
        path=Path(v['path']);require(file_hash(path)==v['image_sha256'],'View hash mismatch')
        entries.append((v['id'],f"{v['view_role']} / {v['capture_date']} / {v['compass_heading_deg']:.1f} deg / station {v['signed_station_m']:.1f} m",Image.open(path).convert('RGB')))
    return entries


def render_report(output,manifest,predictions,summary):
    assets=output/'assets';assets.mkdir(exist_ok=True);lookup={v['id']:v for v in manifest['views']}
    for v in lookup.values():shutil.copy2(v['path'],assets/(v['id']+'.jpg'))
    parts=['<!doctype html><meta charset="utf-8"><title>Traffic control evidence</title><style>body{font:15px/1.5 system-ui;max-width:1200px;margin:auto;padding:24px}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}table{width:100%;border-collapse:collapse}small{color:#666}a{margin-right:10px}</style>',
           '<h1>Traffic control evidence · '+html.escape(str(manifest['site_id']))+'</h1>',
           '<p>Control recognition and lane binding are separate. Capture dates do not establish current rules. Static lamps do not establish signal timing.</p>',
           '<p>'+html.escape(json.dumps(summary))+'</p>']
    for s in predictions:
        parts+=['<h2>'+html.escape(s['direction'])+'</h2><p>'+html.escape(s['sampling_status'])+'</p><table><tr><th>Kind</th><th>Reading</th><th>Scope / binding</th><th>Evidence</th></tr>']
        for c in s['controls']:
            refs=' '.join('<a href="assets/'+html.escape(r['source_id'])+'.jpg">'+html.escape(r['source_id'])+'</a>' for r in c['visual_refs'] if r['source_id'] in lookup)
            parts.append('<tr><td>'+html.escape(c['kind'])+'</td><td>'+html.escape(c['observed_text'] or c['symbol'])+'<br><small>'+html.escape(c['description'])+'</small></td><td>'+html.escape(c['applies_to']+' / '+c['binding_status'])+'</td><td>'+refs+'</td></tr>')
        parts.append('</table><p>'+html.escape(' | '.join(s.get('unresolved',[])))+'</p>')
    (output/'index.html').write_text('\n'.join(parts),encoding='utf-8')


def run_audit(manifest_path,output,cache_only=False,reuse_cache_from=()):
    manifest=read_json(manifest_path);require(manifest.get('schema')=='traffic-control-views-2','Rebuild imagery with the generic profile interface')
    profile=manifest['profile'];policy=profile['policy'];models=profile['models'];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    signature=digest({'manifest':file_hash(manifest_path),'version':'generic-control-audit-2-date-priority'})
    if (output/'input_manifest.json').exists():require(read_json(output/'input_manifest.json')['signature']==signature,'Inputs changed: choose a new run directory')
    write_json(output/'input_manifest.json',{'signature':signature,'views':str(Path(manifest_path).resolve()),'views_sha256':file_hash(manifest_path),'reviewer_answers_used':False})
    write_json(output/'status.json',{'state':'running'});client=StreetViewClient(ROOT,manifest['cache_root'],policy,cache_only=cache_only)
    # Acquisition and adaptive stages share one per-run quota; cache hits remain allowed.
    initial_counts=manifest['acquisition_counts'];client.counts['metadata_requests']=initial_counts['metadata_requests'];client.counts['image_requests']=initial_counts['image_requests']
    vlm=StageClient(ROOT,output,{**models['vlm'],'max_api_calls':policy['max_vlm_calls']},cache_only=cache_only,reuse_cache_from=reuse_cache_from)
    allviews=list(manifest['views']);detections=detect(allviews,models,output) if allviews else {};predictions=[];history=[];attempts=[]
    lane_context=profile.get('lane_context');baseline=Path(lane_context['baseline']) if lane_context else None
    if baseline:require(file_hash(baseline/'inference_config.json')==lane_context['config_sha256'],'Lane context config changed')
    try:
        for approach in profile['approaches']:
            aid=approach['id'];folder=output/safe_id(aid);folder.mkdir(exist_ok=True);views=[v for v in allviews if v['approach_id']==aid]
            scene_path=baseline/'surface_audits'/approach['section_id']/'scene_candidates.png' if baseline else None
            scene_id='scene_'+approach['section_id'] if scene_path and scene_path.exists() else None
            if not views:
                predictions.append({'approach_id':aid,'direction':approach['label'],'section_id':approach['section_id'],'controls':[],'unresolved':['No usable Street View camera was returned'],'sampling_status':'no_usable_imagery','human_verified':False,'movement_constraints_applied':False});continue
            prior_readings=None;result=None;stop_reason='round_limit';round_records=[]
            for round_index in range(policy['max_rounds']):
                rf=folder/f'round_{round_index+1}';rf.mkdir(exist_ok=True)
                if vlm.calls>=policy['max_vlm_calls']-1:stop_reason='vlm_budget_exhausted';break
                contexts=[v for v in views if v['view_role']=='control_context'];meta=source_meta(views)
                proposals={v['id']:[{'class':d['class'],'confidence':d['confidence'],'bbox_xyxy':[c/(v['image_size'][0] if j%2==0 else v['image_size'][1]) for j,c in enumerate(d['bbox_xyxy'])]} for d in detections[v['id']]['detections'] if d['class'] in ['traffic light','stop sign']] for v in contexts}
                entries=entries_for(views,scene_id,scene_path);canvas=sheet(entries,rf/'inventory.png',cell=(900,800),columns=3)
                inventory=vlm.run('inventory_'+safe_id(aid)+f'_r{round_index+1}',transport_jpeg(canvas,rf/'inventory.png'),inventory_prompt(approach,meta,proposals,policy['max_details_per_round']),lambda x:validate_inventory(x,set(meta)))
                require(len(inventory['detail_requests'])<=policy['max_details_per_round'],'Inventory exceeded policy detail limit');write_json(rf/'inventory.json',inventory)
                lookup={v['id']:v for v in views};detail_notes=[]
                for i,r in enumerate(inventory['detail_requests']):
                    parent=lookup[r['source_id']];pose=detail_pose(parent,r['bbox_xyxy'],policy['detail_min_fov_deg'],policy['detail_max_fov_deg'])
                    key=digest([parent['pano_id'],pose])[:12];sid='gsv_detail_'+safe_id(aid)+'_'+key
                    if any(v['id']==sid for v in views):detail_notes.append({'source':r['source_id'],'status':'duplicate_pose'});continue
                    view={**parent,**pose,'id':sid,'view_role':'control_detail','camera_role':'control_detail','sampling_position':'detail',
                          'parent_view_id':parent['id'],'requested_bbox':r['bbox_xyxy'],'detail_question':r['question'],'adaptive_round':round_index+1}
                    try:view=client.image(view)
                    except AcquisitionUnavailable as e:detail_notes.append({'source':r['source_id'],'status':str(e)});continue
                    views.append(view);allviews.append(view)
                meta=source_meta(views);canvas=sheet(entries_for(views,scene_id,scene_path),rf/'final.png',cell=(900,800),columns=3)
                result=vlm.run('final_'+safe_id(aid)+f'_r{round_index+1}',transport_jpeg(canvas,rf/'final.png'),final_prompt(approach,meta,scene_id,policy['preferred_date_window']),
                    lambda x:validate_controls(x,approach['region_ids'],set(meta)|({scene_id} if scene_id else set())))
                for r in result.get('resampling_requests',[]):require(r['source_id'] in meta,'Resampling requires a Street View image')
                result.update(approach_id=aid,direction=approach['label'],section_id=approach['section_id']);readings=reading_signature(result)
                record={'round':round_index+1,'views':len(views),'reading_count':len(readings),'new_readings':len(readings-(prior_readings or set())),'needs_resampling':needs_resampling(result),'detail_notes':detail_notes}
                round_records.append(record);write_json(rf/'prediction.json',result)
                if not needs_resampling(result):stop_reason='reading_sufficient';break
                if prior_readings is not None and not readings-prior_readings:stop_reason='no_new_readings';break
                prior_readings=readings
                if round_index+1>=policy['max_rounds']:stop_reason='round_limit';break
                slots=min(policy['max_adaptive_positions_per_round'],policy['max_context_views_per_approach']-len(contexts))
                if slots<=0:stop_reason='context_view_budget';break
                added=[];deferred=[]
                for request in alternative_requests(profile,approach,contexts,round_index+2):
                    view,attempt=acquire_context(profile,approach,request,client,contexts+added,require_preferred_date=True);attempts.append(attempt)
                    if attempt['status']=='outside_preferred_date_window':deferred.append(attempt)
                    if view:added.append(view)
                    if len(added)>=slots or attempt['status'] in ('metadata_budget_exhausted','image_budget_exhausted'):break
                if not added and policy['date_fallback']:
                    for old in sorted(deferred,key=lambda r:r['assessment']['pose_score']):
                        view,attempt=acquire_context(profile,approach,old['request'],client,contexts);attempt['date_fallback']=True;attempts.append(attempt)
                        if view:view['date_fallback_reason']='No distinct preferred-window panorama in bounded adaptive search';added.append(view);break
                if not added:stop_reason='no_distinct_usable_panorama';break
                views.extend(added);allviews.extend(added);detections.update(detect(added,models,output))
            if result is None:result={'approach_id':aid,'direction':approach['label'],'section_id':approach['section_id'],'controls':[],'unresolved':['Inference budget exhausted before a result'],'movement_constraints_applied':False,'human_verified':False}
            result['sampling_status']=stop_reason;predictions.append(result);history.append({'approach_id':aid,'rounds':round_records,'stop_reason':stop_reason})
            write_json(output/'predictions.partial.json',predictions);write_json(output/'adaptive_history.json',history)
            write_json(output/'views_manifest.json',{**manifest,'views':allviews,'adaptive_attempts':attempts,'acquisition_counts':client.counts})
        write_json(output/'predictions.json',predictions);write_json(output/'yolo/all_detections.json',detections)
        final_manifest={**manifest,'views':allviews,'adaptive_attempts':attempts,'acquisition_counts':client.counts};write_json(output/'views_manifest.json',final_manifest)
        summary={'state':'complete','site_id':profile['site_id'],'approaches':len(predictions),'control_hypotheses':sum(len(s['controls']) for s in predictions),
                 'context_views':sum(v['view_role']=='control_context' for v in allviews),'detail_views':sum(v['view_role']=='control_detail' for v in allviews),
                 'api_calls':vlm.calls,'cache_hits':vlm.hits,'reviewer_answers_used':False,'yolo_model':'yolo26n','movement_constraints_applied':False,
                 'sampling_stop_reasons':{s['approach_id']:s['sampling_status'] for s in predictions},'acquisition_counts':client.counts,
                 'acquisition_calls_this_audit':{k:client.counts[k]-initial_counts[k] for k in ('metadata_requests','image_requests')}}
        if not (output/'initial_execution.json').exists():write_json(output/'initial_execution.json',summary)
        if not (output/'initial_predictions.json').exists():write_json(output/'initial_predictions.json',predictions)
        execution={'completed_utc':datetime.now(timezone.utc).isoformat(),'cache_only':cache_only,'summary':summary,'prediction_sha256':file_hash(output/'predictions.json')}
        write_json(output/'executions'/(digest(execution)[:20]+'.json'),execution)
        write_json(output/'summary.json',summary);write_json(output/'status.json',{'state':'complete'});render_report(output,final_manifest,predictions,summary);print(summary);return summary
    except Exception as e:write_json(output/'status.json',{'state':'failed','error_type':type(e).__name__});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--views',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-only',action='store_true');p.add_argument('--reuse-cache-from',type=Path,action='append',default=[]);a=p.parse_args();run_audit(a.views,a.output,a.cache_only,a.reuse_cache_from)


if __name__=='__main__':main()
