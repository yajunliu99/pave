"""Bounded, replayable Street View acquisition. Cache keys include the full pose."""
import io
import math
import os
import re
from datetime import datetime,timezone
from pathlib import Path
import requests
from PIL import Image
from ..hybrid.common import read_json,write_json,file_hash,digest
from .coordinates import local_lonlat,local_xy
from .control_sampling import assess_panorama,initial_requests,request_at,path_pose,safe_id,heading


class AcquisitionUnavailable(RuntimeError):pass


class StreetViewClient:
    def __init__(self,root,cache,policy,cache_only=False,session=None):
        self.root=Path(root);self.cache=Path(cache).resolve();self.cache.mkdir(parents=True,exist_ok=True)
        self.policy=policy;self.cache_only=cache_only;self.session=session or requests.Session();self.key=None
        self.counts={'metadata_requests':0,'image_requests':0,'metadata_cache_hits':0,'image_cache_hits':0}

    def _key(self):
        if self.key is None:
            env=dict(os.environ);path=self.root/'.env'
            if path.exists():
                for line in path.read_text(encoding='utf-8-sig').splitlines():
                    if '=' in line and not line.lstrip().startswith('#'):
                        k,v=line.split('=',1);env.setdefault(k.strip(),v.strip().strip('"').strip("'"))
            self.key=env.get('GSV_API_KEY') or env.get('GOOGLE_MAPS_API_KEY') or env.get('GOOGLE_API_KEY')
            if not self.key:raise AcquisitionUnavailable('missing_streetview_key')
        return self.key

    def metadata(self,request):
        # A request naming a panorama ('pano_id') asks for that one; otherwise the service picks one near the location.
        if request.get('pano_id'):params={'pano':request['pano_id']}
        else:ll=request['requested_lonlat'];params={'location':f'{ll[1]:.8f},{ll[0]:.8f}','radius':request['radius_m'],'source':'outdoor'}
        path=self.cache/'metadata'/(digest(params)+'.json')
        if path.exists():
            cached=read_json(path)
            if cached.get('status') in ('OK','ZERO_RESULTS') or self.cache_only:
                self.counts['metadata_cache_hits']+=1;return cached
        if self.cache_only:raise AcquisitionUnavailable('metadata_cache_miss')
        if self.counts['metadata_requests']>=self.policy['max_metadata_requests']:raise AcquisitionUnavailable('metadata_budget_exhausted')
        key=self._key();self.counts['metadata_requests']+=1
        try:
            response=self.session.get('https://maps.googleapis.com/maps/api/streetview/metadata',params={**params,'key':key},timeout=30)
            body=response.json() if response.status_code==200 else {'status':'HTTP_'+str(response.status_code)}
        except (requests.RequestException,ValueError):raise AcquisitionUnavailable('metadata_transport_error') from None
        record={'status':body.get('status','invalid_response'),'request':params,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
        if record['status']=='OK':
            if not all(k in body for k in ('location','pano_id')):raise AcquisitionUnavailable('invalid_metadata_response')
            record.update(actual_lonlat=[body['location']['lng'],body['location']['lat']],pano_id=body['pano_id'],capture_date=body.get('date'),copyright=body.get('copyright'))
        write_json(path,record);return record

    def image(self,view):
        size=self.policy['image_size'];params={'pano':view['pano_id'],'size':f'{size[0]}x{size[1]}','heading':round(view['compass_heading_deg'],4),
                'pitch':round(view['pitch_deg'],4),'fov':round(view['hfov_deg'],4),'return_error_code':'true'}
        signature=digest(params);path=self.cache/'images'/(signature+'.jpg');sidecar=path.with_suffix('.json')
        if path.exists() and sidecar.exists() and read_json(sidecar).get('image_sha256')==file_hash(path):self.counts['image_cache_hits']+=1
        else:
            if self.cache_only:raise AcquisitionUnavailable('image_cache_miss')
            if self.counts['image_requests']>=self.policy['max_image_requests']:raise AcquisitionUnavailable('image_budget_exhausted')
            key=self._key();self.counts['image_requests']+=1
            try:response=self.session.get('https://maps.googleapis.com/maps/api/streetview',params={**params,'key':key},timeout=40)
            except requests.RequestException:raise AcquisitionUnavailable('image_transport_error') from None
            if response.status_code!=200:raise AcquisitionUnavailable('image_HTTP_'+str(response.status_code))
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with Image.open(io.BytesIO(response.content)) as im:im.convert('RGB').save(path,quality=95,subsampling=0)
            except (OSError,ValueError):raise AcquisitionUnavailable('invalid_image_response') from None
            write_json(sidecar,{'request':params,'image_sha256':file_hash(path),'retrieved_utc':datetime.now(timezone.utc).isoformat()})
        with Image.open(path) as im:actual_size=list(im.size)
        return {**view,'path':str(path),'image_size':actual_size,'image_sha256':file_hash(path),'image_request_signature':signature,'image_origin':'official_static_street_view'}


def probe_points(xy,unit,search):
    """Points around a requested camera position, along (`unit`) and across the road, nearest first.

    The metadata service answers a location with the panorama it prefers there, usually the latest. Earlier
    panoramas were taken a few metres apart, so asking at nearby points with a small radius brings them out."""
    step=search['step_m'];perp=(unit[1],-unit[0]);points=[]
    for i in range(-int(search['along_m']//step),int(search['along_m']//step)+1):
        for j in range(-int(search['across_m']//step),int(search['across_m']//step)+1):
            points.append((xy[0]+i*step*unit[0]+j*step*perp[0],xy[1]+i*step*unit[1]+j*step*perp[1]))
    return sorted(points,key=lambda q:math.dist(q,xy))


def official(meta):
    """A panorama from Google's own capture. Searching with small radii also turns up contributed panoramas,
    including indoor ones, which the outdoor filter does not always exclude; their copyright names the contributor."""
    return re.fullmatch(r'©\s*(\d{4}\s*)?Google',(meta.get('copyright') or '').strip()) is not None


def panoramas_near(client,request,lonlats,radius_m,official_only=True):
    """Distinct panoramas returned by metadata requests at the given points (one request per point, cached)."""
    found={}
    for lonlat in lonlats:
        meta=client.metadata({**request,'requested_lonlat':list(lonlat),'radius_m':radius_m})
        if meta.get('status')=='OK' and (official(meta) or not official_only):found.setdefault(meta['pano_id'],meta)
    return list(found.values())


def context_from_metadata(profile,approach,request,meta,assessment):
    station=assessment['actual_station_m'];point,u=path_pose(approach,station)
    # A height prior aims the camera only; it never locates an observed signal in 3D.
    distance=max(8,approach['carriageway_width_m']-station)
    delta=profile['policy']['height_delta_prior_m'];pitch=profile['policy']['context_pitch_deg']
    if pitch is None:pitch=max(3,min(25,math.degrees(math.atan2(delta,distance))))
    return {**meta,**assessment,'id':'gsv_control_'+safe_id(approach['id'])+'_'+digest([meta['pano_id'],heading(u),pitch])[:10],
            'approach_id':approach['id'],'direction':approach['label'],'target_section':approach['section_id'],'sampling_domain':'traffic_control',
            'sampling_position':request['role'],'view_role':'control_context','camera_role':'control_context','reverse_view':False,
            'signed_station_m':station,'requested_station_m':request['requested_station_m'],'compass_heading_deg':heading(u),'pitch_deg':pitch,
            'hfov_deg':profile['policy']['context_fov_deg'],'pitch_basis':'configured override' if profile['policy']['context_pitch_deg'] is not None else f'{delta}m target-minus-camera height prior / longitudinal range; aim only, not calibration',
            'reference_kind':approach['reference_kind'],'request_id':request['id'],'camera_position_error_m':None}


def acquire_context(profile,approach,request,client,existing,require_preferred_date=False):
    record={'request':request}
    try:meta=client.metadata(request)
    except AcquisitionUnavailable as e:return None,{**record,'status':str(e)}
    assessment=assess_panorama(profile,approach,request,meta);record.update(metadata=meta,assessment=assessment)
    if not assessment['usable']:return None,{**record,'status':'rejected_pose'}
    if any(v['pano_id']==meta['pano_id'] for v in existing):return None,{**record,'status':'duplicate_panorama'}
    xy=local_xy(*meta['actual_lonlat'],profile['origin_lonlat'])
    if any(math.dist(xy,local_xy(*v['actual_lonlat'],profile['origin_lonlat']))<profile['policy']['minimum_distinct_position_m'] for v in existing):return None,{**record,'status':'duplicate_position'}
    if require_preferred_date and assessment['capture_relation'] in ('historical','later','unknown_date'):
        search=profile['policy'].get('date_search')
        if not search:return None,{**record,'status':'outside_preferred_date_window'}
        points=[local_lonlat(*q,profile['origin_lonlat']) for q in probe_points(request['requested_local_xy'],approach['forward_unit'],search)]
        try:
            found=panoramas_near(client,request,points,search['radius_m'])
            for pid in profile['policy']['known_panoramas'].get(approach['section_id'],[]):
                m=client.metadata({**request,'pano_id':pid})
                if m.get('status')=='OK' and official(m) and all(m['pano_id']!=f['pano_id'] for f in found):found.append(m)
        except AcquisitionUnavailable as e:return None,{**record,'status':str(e)}
        ok=[]
        for m in found:
            a=assess_panorama(profile,approach,request,m)
            xy=local_xy(*m['actual_lonlat'],profile['origin_lonlat'])
            if a['usable'] and a['capture_relation']=='target_window' and not any(v['pano_id']==m['pano_id'] or
                    math.dist(xy,local_xy(*v['actual_lonlat'],profile['origin_lonlat']))<profile['policy']['minimum_distinct_position_m'] for v in existing):
                ok.append((a['pose_score'],m,a))
        record['date_search']={'probes':len(points),'panoramas':len(found),'in_window':len(ok)}
        if not ok:return None,{**record,'status':'outside_preferred_date_window'}
        _,meta,assessment=min(ok,key=lambda x:x[0]);record.update(metadata=meta,assessment=assessment)
    try:view=client.image(context_from_metadata(profile,approach,request,meta,assessment))
    except AcquisitionUnavailable as e:return None,{**record,'status':str(e)}
    return view,{**record,'status':'accepted','view_id':view['id']}


def acquire_initial(profile,client):
    views=[];attempts=[];coverage=[]
    for approach in profile['approaches']:
        selected=[]
        for request in initial_requests(profile,approach):
            candidates=[request]+[request_at(profile,approach,request['requested_station_m']+shift*approach['carriageway_width_m'],request['role']+'_fallback') for shift in (-.45,.45)]
            deferred=[];accepted=False
            for candidate in candidates:
                view,record=acquire_context(profile,approach,candidate,client,selected,require_preferred_date=True);attempts.append(record)
                if record['status']=='outside_preferred_date_window':deferred.append(record)
                if view:selected.append(view);accepted=True;break
                if record['status'] in ('metadata_budget_exhausted','image_budget_exhausted','missing_streetview_key'):break
            if not accepted and profile['policy']['date_fallback']:
                for old in sorted(deferred,key=lambda r:r['assessment']['pose_score']):
                    view,record=acquire_context(profile,approach,old['request'],client,selected);record['date_fallback']=True;attempts.append(record)
                    if view:view['date_fallback_reason']='No usable preferred-window panorama among bounded candidate positions';selected.append(view);break
        views.extend(selected);coverage.append({'approach_id':approach['id'],'views':len(selected),'status':'complete' if len(selected)==3 else 'partial' if selected else 'unavailable'})
    return {'schema':'traffic-control-views-2','site_id':profile['site_id'],'origin_lonlat':profile['origin_lonlat'],'profile':profile,
            'views':views,'positions':attempts,'coverage':coverage,'acquisition_counts':client.counts,'cache_root':str(client.cache),
            'coordinate_note':profile['coordinate_note']}


def alternative_requests(profile,approach,views,round_index):
    """Bounded new stations around existing views, without fabricating target depth."""
    contexts=[v for v in views if v['view_role']=='control_context'];scale=approach['carriageway_width_m']
    anchors=sorted(contexts,key=lambda v:v['signed_station_m'],reverse=True)
    requests=[];seen=set()
    for v in anchors:
        for shift in (-.55,.55):
            r=request_at(profile,approach,v['signed_station_m']+shift*scale,f'adaptive_round{round_index}')
            station=round(r['requested_station_m'],1)
            if station in seen:continue
            if any(abs(station-x['signed_station_m'])<profile['policy']['minimum_distinct_position_m'] for x in contexts):continue
            seen.add(station);requests.append(r)
    return requests
