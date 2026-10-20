"""Site-independent directed-road sampling with explicit geometric fallbacks.

Coordinates are WGS84 lon/lat; local calculations use a small-area metric frame.
Map widths / lane counts are camera-placement priors, never lane-use evidence.
"""
import copy
import csv
import math
import re
from pathlib import Path
from .coordinates import local_xy,local_lonlat
from ..hybrid.common import require,digest
from ..autoloop.frames import shares_axis

DEFAULT_POLICY={
    "driving_side":"right","default_carriageway_width_m":10.5,"lane_width_prior_m":3.2,
    "context_pitch_deg":None,"height_delta_prior_m":3.5,"context_fov_deg":100.,"image_size":[640,640],
    "near_reference_fraction":.25,"upstream_reference_fraction":.7,"interior_fraction":.4,
    "tangent_trim_m":8.,"tangent_lookback_m":14.,"crosswalk_margin_m":3.,
    "metadata_radius_m":10,"max_snap_distance_m":14.,"max_station_error_m":18.,
    "minimum_distinct_position_m":3.,"max_metadata_requests":64,"max_image_requests":64,
    "max_context_views_per_approach":5,"max_rounds":2,"max_details_per_round":4,
    "max_adaptive_positions_per_round":2,"max_vlm_calls":24,"detail_min_fov_deg":15.,"detail_max_fov_deg":55.,
    "preferred_date_window":None,
    # With a preferred window: search around each requested position for a panorama inside it (see control_acquisition.probe_points).
    "date_search":None,
    # section_id -> panorama ids the location search cannot return because a later capture hides them there (found, for
    # example, in Google Maps' time travel). They join the date search's candidates and are judged like the rest.
    "known_panoramas":{},
    # With a preferred window and no panorama inside it at a position: True takes the best one outside the window
    # (marked date_fallback); False leaves the position without a view.
    "date_fallback":True,
}


def policy_with(overrides=None):
    p={**DEFAULT_POLICY,**(overrides or {})}
    require(p['driving_side'] in ('left','right'),'driving_side must be left or right')
    for k in ('max_rounds','max_details_per_round','max_context_views_per_approach','max_adaptive_positions_per_round','max_metadata_requests','max_image_requests','max_vlm_calls','metadata_radius_m'):
        require(type(p[k]) is int and p[k]>0,'Expected a positive integer for '+k)
    require(p['max_context_views_per_approach']>=3,'At least three initial context slots required')
    require(p['tangent_trim_m']>=0 and p['tangent_lookback_m']>0,'Invalid tangent window')
    require(all(0<=p[k]<=2 for k in ('near_reference_fraction','upstream_reference_fraction','interior_fraction')),'Invalid sampling fraction')
    require(1<=p['max_rounds']<=3,'max_rounds must be 1..3')
    require(1<=p['max_details_per_round']<=4,'max_details_per_round must be 1..4')
    for k in ['default_carriageway_width_m','lane_width_prior_m','metadata_radius_m','max_snap_distance_m','max_station_error_m','minimum_distinct_position_m','max_metadata_requests','max_image_requests','max_vlm_calls']:
        require(isinstance(p[k],(int,float)) and p[k]>0,'Invalid policy '+k)
    require(0<p['detail_min_fov_deg']<=p['detail_max_fov_deg']<=120,'Invalid detail FOV limits')
    require(0<p['context_fov_deg']<=120 and (p['context_pitch_deg'] is None or -75<=p['context_pitch_deg']<=75),'Invalid context pose')
    require(p['height_delta_prior_m']>0,'Invalid height-difference prior')
    require(len(p['image_size'])==2 and all(isinstance(v,int) and 1<=v<=640 for v in p['image_size']),'Image dimensions must be 1..640')
    w=p['preferred_date_window']
    require(w is None or (len(w)==2 and w[0]<=w[1]),'Invalid preferred date window')
    s=p['date_search']
    require(s is None or (all(isinstance(s.get(k),(int,float)) and s[k]>0 for k in ('along_m','across_m','step_m','radius_m'))),'Invalid date search')
    k=p['known_panoramas']
    require(isinstance(k,dict) and all(isinstance(v,list) and all(isinstance(x,str) and x for x in v) for v in k.values()),'known_panoramas maps a section to a list of panorama ids')
    require(not any(k.values()) or (w and s),'known_panoramas need a preferred date window and a date search')
    require(isinstance(p['date_fallback'],bool),'date_fallback must be true or false')
    return p


def safe_id(value):
    raw=str(value);clean=re.sub(r'[^A-Za-z0-9_-]','_',raw)[:48]
    return clean if clean==raw else clean+'_'+digest(raw)[:8]


def heading(unit):return math.degrees(math.atan2(unit[0],unit[1]))%360
def delta_heading(a,b):return abs((a-b+180)%360-180)
def length(line):return sum(math.dist(a,b) for a,b in zip(line,line[1:]))


def clean_line(points):
    out=[]
    for p in points:
        require(len(p)>=2 and all(math.isfinite(float(v)) for v in p[:2]),'Invalid road coordinate')
        q=[float(p[0]),float(p[1])]
        if not out or math.dist(out[-1],q)>1e-6:out.append(q)
    require(len(out)>=2,'Road needs two distinct coordinates')
    return out


def point_at(line,station,forward_unit=None):
    """Station 0 is the incoming road endpoint; negative is upstream arc length."""
    total=length(line);distance=total+station
    # A station at the far end can land a rounding error past it (terminal_unit asks for -length exactly).
    if distance<-1e-6:raise ValueError('Station lies beyond available incoming geometry')
    distance=max(distance,0.)
    if distance>total:
        u=forward_unit or unit_between(line[-2],line[-1]);return [line[-1][i]+station*u[i] for i in (0,1)]
    for a,b in zip(line,line[1:]):
        size=math.dist(a,b)
        if distance<=size:return [a[i]+(b[i]-a[i])*distance/size for i in (0,1)]
        distance-=size
    return list(line[-1])


def unit_between(a,b):
    size=math.dist(a,b)
    require(size>1e-8,'Degenerate tangent')
    return [(b[i]-a[i])/size for i in (0,1)]


def terminal_unit(line,p):
    total=length(line);trim=min(p['tangent_trim_m'],total*.2);look=min(p['tangent_lookback_m'],total-trim)
    return unit_between(point_at(line,-trim-look),point_at(line,-trim))


def path_pose(approach,station):
    line=approach['centerline_local'];u=approach['forward_unit'];point=point_at(line,station,u)
    if station<-approach.get('tangent_trim_m',8):
        lo=max(-length(line),station-4);hi=min(0,station+4)
        u=unit_between(point_at(line,lo),point_at(line,hi))
    return point,u


def project_to_path(approach,point):
    line=approach['centerline_local'];total=length(line);distance=0.;best=None
    segments=list(zip(line,line[1:]))
    if approach['allow_interior']:
        end=line[-1];u=approach['forward_unit'];segments.append((end,[end[i]+approach['interior_limit_m']*u[i] for i in (0,1)]))
    for a,b in segments:
        size=math.dist(a,b);u=unit_between(a,b);t=max(0,min(size,sum((point[i]-a[i])*u[i] for i in (0,1))))
        closest=[a[i]+t*u[i] for i in (0,1)];offset=math.dist(point,closest)
        if best is None or offset<best['distance_m']:
            best={'station_m':distance+t-total,'lateral_m':(point[0]-closest[0])*u[1]-(point[1]-closest[1])*u[0],'distance_m':offset,'closest':closest}
        distance+=size
    return best


def prepare_profile(config):
    cfg=copy.deepcopy(config);require(cfg.get('schema')=='control-site-1','Expected control-site-1')
    require(cfg.get('site_id') is not None,'site_id required');cfg['site_id']=str(cfg['site_id']);cfg['policy']=policy_with(cfg.get('policy'))
    origin=cfg.get('origin_lonlat');require(isinstance(origin,list) and len(origin)==2,'origin_lonlat required')
    require(-180<=origin[0]<=180 and -85<origin[1]<85,'Invalid site origin')
    approaches=cfg.get('approaches',[]);require(approaches,'No motor-vehicle incoming approaches')
    require(len({a['id'] for a in approaches})==len(approaches),'Duplicate approach IDs')
    require(len({safe_id(a['id']) for a in approaches})==len(approaches),'Colliding approach slugs')
    for a in approaches:
        a['id']=str(a['id']);a.setdefault('label',a['id']);a.setdefault('section_id','in_'+safe_id(a['id']));a.setdefault('region_ids',[])
        require(len(a['region_ids'])==len(set(a['region_ids'])),'Duplicate candidate region IDs')
        coordinates=clean_line(a['centerline_lonlat']);line=[list(local_xy(*p,origin)) for p in coordinates]
        require(length(line)>2,'Incoming geometry too short')
        # Caller supplies upstream-to-junction order; reject silent orientation mistakes.
        require(math.hypot(*line[-1])<=math.hypot(*line[0])+1,'Incoming centerline must end at junction')
        a['centerline_local']=line;a['forward_unit']=terminal_unit(line,cfg['policy']);a['heading_deg']=heading(a['forward_unit']);a['tangent_trim_m']=cfg['policy']['tangent_trim_m']
        width=a.get('carriageway_width_m',cfg['policy']['default_carriageway_width_m']);require(isinstance(width,(int,float)) and 2<=width<=40,'Invalid carriageway width prior')
        a['carriageway_width_m']=width;a.setdefault('width_source','policy_fallback_not_observed_width')
        require(a.get('centerline_reference') in ('directed_carriageway','shared_road_axis'),'Unknown centerline reference')
        a.setdefault('allow_interior',False);a['interior_limit_m']=min(20,max(5,width))
    require(len({a['section_id'] for a in approaches})==len(approaches),'Duplicate section IDs')
    for a in approaches:
        stop=a.get('stopline')
        if stop:
            require(stop.get('provenance') and stop.get('geometry_lonlat'),'Stopline must have geometry and provenance')
            require(len(stop['geometry_lonlat'])>=2,'Stopline geometry must have at least two points')
            pts=[local_xy(*p,origin) for p in stop['geometry_lonlat']];center=[sum(p[i] for p in pts)/len(pts) for i in (0,1)]
            projection=project_to_path(a,center)
            require(projection['distance_m']<max(30,a['carriageway_width_m']*2),'Stopline too far from approach')
            a['reference_station_m']=min(0,projection['station_m']);a['reference_kind']='provided_stopline';a['reference_provenance']=stop['provenance']
        else:
            cross=[b['carriageway_width_m']/max(.4,abs(math.sin(math.radians(b['heading_deg']-a['heading_deg'])))) for b in approaches if b is not a and 30<delta_heading(a['heading_deg'],b['heading_deg'])<150]
            radius=min(28,max(cross or [a['carriageway_width_m']]))+cfg['policy']['crosswalk_margin_m']
            a['reference_station_m']=-min(radius,length(a['centerline_local'])*.55);a['reference_kind']='estimated_junction_envelope';a['reference_provenance']='crossing road width priors; NOT an observed stopline'
    cfg['coordinate_note']='Stations are arc lengths relative to each incoming road endpoint, not distance to a traffic control or necessarily the node center.'
    return cfg


def request_at(profile,approach,station,role):
    p=profile['policy'];total=length(approach['centerline_local']);station=max(-total+.5,min(station,approach['interior_limit_m'] if approach['allow_interior'] else -.5))
    point,u=path_pose(approach,station);side=1 if p['driving_side']=='right' else -1
    offset=side*approach['carriageway_width_m']/2 if approach['centerline_reference']=='shared_road_axis' else 0.
    # Shared-axis width describes one directional carriageway; half-width locates its nominal middle.
    xy=[point[0]+offset*u[1],point[1]-offset*u[0]];ll=list(local_lonlat(*xy,profile['origin_lonlat']))
    return {'approach_id':approach['id'],'direction':approach['label'],'target_section':approach['section_id'],'role':role,
            'requested_station_m':round(station,3),'requested_lateral_m':offset,'requested_lonlat':ll,'requested_local_xy':xy,
            'nominal_heading_deg':heading(u),'radius_m':p['metadata_radius_m'],'reference_kind':approach['reference_kind'],
            'id':'request_'+safe_id(approach['id'])+'_'+digest([ll,role])[:12]}


def initial_requests(profile,approach):
    p=profile['policy'];scale=approach['carriageway_width_m'];reference=approach['reference_station_m']
    stations=[reference-p['upstream_reference_fraction']*scale,reference+p['near_reference_fraction']*scale,
              p['interior_fraction']*scale if approach['allow_interior'] else reference*.35]
    return [request_at(profile,approach,s,r) for s,r in zip(stations,['upstream_context','near_reference','junction_context'])]


def assess_panorama(profile,approach,request,metadata):
    p=profile['policy']
    if metadata.get('status')!='OK':return {'usable':False,'reasons':[metadata.get('status','metadata_error')]}
    xy=local_xy(*metadata['actual_lonlat'],profile['origin_lonlat']);proj=project_to_path(approach,xy);snap=math.dist(xy,request['requested_local_xy']);reasons=[]
    if snap>p['max_snap_distance_m']:reasons.append('excessive_snap_distance')
    if abs(proj['station_m']-request['requested_station_m'])>p['max_station_error_m']:reasons.append('wrong_longitudinal_position')
    if proj['distance_m']>max(6,approach['carriageway_width_m']*.75):reasons.append('outside_nominal_road_corridor')
    side=1 if p['driving_side']=='right' else -1
    if approach['centerline_reference']=='shared_road_axis' and proj['station_m']<approach['reference_station_m']*.5 and side*proj['lateral_m']<-1.5:reasons.append('opposing_carriageway_snap')
    window=p['preferred_date_window'];date=metadata.get('capture_date')
    relation='capture_only' if not window else 'unknown_date' if not date else 'historical' if date<window[0] else 'later' if date>window[1] else 'target_window'
    return {'usable':not reasons,'reasons':reasons,'actual_station_m':proj['station_m'],'right_of_axis_m':proj['lateral_m'],'request_snap_distance_m':snap,
            'distance_to_center_m':math.hypot(*xy),'capture_relation':relation,'camera_position_error_m':None,
            'carriageway_status':'nominal_approach_corridor_not_surveyed','pose_score':round(snap+(10 if relation in ('historical','later') else 5 if relation=='unknown_date' else 0),3)}


def from_network(node_csv,link_csv,node_id,policy=None,labels=None,stoplines=None):
    node_id=str(node_id);labels=labels or {};p=policy_with(policy)
    with Path(node_csv).open(encoding='utf-8-sig',newline='') as f:nodes={r['node_id']:r for r in csv.DictReader(f)}
    require(node_id in nodes,'Network node not found');node=nodes[node_id];origin=[float(node['x_coord']),float(node['y_coord'])]
    with Path(link_csv).open(encoding='utf-8-sig',newline='') as f:links=list(csv.DictReader(f))
    incoming=[r for r in links if r['to_node_id']==node_id and ('auto' in r.get('allowed_uses','auto') or 'motor' in r.get('allowed_uses',''))]
    outgoing=[r for r in links if r['from_node_id']==node_id and 'auto' in r.get('allowed_uses','auto')]
    def coords(r):
        value=r['geometry'].strip();require(re.match(r'^LINESTRING\s*\(',value,re.I),'Only WGS84 LINESTRING geometry supported')
        pts=[[float(v) for v in pair.split()[:2]] for pair in value[value.index('(')+1:value.rindex(')')].split(',')]
        start=nodes.get(r['from_node_id'])
        if start:
            ll=[float(start['x_coord']),float(start['y_coord'])]
            if math.dist(pts[-1],ll)<math.dist(pts[0],ll):pts.reverse()
        return pts
    approaches=[]
    for r in incoming:
        points=coords(r);line=[local_xy(*q,origin) for q in points];h=heading(terminal_unit(line,p));continuation=False
        for out in outgoing:
            if out['to_node_id']==r['from_node_id']:continue
            ol=[local_xy(*q,origin) for q in coords(out)]
            if len(ol)<2:continue
            look=min(length(ol),20);u=unit_between(ol[0],point_at(ol,-length(ol)+look))
            if delta_heading(h,heading(u))<35:continuation=True
        raw_width=r.get('width') or r.get('width_m');lanes=r.get('lanes')
        if raw_width:
            width=float(raw_width);source='map_width_attribute_prior'
        elif lanes and lanes.replace('.','',1).isdigit():
            width=max(3,min(25,float(lanes)*p['lane_width_prior_m']));source='map_lane_count_times_width_prior_NOT_lane_evidence'
        else:width=p['default_carriageway_width_m'];source='policy_fallback_not_observed_width'
        label=labels.get(r['link_id'],{});label={'label':label} if isinstance(label,str) else label
        aid='link_'+r['link_id'];a={'id':aid,'label':label.get('label',aid),'section_id':label.get('section_id','in_'+aid),'region_ids':label.get('region_ids',[]),
              'centerline_lonlat':points,'centerline_reference':'shared_road_axis' if shares_axis(r,links) else 'directed_carriageway',
              'carriageway_width_m':width,'width_source':source,'allow_interior':continuation,'road_name':r.get('name',''),'in_link_id':r['link_id']}
        if stoplines and r['link_id'] in stoplines:a['stopline']=stoplines[r['link_id']]
        approaches.append(a)
    return {'schema':'control-site-1','site_id':node_id,'name':node.get('name') or 'Network node '+node_id,'origin_lonlat':origin,'policy':p,'approaches':approaches,
            'geometry_provenance':{'node_csv':str(Path(node_csv).resolve()),'link_csv':str(Path(link_csv).resolve()),'lane_attributes_role':'sampling_width_prior_only'}}
