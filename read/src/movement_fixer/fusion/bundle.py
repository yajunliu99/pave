"""Build a local, traceable camera/observation/lane atlas from a frozen run.

No VLM call, semantic reference file or road lane-count attribute is read here.
Image-space associations remain hypotheses unless independently calibrated.
"""
import copy
import csv
import math
import re
import shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image

from ..hybrid.common import read_json,write_json,file_hash,resource_path,digest
from ..hybrid.geometry import load_source,unrotate_point
from .coordinates import Viewport,affine_points,inverse_affine,local_xy,local_lonlat,compass_from_projection,bearing_interval,sector_polygon
from .registration import register_rasters
from .graph import LaneGraph,stable_id,time_relation,overlap_fraction
from .presence import apply_presence
from .downstream import camera_context,acquisition_summary
from . import SCHEMA_VERSION


def normalized_polygon(box,size):
    x1,y1,x2,y2=box;w,h=size
    return [[x1*w,y1*h],[x2*w,y1*h],[x2*w,y2*h],[x1*w,y2*h]]


def scene_to_primary(points,metadata):
    x1,y1,x2,y2=metadata["source_crop_xyxy"]
    return [[x+x1,y+y1] for x,y in [unrotate_point(p[0],p[1],x2-x1,y2-y1,metadata["rotation_ccw"]) for p in points]]


def copy_asset(path,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or file_hash(path)!=file_hash(destination):shutil.copy2(path,destination)
    return destination


def month_index(date):
    year,month=map(int,str(date)[:7].split("-"));return year*12+month-1


def target_window(cfg,months=6):
    """The newest street-view campaign: forward capture months within `months` of the newest one.
    A config may set target_window explicitly; with no dated view there is no window."""
    if cfg.get("target_window"):return tuple(cfg["target_window"])
    dates=sorted({cfg["sources"][v["source_id"]].get("capture_date") for v in cfg.get("context_views",[])}-{None})
    if not dates:return None
    newest=dates[-1]
    return min(d for d in dates if month_index(newest)-month_index(d)<months)[:7],newest[:7]


def satellite_georeference(root,cfg,origin):
    """Atlas viewport, primary->atlas affine, georeference record, atlas image and primary image.

    A satellite source requested by centre and zoom carries its viewport, and is its own atlas. A source
    without one is georeferenced the legacy way: registered against wider requests of the same centre
    stored beside it (bbox metadata files with buffer_meters), the widest serving as the atlas."""
    source=cfg["sources"]["satellite"];primary=resource_path(root,source["path"]);spec=source.get("georeference")
    with Image.open(primary) as image:size=image.size
    if spec:
        if spec["kind"]!="viewport":raise ValueError("Unknown satellite georeference kind")
        atlas=Viewport(**spec["viewport"]);identity=np.array([[1.,0.,0.],[0.,1.,0.]]);mpp=atlas.ground_mpp(origin[1])
        georef={"atlas_viewport":atlas.serialize(),"primary_to_atlas":identity.tolist(),"primary_effective_ground_mpp":mpp,
            "primary_effective_width_m":mpp*size[0],"primary_footprint_atlas_px":[[0,0],[size[0],0],[size[0],size[1]],[0,size[1]]],
            "basis":"Static image requested by centre and zoom; its Web Mercator extent follows from the tile arithmetic, so no registration is needed.",
            "absolute_accuracy":{"status":"unverified","error_m":None,"reason":"No independent surveyed geographic ground-control points; imagery alignment itself is the provider's."},
            "satellite_capture_date":None,"provider_docs":"https://docs.mapbox.com/api/maps/static-images/"}
        return atlas,identity,georef,primary,primary
    folder=primary.parent;sat={}
    # Match each alias to the hash-keyed cache image before trusting its metadata.
    for path in folder.glob("*.json"):
        meta=read_json(path)
        if not isinstance(meta,dict) or "buffer_meters" not in meta or "bbox" not in meta:continue
        n=int(meta["buffer_meters"]);hashed=path.with_suffix(".jpg");alias=folder/re.sub(r"buf\d+",f"buf{n}",primary.name)
        if file_hash(alias)!=file_hash(hashed):raise ValueError("Satellite alias/metadata image mismatch")
        with Image.open(alias) as image:alias_size=image.size
        requested=[float(f"{x:.6f}") for x in meta["bbox"]]
        sat[n]={"metadata":meta,"metadata_file":str(path),"request_bbox":requested,"image":alias.resolve(),"size":alias_size,
            "image_sha256":file_hash(alias),"viewport":Viewport.fit_bbox(requested,*alias_size)}
    first=next((n for n,s in sat.items() if s["image"]==primary.resolve()),None)
    wider=sorted(n for n in sat if first is not None and n>first)
    if first is None or len(wider)<2:raise ValueError("The satellite source records no viewport, and three cached scales of it are needed for registration")
    mid,top=wider[0],wider[-1];w,h=size
    pair_top=register_rasters(sat[first]["image"],sat[top]["image"])
    pair_mid=register_rasters(sat[first]["image"],sat[mid]["image"])
    pair_mid_top=register_rasters(sat[mid]["image"],sat[top]["image"])
    primary_to_atlas=np.asarray(pair_top["source_to_target"])
    atlas=Viewport.fit_bbox(sat[top]["request_bbox"],*sat[top]["size"],provenance="reconstructed_from_rounded_bbox_request; zero_padding; absolute_accuracy_unverified")
    grid=np.array([[x*w/2560,y*h/2560] for x in (300,900,1500,2100) for y in (300,900,1500,2100)])
    direct=affine_points(grid,primary_to_atlas)
    indirect=affine_points(affine_points(grid,pair_mid["source_to_target"]),pair_mid_top["source_to_target"])
    cycle=np.linalg.norm(direct-indirect,axis=1)
    image_extent=affine_points([[0,0],[w,0],[w,h],[0,h]],primary_to_atlas).tolist()
    ground_mpp=atlas.ground_mpp(origin[1])*pair_top["scale"]
    georef={"atlas_viewport":atlas.serialize(),"primary_to_atlas":primary_to_atlas.tolist(),
        "primary_effective_ground_mpp":ground_mpp,"primary_effective_width_m":ground_mpp*w,
        "primary_requested_bbox_width_m":sat[first]["viewport"].ground_mpp(origin[1])*w,
        "primary_footprint_atlas_px":image_extent,"primary_request_bbox":sat[first]["request_bbox"],
        "relative_checks":{f"{first}_to_{top}":pair_top,f"{first}_to_{mid}":pair_mid,f"{mid}_to_{top}":pair_mid_top,
            "cycle_consistency_median_px":float(np.median(cycle)),"cycle_consistency_max_px":float(np.max(cycle))},
        "absolute_accuracy":{"status":"unverified","error_m":None,"reason":"No independent surveyed geographic ground-control points. Raster correspondence checks validate relative image alignment only."},
        "metadata_warning":f"The buf{first} request bbox is not the rendered viewport; the image scale was measured by registration against buf{top}.",
        "satellite_capture_date":None,"provider_docs":"https://docs.mapbox.com/api/maps/static-images/"}
    return atlas,primary_to_atlas,georef,sat[top]["image"],primary


def build_bundle(root,baseline,output,downstream_manifest=None,downstream_audit=None,legacy_gsv=None):
    """legacy_gsv: {"manifest": tile-client _manifest.json, "pano_yaw": pano_yaw.json} for street-view sources
    that do not record their own camera pose (positions, compass headings) in the config."""
    root=Path(root).resolve();baseline=Path(baseline).resolve();output=Path(output).resolve()
    cfg=read_json(baseline/"inference_config.json");summary=read_json(baseline/"summary.json")
    window=target_window(cfg)
    def relation(date):return time_relation(date,window)
    lanes=read_json(baseline/"lanes.json");initial=read_json(baseline/"surface_predictions.json")
    uses=read_json(baseline/"lane_use_audits.json");contexts=read_json(baseline/"gsv_context.json")
    sampling=read_json(baseline/"gsv_sampling.json")
    downstream=read_json(downstream_manifest) if downstream_manifest else None
    exit_audits=read_json(downstream_audit) if downstream_audit else []
    origin=tuple(reversed(cfg["gsv_sampling"]["center_lat_lon"]))
    output.mkdir(parents=True,exist_ok=True)
    if downstream_manifest:
        copy_asset(Path(downstream_manifest),output/"downstream_evidence/manifest.json")
    if downstream_audit:
        audit_folder=Path(downstream_audit).parent
        if read_json(audit_folder/"status.json").get("state")!="complete":raise ValueError("Downstream audit not complete")
        supplemental_files=[]
        for path in sorted(audit_folder.rglob("*")):
            if path.is_file() and path.suffix in (".json",".txt",".jpg"):
                destination=output/"downstream_evidence/audit"/path.relative_to(audit_folder)
                copy_asset(path,destination);supplemental_files.append({"source_path":str(path.resolve()),"sha256":file_hash(path)})
        write_json(output/"downstream_evidence/provenance.json",{"view_manifest_sha256":file_hash(downstream_manifest),"audit_files":supplemental_files,"reviewer_answers_used":False})
    frozen_names=["inference_config.json","input_manifest.json","summary.json","lanes.json","surface_predictions.json",
        "lane_use_audits.json","gsv_context.json","movements.json","gsv_sampling.json"]
    frozen_names += [str(p.relative_to(baseline)) for p in sorted((baseline/"vlm").rglob("*")) if p.is_file() and p.suffix in (".json",".txt")]
    frozen=[]
    for name in frozen_names:
        path=baseline/name;frozen.append({"path":str(path),"sha256":file_hash(path),"size":path.stat().st_size})
        copy_asset(path,output/"baseline"/name)
    code_files=sorted((root/"src/movement_fixer/hybrid").glob("*.py"))
    frozen_code=[{"path":str(p.relative_to(root)),"sha256":file_hash(p)} for p in code_files]
    for p in code_files:
        destination=output/"baseline_code"/p.relative_to(root/"src")
        if destination.exists() and file_hash(destination)!=file_hash(p):raise ValueError("Baseline code changed; choose a new output directory")
        copy_asset(p,destination)
    freeze={"schema":SCHEMA_VERSION,"baseline":str(baseline),"files":frozen,"code":frozen_code,
        "reviewer_reference_used":False,"target_observation_window":{"start":window[0],"end":window[1]} if window else None,
        "valid_road_state_interval":None,"note":"Image capture dates are not road-rule validity intervals."}
    freeze_path=output/"baseline_freeze.json"
    if freeze_path.exists():
        old=read_json(freeze_path)
        # Compare by path inside the baseline run, so a run folder that was moved still matches its freeze.
        def inside(entry,base):
            try:return str(Path(entry["path"]).relative_to(base))
            except ValueError:return entry["path"]
        new_map={inside(x,baseline):x["sha256"] for x in frozen}
        old_base=Path(old.get("baseline",baseline))
        if any(new_map.get(inside(x,old_base))!=x["sha256"] for x in old["files"]):raise ValueError("Frozen baseline changed; use a new output directory")
        if old["files"]!=frozen:
            old_hash=file_hash(freeze_path);copy_asset(freeze_path,output/("baseline_freeze_"+old_hash[:12]+".json"))
            freeze["previous_manifest_sha256"]=old_hash
        elif old.get("previous_manifest_sha256"):freeze["previous_manifest_sha256"]=old["previous_manifest_sha256"]
    write_json(freeze_path,freeze)

    atlas,primary_to_atlas,georef,atlas_image,primary_image=satellite_georeference(root,cfg,origin)
    write_json(output/"georeference.json",georef)
    atlas_asset="assets/satellite_atlas"+atlas_image.suffix.lower()
    copy_asset(atlas_image,output/atlas_asset)
    copy_asset(primary_image,output/("assets/satellite_primary"+primary_image.suffix.lower()))
    with Image.open(atlas_image) as image:atlas_size=list(image.size)

    def atlas_to_local(points):return [list(local_xy(*atlas.to_lonlat(*p),origin)) for p in points]
    def local_to_atlas(points):return [list(atlas.to_pixel(*local_lonlat(*p,origin))) for p in points]
    legacy_gsv=legacy_gsv or {}
    manifest=read_json(legacy_gsv["manifest"]) if legacy_gsv.get("manifest") else []
    yaw=read_json(legacy_gsv["pano_yaw"]) if legacy_gsv.get("pano_yaw") else {}
    source_defs=copy.deepcopy(cfg["sources"])
    # An approach seen only through panorama cut-outs, with no look-back view, gets one rendered from its near panorama.
    for sid,source in list(source_defs.items()):
        if source.get("role")!="gsv" or source["kind"]!="pano" or source.get("reverse_view") or source.get("sampling_position")!="near":continue
        if any(s.get("role")=="gsv" and s.get("reverse_view") and s.get("direction")==source["direction"] for s in source_defs.values()):continue
        extra=copy.deepcopy(source)
        extra["reverse_view"]=True;extra["camera_role"]="reverse_auxiliary"
        extra["projection"]["heading"]=(extra["projection"]["heading"]+180)%360
        source_defs[sid[:-len("_forward")]+"_reverse" if sid.endswith("_forward") else sid+"_reverse"]=extra
    sources={};views=[]
    for sid,source in source_defs.items():
        if source["role"]!="gsv":continue
        new=sid not in cfg["sources"]
        if "compass_heading_deg" in source:
            # Pose recorded at acquisition: an official Street View image of a known panorama position and heading.
            ll=[source["actual_lon"],source["actual_lat"]];m={"pano_err_m":source.get("request_snap_distance_m")}
        else:
            matches=[m for m in manifest if m["pano_id"]==source["pano_id"]]
            if not matches:raise ValueError(f"{sid} records no camera pose; supply the tile-client manifest and panorama yaw records")
            m=matches[0];ll=[m["actual_lon"],m["actual_lat"]]
        xy=local_xy(*ll,origin)
        source_path=baseline/"sources"/(sid+".png")
        asset=output/"assets"/(sid+".png")
        if new:
            asset.parent.mkdir(exist_ok=True);load_source(root,source).save(asset)
        else:copy_asset(source_path,asset)
        with Image.open(asset) as image:width,height=image.size
        if "compass_heading_deg" in source:
            heading=source["compass_heading_deg"];fov=source["view_settings"]["fov"];pitch=source["view_settings"]["pitch"]
            projected_heading=None
        elif source["kind"]=="pano":
            projection=source["projection"];heading=compass_from_projection(projection["heading"],yaw[source["pano_id"]])
            fov,pitch=projection["fov"],projection["pitch"];projected_heading=projection["heading"]
        else:
            heading=m["heading_deg"];fov=source["view_settings"]["fov"];pitch=source["view_settings"]["pitch"]
            projected_heading=(heading+yaw[source["pano_id"]])%360
        position=list(atlas.to_pixel(*ll));local=list(xy)
        full_bearings=bearing_interval([0,0,1,1],[width,height],fov,pitch,heading)
        sector=local_to_atlas(sector_polygon(xy,*full_bearings,150))
        view={"id":sid,"kind":"gsv","asset":"assets/"+asset.name,"image_size":[width,height],"pano_id":source["pano_id"],
            "source_group":"panorama/"+source["pano_id"],"capture_date":source.get("capture_date"),
            "time_relation":relation(source.get("capture_date")),"valid_time":None,
            "direction":source["direction"],"sampling_position":source.get("sampling_position","near"),
            "camera_role":source["camera_role"],"reverse_view":source.get("reverse_view",False),
            "actual_lonlat":ll,"local_xy":local,"atlas_xy":position,"distance_to_center_m":math.hypot(*xy),
            "distance_method":"local scaled Web Mercator approximation; baseline used 111000 m/degree",
            "baseline_distance_to_center_m":source.get("distance_to_center_m"),
            "compass_heading_deg":heading,"pano_projection_heading_deg":projected_heading,"pano_yaw_offset_deg":yaw.get(source["pano_id"]),
            "hfov_deg":fov,"pitch_deg":pitch,"fov_polygon_atlas":sector,"display_range_m":150,
            "horizontal_bearing_bounds_deg":list(full_bearings),
            "request_snap_distance_m":m.get("pano_err_m"),"camera_position_error_m":None,"calibrated_height_m":None,
            "in_baseline":not new,"analysis_status":"new_view_not_analyzed" if new else "baseline_model_observations",
            "rendered_sha256":file_hash(asset),"source_path":str(resource_path(root,source.get("path") or source["pano_path"])),
            "visibility_note":"Horizontal FOV is possible coverage only; no occlusion, depth or calibrated positional-error guarantee."}
        sources[sid]=view;views.append(view)
    if downstream:
        for v in downstream["views"]:
            path=resource_path(root,v["path"])
            if file_hash(path)!=v["image_sha256"]:raise ValueError("Downstream image hash mismatch")
            asset=copy_asset(path,output/"assets"/path.name);xy=local_xy(*v["actual_lonlat"],origin)
            full=bearing_interval([0,0,1,1],v["image_size"],v["hfov_deg"],v["pitch_deg"],v["compass_heading_deg"])
            view={**v,"kind":"gsv","asset":"assets/"+asset.name,"source_path":str(path),"rendered_sha256":file_hash(asset),
                "source_group":"panorama/"+v["pano_id"],"time_relation":relation(v["capture_date"]),"valid_time":None,
                "local_xy":list(xy),"atlas_xy":list(atlas.to_pixel(*v["actual_lonlat"])),
                "sampling_domain":"outbound","camera_role":camera_context(v)["camera_role"],"reverse_view":v["view_role"]=="toward",
                "pano_projection_heading_deg":(v["compass_heading_deg"]+yaw[v["pano_id"]])%360 if v["pano_id"] in yaw else None,
                "pano_yaw_offset_deg":yaw.get(v["pano_id"]),"calibrated_height_m":None,
                "fov_polygon_atlas":local_to_atlas(sector_polygon(xy,*full,150)),"display_range_m":150,"horizontal_bearing_bounds_deg":list(full),
                "in_baseline":False,"analysis_status":"provided_to_downstream_audit" if exit_audits else "new_view_not_analyzed",
                "visibility_note":camera_context(v)["target_relation"]+" Nominal coverage is not confirmed visibility or a unique lane correspondence."}
            sources[v["id"]]=view;views.append(view)
    scenes={}
    for section in cfg["sections"]:
        sid=section["id"];meta=read_json(baseline/"surface_audits"/sid/"scene_geometry.json")
        original=baseline/"surface_audits"/sid/"scene_raw.png";dest=output/"assets"/("scene_"+sid+".png")
        copy_asset(original,dest)
        scenes["scene_"+sid]=meta
        sources["scene_"+sid]={"id":"scene_"+sid,"kind":"satellite_crop","asset":"assets/"+dest.name,
            "image_size":meta["image_size"],"source_group":"satellite/"+file_hash(primary_image),"capture_date":None,
            "time_relation":"unknown_date","valid_time":None,"rendered_sha256":file_hash(dest),"projection":meta}

    baseline_predictions={s["section_id"]:s for s in lanes}
    predictions={**baseline_predictions,**{s["section_id"]:s for s in exit_audits}};segments=[];lookup={}
    for section in cfg["sections"]:
        pred={r["region_id"]:r for r in predictions[section["id"]]["regions"]}
        for region in section["regions"]:
            sid=f"lane/{cfg['node_id']}/{section['id']}/{region['id']}";p=pred[region["id"]]
            display=affine_points(region["polygon"],primary_to_atlas).tolist();world=atlas_to_local(display)
            item={"id":sid,"section_id":section["id"],"region_id":region["id"],"direction":section["direction"],
                "object_status":"candidate_not_verified_physical_lane",
                "section_kind":section["kind"],"geometry_source":"fixed_candidate_from_v6_baseline","geometry_revision":0,
                "polygon_primary_px":region["polygon"],"polygon_atlas_px":display,"polygon_local_m":world,
                "polygon_lonlat":[list(atlas.to_lonlat(*x)) for x in display],"model_surface_type":p["surface_type"],
                "model_existence":p["existence"],"model_use":p.get("lane_use_prediction",{}).get("use","unknown"),
                "model_confidence":p.get("lane_use_prediction",{}).get("confidence",p.get("confidence")),
                "model_motor_index":p["motor_lane_index"],"initial_surface_audit":p.get("initial_surface_audit"),
                "model_reason":p["evidence"],"model_binding":p.get("lane_use_prediction",{}).get("binding",""),
                "rail_presence":"indicated_by_model" if "rail" in p["surface_type"] else "unestablished",
                "motor_permission":"unverified","valid_time":None,"superseded":False}
            item["prediction_origin"]="downstream_gsv_audit" if section["id"] in {s["section_id"] for s in exit_audits} else "baseline_v6"
            if item["prediction_origin"]=="downstream_gsv_audit":
                old=next(r for r in baseline_predictions[section["id"]]["regions"] if r["region_id"]==region["id"])
                item["previous_model_prediction"]={k:old[k] for k in ("surface_type","existence","evidence")}
            apply_presence(item)
            segments.append(item);lookup[(section["id"],region["id"])]=item
    observations=[];associations=[]
    def add_reference(section_id,region_id,stage,reference,claim):
        source_id=reference["source_id"]
        if source_id not in sources:raise ValueError("Unknown observation image")
        source=sources[source_id];box=reference["bbox_xyxy"]
        oid=stable_id("obs/",[stage,section_id,region_id,reference])
        observation={"id":oid,"source_id":source_id,"source_group":source["source_group"],"stage":stage,
            "pixel_bbox_normalized":box,"finding":reference.get("finding",""),"claim":claim,
            "capture_date":source["capture_date"],"valid_time":None,"time_relation":source["time_relation"],
            "actor":"downstream_vlm" if stage=="downstream_audit" else "baseline_vlm","map_location":None,"location_status":"unlocated","source_accuracy_m":None}
        footprint=None
        if source["kind"]=="satellite_crop":
            polygon=scene_to_primary(normalized_polygon(box,source["image_size"]),scenes[source_id])
            display=affine_points(polygon,primary_to_atlas).tolist();footprint=atlas_to_local(display)
            observation.update(map_location=display,location_status="satellite_pixel_transform",geometry_kind="polygon",
                position_error_note="Relative raster registration checked; absolute geolocation uncertainty unknown.")
        else:
            start,end=bearing_interval(box,source["image_size"],source["hfov_deg"],source["pitch_deg"],source["compass_heading_deg"])
            footprint=sector_polygon(source["local_xy"],start,end,150)
            observation.update(bearing_interval_deg=[start,end],possible_ray_region_atlas=local_to_atlas(footprint),
                location_status="bearing_only_no_depth",position_error_note="No calibrated ground plane or object height; the ray fan is not a measured object footprint.")
        candidates=[]
        if footprint:
            for lane in segments:
                overlap=overlap_fraction(lane["polygon_local_m"],footprint)
                if overlap>.02:candidates.append({"region_id":lane["id"],"graph_role":lane["graph_role"],"nominal_overlap_fraction":round(overlap,4)})
        observation["spatial_candidates"]=candidates
        lane=lookup[(section_id,region_id)]
        association={"id":stable_id("assoc/",[oid,lane["id"]]),"observation_id":oid,"region_id":lane["id"],"lane_id":lane["id"] if lane["is_motor_lane"] else None,
            "relation":"supports_model_interpretation","status":"model_claim_unverified",
            "nominal_spatial_overlap":any(x["region_id"]==lane["id"] for x in candidates),
            "unique_lane_binding":False,"date_conflict":source["time_relation"] in ("historical","later"),
            "note":"A sign's location is distinct from the lane it governs; far-view support requires longitudinal correspondence. Nominal ray overlap never confirms governing scope."}
        observations.append(observation);associations.append(association)
    for section in initial:
        for r in section["regions"]:
            claim={"surface_type":r["surface_type"],"existence":r["existence"],"confidence":r["confidence"]}
            for reference in r.get("visual_refs",[]):add_reference(section["section_id"],r["region_id"],"surface_audit",reference,claim)
    for direction,use in uses.items():
        for r in use["regions"]:
            claim={k:r.get(k) for k in ("surface_type","existence","use","confidence","binding")}
            for reference in r.get("visual_refs",[]):add_reference("in_"+direction,r["region_id"],"joint_usage",reference,claim)
    for section in exit_audits:
        for r in section["regions"]:
            claim={"surface_type":r["surface_type"],"existence":r["existence"],"confidence":r["confidence"]}
            for reference in r.get("visual_refs",[]):add_reference(section["section_id"],r["region_id"],"downstream_audit",reference,claim)
    # Keep context observations even when they have no bbox or lane binding.
    for context in contexts:
        source=sources[context["source_id"]]
        for o in context["observations"]:
            observations.append({"id":stable_id("obs/",[context["section_id"],o]),"source_id":source["id"],
                "source_group":source["source_group"],"stage":"gsv_context","finding":o["evidence"],"claim":o,
                "capture_date":source["capture_date"],"time_relation":source["time_relation"],"valid_time":None,
                "pixel_bbox_normalized":None,"map_location":None,"location_status":"unlocated","spatial_candidates":[]})
    coverage=[]
    obs_by_id={o["id"]:o for o in observations}
    for section in cfg["sections"]:
        section_lanes=[x for x in segments if x["section_id"]==section["id"]]
        for view in views:
            wedge=sector_polygon(view["local_xy"],*view["horizontal_bearing_bounds_deg"],150)
            overlaps=[overlap_fraction(l["polygon_local_m"],wedge) for l in section_lanes]
            linked=[a for a in associations if a["region_id"] in {l["id"] for l in section_lanes} and obs_by_id[a["observation_id"]]["source_id"]==view["id"]]
            possible=max(overlaps,default=0)>.02
            coverage.append({"section_id":section["id"],"view_id":view["id"],"horizontal_fov_possible":possible,
                "max_nominal_lane_overlap":round(max(overlaps,default=0),4),"model_association_count":len(linked),
                "actual_visibility_verified":False,"association_verified":False,
                "status":"model_associated_unverified" if linked else "fov_only" if possible else "not_in_nominal_fov",
                "pose_error_m":None,"date_relation":view["time_relation"],"new_view":not view["in_baseline"]})
    graph=LaneGraph(segments,observations,associations).serialize()
    graph["region_registry"]=graph.pop("segments")
    active_lanes=[r for r in segments if r["is_motor_lane"]]
    facilities=[r for r in segments if r["graph_role"]=="non_motor_facility"]
    rejected=[r for r in segments if r["graph_role"]=="rejected_region"]
    unresolved=[r for r in segments if r["graph_role"]=="unresolved_candidate"]
    graph.update(segments=active_lanes,facilities=facilities,rejected_regions=rejected,unresolved_regions=unresolved)
    graph["schema_version"]=SCHEMA_VERSION
    graph["id_semantics"]="Stable legacy IDs identify candidate regions; graph_role and collection membership define current object type. Rejected regions are not lane segments."
    graph["baseline_movement_hypotheses"]=read_json(baseline/"movements.json")
    graph["movement_generation_status"]="not_rerun_after_downstream_audit" if exit_audits else "baseline_only"
    graph["within_section_adjacency"]=[{"left":lookup[(s["id"],a["id"])]["id"],"right":lookup[(s["id"],b["id"])]["id"],
        "status":"candidate_order_only"} for s in cfg["sections"] for a,b in zip(s["regions"],s["regions"][1:])]
    graph["global_geometry_error_m"]=None
    write_json(output/"lane_graph.json",graph);write_json(output/"sources.json",sources);write_json(output/"coverage_matrix.json",coverage)
    write_json(output/"ground_registration_status.json",{"status":"not_calibrated","reason":"No independent paired ground landmarks or measured camera height supplied. Bearing candidates are available; no GSV bbox is treated as an exact ground position.",
        "implementation":"fit_ground_correspondence supports independent fit/check sets; elevated objects never use ground homography."})
    # Road skeleton only: lane counts and pre-existing movement labels are not imported.
    road_ids={str(x[k]) for x in cfg["legs"].values() for k in ("in_link_id","out_link_id") if x.get(k) is not None}
    roads=[]
    with resource_path(root,cfg["network"]["link_csv"]).open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["link_id"] not in road_ids:continue
            coords=[[float(v) for v in pair.strip().split()[:2]] for pair in re.sub(r"^LINESTRING\s*\(|\)$","",r["geometry"]).split(",")]
            roads.append({"id":r["link_id"],"name":r["name"],"lonlat":coords,"atlas_xy":[list(atlas.to_pixel(*p)) for p in coords],"role":"road_skeleton_prior_not_lane_truth"})
    data={"schema_version":SCHEMA_VERSION,"title":cfg.get("name",f"Node {cfg['node_id']}").replace(" x "," × "),
        "target_window":"—".join(window) if window else None,
        "baseline":baseline.name,"origin_lonlat":origin,"atlas":{"asset":atlas_asset,"size":atlas_size},
        "georeference":georef,"sources":sources,"views":views,"regions":segments,"lanes":active_lanes,"facilities":facilities,
        "rejected_regions":rejected,"unresolved_regions":unresolved,"observations":observations,"associations":associations,
        "downstream_acquisition":None if not downstream else acquisition_summary(downstream),
        "coverage":coverage,"roads":roads,"limitations":["Absolute geographic error has not been independently calibrated","FOV does not imply an unobstructed view","GSV observations are sight-line candidates only, without precise ground projection","Lane classes and uses shown are model output, not human-confirmed"]}
    write_json(output/"atlas_data.json",data)
    features=[]
    for lane in segments:
        coords=lane["polygon_lonlat"]+[lane["polygon_lonlat"][0]]
        features.append({"type":"Feature","id":lane["id"],"geometry":{"type":"Polygon","coordinates":[coords]},
            "properties":{k:lane[k] for k in ("section_id","region_id","model_surface_type","model_existence","model_use","graph_role","movement_endpoint_eligible")}})
    for v in views:features.append({"type":"Feature","id":v["id"],"geometry":{"type":"Point","coordinates":v["actual_lonlat"]},
        "properties":{k:v[k] for k in ("pano_id","capture_date","compass_heading_deg","camera_role","camera_position_error_m")}})
    write_json(output/"atlas.geojson",{"type":"FeatureCollection","features":features})
    write_json(output/"active_lanes.geojson",{"type":"FeatureCollection","features":[f for f in features if f.get("properties",{}).get("movement_endpoint_eligible")]})
    write_json(output/"rejected_regions.geojson",{"type":"FeatureCollection","features":[f for f in features if f.get("properties",{}).get("graph_role")=="rejected_region"]})
    return data,freeze
