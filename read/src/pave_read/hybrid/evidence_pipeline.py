"""Evidence-derived lane existence/use; reviewer references are never read here."""
import copy
import json
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw

from .common import read_json,write_json,load_config,file_hash,resource_path,digest,require,parallel_map
from .evidence_config import assert_no_answer_fields
from .evidence_visuals import scene_views,sheet,detail_sheet,transport_jpeg
from .evidence_reasoning import surface_prompt,usage_prompt,validate_surface,validate_usage,evidence_movement_prompt,fuse_surface_usage,satellite_date
from .geometry import load_source,render_section,font
from .geometry_checks import audit_geometry
from .sampling import audit_sampling
from .inference import StageClient
from .prompts import context_prompt
from .legs import approaches,receiving,describe,opposing_approach
from .validation import MOTOR_TYPES,validate_context,validate_movement_output
from .yolo_aux import run_yolo
from .turn_rules import apply_kerb_turn_default

VERSION="hybrid-pilot-v8-street-view-counts"


def source_meta(source):
    return {k:source.get(k) for k in ("direction","camera_role","reverse_view","sampling_position",
        "distance_to_center_m","capture_date","pano_id")}


def look_back_source(config,section):
    """A reverse view taken on the receiving section's own arm by the approach there, looking away from the junction."""
    approach=opposing_approach(config["sections"],section)
    found=[(s.get("distance_to_center_m") or 0,sid) for sid,s in config["sources"].items()
           if approach and s.get("reverse_view") and s.get("direction")==approach["direction"]]
    return min(found)[1] if found else None


def draw_supported_overview(config,lanes,satellite,path):
    image=satellite.copy();draw=ImageDraw.Draw(image)
    byid={s["section_id"]:s for s in lanes};effective=copy.deepcopy(config)
    for s in effective["sections"]:
        predicted={r["region_id"]:r for r in byid[s["id"]]["regions"]}
        active=[]
        for r in s["regions"]:
            p=predicted[r["id"]]
            if p["existence"]=="rejected": continue
            active.append(r);color=s["color"] if p["existence"]=="supported" else "#e3b65a"
            pts=[tuple(x) for x in r["polygon"]];draw.line(pts+[pts[0]],fill=color,width=3)
            x,y=r["gate_point"];draw.text((x+2,y+4),s["id"]+":"+r["id"],font=font(16),fill=color,stroke_width=1,stroke_fill="black")
        s["regions"]=active
    image.save(path)
    return image,effective


def export_evidence(config,output,lanes,uses,movements,yolo,sampling,client,hashes,detail_count):
    from .pipeline import write_csv
    output=Path(output)
    lane_rows=[];pairs=[]
    for s in lanes:
        for r in s["regions"]:
            u=r.get("lane_use_prediction",{})
            lane_rows.append({"section_id":s["section_id"],"region_id":r["region_id"],"surface_type":r["surface_type"],
                "existence":r["existence"],"motor_lane_index":r["motor_lane_index"],"predicted_use":u.get("use",""),
                "predicted_turns":"|".join(u.get("allowed_turns",[])),"evidence":r["evidence"],"binding":u.get("binding","")})
    for m in movements:
        for p in m["lane_pairs"]:
            pairs.append({"from_direction":m["from_direction"],"turn":m["turn"],"to_direction":m["to_direction"],
                "in_region_id":p["in_region_id"],"out_region_id":p["out_region_id"],"ib_lane":p["ib_lane"],"ob_lane":p["ob_lane"],
                "review_status":"needs_review","auto_apply":False,"flags":"|".join(m["validation_flags"]),"reason":m["reason"]})
    write_csv(output/"lane_predictions.csv",lane_rows,list(lane_rows[0]))
    write_csv(output/"movement_candidates.csv",pairs,list(pairs[0]) if pairs else ["from_direction","turn","to_direction","in_region_id","out_region_id"])
    summary={"pipeline_version":VERSION,"pilot_id":config["pilot_id"],"lane_sections":lanes,"lane_use_audits":uses,
        "movement_decisions":movements,"candidate_movements":sum(m["status"]=="candidate" for m in movements),
        "unresolved_movements":sum(m["status"]=="unresolved" for m in movements),"lane_pair_rows":len(pairs),
        "vlm_calls_this_run":client.calls,"vlm_cache_hits_this_run":client.hits,"detail_audits":detail_count,
        "gsv_sampling":sampling,"yolo_unique_sources":len({v.get("source_id") for v in yolo.values() if v.get("enabled")}),
        "network_modified":False,"actual_model":{"provider":client.provider,"model":client.model},
        "reviewer_answers_used":False,"geometry_scope":"Previously drawn proposal regions; not automatic end-to-end geometry extraction"}
    write_json(output/"summary.json",summary)
    text=["## Material Passport","","- Origin Skill: experiment-agent","- Origin Mode: validate",
        "- Origin Date: "+datetime.now(timezone.utc).date().isoformat(),"- Verification Status: ANALYZED",
        "- Version Label: visual_evidence_v6","","## 基于证据的车道推断","",
        "本轮不读取人工车道用途、删除记录或复核道路包络。恢复修订前的候选区域，全部交给模型判断存在性。以下是模型预测，不是写入配置的参考答案。",
        "输入仍依赖此前辅助绘制的候选几何，因此只能验证给定候选区域条件下的分类和关联，不能证明自动提取车道边界的能力。","",
        "| 断面 | 机动车支持数 | 模型否定区域 | 未确定区域 |","|---|---:|---|---|"]
    for s in lanes:
        text.append(f"| {s['section_id']} | {s['count']['model_motor_count']} | "+", ".join(r["region_id"] for r in s["regions"] if r["existence"]=="rejected")+" | "+", ".join(r["region_id"] for r in s["regions"] if r["existence"]=="uncertain")+" |")
    text += ["","## 模型推断的进口用途","","| 断面/区域 | 用途预测 | 证据绑定 |","|---|---|---|"]
    for s in lanes:
        if not s["section_id"].startswith("in_"):continue
        for r in s["regions"]:
            u=r.get("lane_use_prediction",{})
            text.append(f"| {s['section_id']}:{r['region_id']} | {u.get('use','unknown')} | {u.get('binding','').replace('|','/')} |")
    text += ["","## movement 候选","","| 进口/转向 | 状态 | 车道对 | 说明 |","|---|---|---|---|"]
    for m in movements:
        text.append(f"| {m['from_direction']} {m['turn']} | {m['status']} | "+", ".join(f"{p['ib_lane']}→{p['ob_lane']}" for p in m["lane_pairs"])+f" | {m['reason'].replace('|','/')} |")
    text += ["","## 输入与验证边界","",
        "- inference_config.json / input_manifest.json：实际推断配置和文件哈希，不含人工答案。",
        "- surface_audits/：保留背景的原始局部图、候选区域图和模型存在性判断。",
        "- usage_audits/：三档 GSV 对照、模型自主请求的局部细节及标志/车道绑定。",
        "- lanes.json / lane_use_audits.json：模型输出；不按外部参考强制修改。",
        "- candidate_geometry.png 保留全部待判断区域；model_geometry.png 仅按模型结果显示支持/不确定区域。",
        "- cache replay 只验证缓存下的确定性执行，不等于外部 VLM 新请求可重复产生相同答案。",
        "- 单一路口且开发者已见过反馈；不作泛化准确率或因果改善的统计声明。",
        f"- 本次新增 API 请求 {client.calls}，缓存复用 {client.hits}，细节复核 {detail_count}。"]
    (output/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    return summary


def run_evidence(config_path,root,output,*,prepare_only=False,cache_only=False,disable_yolo=False,reuse_cache_from=()):
    from .pipeline import network_context,overview,draw_movements
    root=Path(root);output=Path(output);config=load_config(config_path,root);assert_no_answer_fields(config)
    output.mkdir(parents=True,exist_ok=True)
    hashes={sid:file_hash(resource_path(root,s.get("path") or s["pano_path"])) for sid,s in config["sources"].items()}
    hashes.update({k:file_hash(resource_path(root,p)) for k,p in config["network"].items()})
    manifest={"version":VERSION,"config":config,"input_sha256":hashes,"disable_yolo":disable_yolo};signature=digest(manifest)
    path=output/"input_manifest.json"
    if path.exists():require(read_json(path)["signature"]==signature,"Inputs changed: choose a new output directory")
    write_json(path,{"signature":signature,**manifest});write_json(output/"inference_config.json",config)
    write_json(output/"status.json",{"state":"running","started_utc":datetime.now(timezone.utc).isoformat()})
    client=StageClient(root,output,config["vlm"],cache_only,reuse_cache_from)
    sampling=audit_sampling(config);write_json(output/"gsv_sampling.json",sampling)
    network_context(config,root)
    geometry=audit_geometry(config);write_json(output/"geometry_hints.json",geometry)
    images={};masks={};scenes={};scene_meta={}
    (output/"sources").mkdir(exist_ok=True)
    for sid,source in config["sources"].items():
        images[sid]=load_source(root,source);images[sid].save(output/"sources"/(sid+".png"))
    for s in config["sections"]+config["gsv_views"]:
        _,masks[s["id"]]=render_section(images[s["source_id"]],s,output/"sections"/s["id"])
    overview(config,images["satellite"],output).save(output/"candidate_geometry.png")
    for s in config["sections"]:
        folder=output/"surface_audits"/s["id"]
        raw,ann,meta=scene_views(images["satellite"],s,folder)
        scenes[s["id"]]=(raw,ann);scene_meta[s["id"]]=meta
    if prepare_only:
        write_json(output/"status.json",{"state":"prepared"});return {"state":"prepared"}
    yolo=run_yolo(config,root,output,images,masks,disable_yolo)
    # Independent requests (one per view, section or approach) may run side by side: NET2CELL_PARALLEL_REQUESTS.
    def context_of(view):
        source=config["sources"][view["source_id"]]
        return client.run("context_"+view["id"],output/"sources"/(view["source_id"]+".png"),
            context_prompt(view,source,yolo[view["id"]]),lambda v,w=view,s=source:validate_context(v,w,s))
    contexts=parallel_map(context_of,config["context_views"])
    write_json(output/"gsv_context.json",contexts)
    def surface_of(s):
        sid=s["id"];raw,ann=scenes[sid];scene_id="scene_"+sid
        entries=[(scene_id,"Unmarked satellite context; travel UP",raw),(scene_id,"SAME scene with hypothesis IDs; travel UP",ann)]
        metadata={scene_id:{"role":"satellite_scene","traffic_direction":s["direction"],**satellite_date(config["sources"][s["source_id"]])}}
        if s["kind"]=="outbound_receiving":
            other=look_back_source(config,s)
            if other in images:
                entries.append((other,"Supplemental opposing-approach reverse view",images[other]));metadata[other]=source_meta(config["sources"][other])
        panel=output/"surface_audits"/sid/"input.png";sheet(entries,panel,cell=(1300,1100))
        mpp=config["sources"][s["source_id"]].get("georeference",{}).get("ground_mpp")
        return client.run("surface_"+sid,panel,surface_prompt(s,geometry["sections"].get(sid),metadata,mpp),
            lambda v,w=s,ids=set(metadata):validate_surface(v,w,ids))
    lanes=parallel_map(surface_of,config["sections"]);write_json(output/"lanes.partial.json",lanes)
    write_json(output/"surface_predictions.json",lanes)
    uses={};detail_count=0
    directions=[s["direction"] for s in approaches(config["sections"])]
    def usage_of(direction):
        inlet=next(s for s in lanes if s["section_id"]=="in_"+direction)
        sid=inlet["section_id"];scene_id="scene_"+sid
        srcs={scene_id:scenes[sid][0]};meta={scene_id:{"role":"satellite_scene","traffic_direction":direction}}
        entries=[(scene_id,"Satellite candidate IDs; traffic UP",scenes[sid][1])]
        relevant=[c for c in contexts if c["direction"]==direction]
        for c in relevant:
            source_id=c["source_id"];source=config["sources"][source_id];srcs[source_id]=images[source_id];meta[source_id]=source_meta(source)
            entries.append((source_id,f"{c['sampling_position']} / {c['distance_to_center_m']} m / {c['capture_date']}",images[source_id]))
        folder=output/"usage_audits"/direction;folder.mkdir(parents=True,exist_ok=True)
        panel=folder/"input.png";canvas=sheet(entries,panel)
        prompt=usage_prompt(direction,inlet,meta,relevant,config.get("classification_review",{}).get(direction))
        validator=lambda v,s=inlet,ids=set(srcs):validate_usage(v,s,ids)
        transport=transport_jpeg(canvas,panel)
        use=client.run("usage_"+direction,transport,prompt,validator);details=0
        if use.get("detail_requests"):
            requests=use["detail_requests"];detail=folder/"details.png"
            detail_canvas=detail_sheet(requests,srcs,canvas,detail)
            follow=prompt+"\nYour preliminary image-based result was: "+json.dumps(use,ensure_ascii=False)
            follow+="\nThe new panel contains the native-pixel crops YOU requested, with original source bbox mappings. Reassess the complete lane-use result. Cite original source coordinates. If still unreadable, keep unknown; no more crops will be taken in this run. Do not treat an enlarged blur as new information.\nCrop mapping: "+json.dumps(requests)
            use=client.run("usage_detail_"+direction,transport_jpeg(detail_canvas,detail),follow,validator);details=1
        return inlet,use,details
    for direction,(inlet,use,details) in zip(directions,parallel_map(usage_of,directions)):
        uses[direction]=use;detail_count+=details
        fuse_surface_usage(inlet,use)
    write_json(output/"lane_use_audits.partial.json",uses)
    write_json(output/"lane_use_audits.json",uses);write_json(output/"lanes.json",lanes)
    background,effective=draw_supported_overview(config,lanes,images["satellite"],output/"model_geometry.png")
    movements=[];byid={s["id"]:s for s in config["sections"]}
    def movements_of(direction):
        relevant=[s for s in lanes if s["section_id"]=="in_"+direction or s["kind"]=="outbound_receiving"]
        context=[c for c in contexts if c["direction"]==direction]
        options=receiving(config["sections"],byid["in_"+direction])
        if not options:  # no receiving section was traced, so there is nothing to pair this approach's lanes with
            print(f"No traced receiving section for {direction}; movements not inferred",flush=True);return []
        return client.run("evidence_movements_"+direction,output/"model_geometry.png",
            evidence_movement_prompt(direction,relevant,uses[direction],context,config["legs"],options,
                                     describe(config["sections"],byid["in_"+direction],options),
                                     [u["id"] for u in config.get("untraced_sections",[]) if u["id"].startswith("out_")]),
            lambda v,d=direction,ss=relevant+context,o=options:validate_movement_output(v,d,ss,enforce_turn_constraints=False,receiving=o))
    for value in parallel_map(movements_of,directions):
        movements.extend(value)
    write_json(output/"movements.partial.json",movements)
    # Traffic-law default for the kerbside lane; the model's own decisions stay on record (turn_rules.py).
    movements,turn_notes=apply_kerb_turn_default(movements,lanes,config["sections"],config.get("driving_side","right"))
    write_json(output/"turn_rule_notes.json",turn_notes)
    write_json(output/"movements.json",movements);draw_movements(effective,movements,background,output)
    summary=export_evidence(config,output,lanes,uses,movements,yolo,sampling,client,hashes,detail_count)
    write_json(output/"status.json",{"state":"complete","finished_utc":datetime.now(timezone.utc).isoformat(),
        "candidate_movements":summary["candidate_movements"],"lane_pair_rows":summary["lane_pair_rows"],"reviewer_answers_used":False})
    return summary
