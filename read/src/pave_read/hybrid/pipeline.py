from __future__ import annotations

import csv
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from . import PIPELINE_VERSION
from .common import digest, file_hash, load_config, read_json, require, resource_path, write_json
from .geometry import font, load_source, render_section
from .geometry_checks import audit_geometry
from .sampling import audit_sampling
from .inference import StageClient
from .prompts import lane_prompt, movement_prompt, context_prompt
from .validation import DIRECTIONS, MOTOR_TYPES, validate_lanes, validate_movement_output, validate_context
from .yolo_aux import run_yolo


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def preserve_review_fields(path, rows, keys, fields):
    """Refresh generated evidence while preserving user-entered review columns."""
    if not Path(path).exists(): return rows
    previous={tuple(r.get(k,"") for k in keys):r for r in read_csv(path)}
    for row in rows:
        old=previous.get(tuple(str(row.get(k,"")) for k in keys),{})
        for field in fields:
            if field in old: row[field]=old[field]
    return rows


def bearing(wkt):
    pairs=re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",wkt)
    require(len(pairs)>=2,"Link needs a LINESTRING geometry")
    lon1,lat1=map(float,pairs[0]); lon2,lat2=map(float,pairs[-1])
    return math.degrees(math.atan2((lon2-lon1)*math.cos(math.radians((lat1+lat2)/2)),lat2-lat1))%360


def network_context(config,root):
    from .legs import heading
    links={int(r["link_id"]):r for r in read_csv(resource_path(root,config["network"]["link_csv"]))}
    sections={s["id"]:s for s in config["sections"]}
    context={}
    for direction,leg in config["legs"].items():
        context[direction]={}
        for kind,key,node_key,prefix in (("inbound","in_link_id","to_node_id","in_"),("outbound","out_link_id","from_node_id","out_")):
            if leg.get(key) is None: continue  # a one-way leg, or the stem of a T
            require(leg[key] in links,f"Missing network link {leg[key]}")
            row=links[leg[key]]
            require(int(row[node_key])==config["node_id"],"Link is not incident in specified direction")
            angle=bearing(row["geometry"])
            section=sections.get(prefix+direction,{"id":prefix+direction,"direction":direction})
            expected=heading(section)
            error=abs((angle-expected+180)%360-180)
            # A hand-written leg must match its link. A traced section takes its heading from that same link near the
            # junction, so a difference only means the link bends or starts with a stub: recorded, not fatal.
            require(error<45 or section.get("heading_deg") is not None,"Network link geometry does not match directed leg")
            context[direction][kind]={"link_id":leg[key],"name":row.get("name"),"bearing":round(angle,2),
                "baseline_link_lanes":row.get("lanes"),"note":"Baseline attribute, not visual ground truth",
                **({"end_to_end_bearing_differs_from_section_heading_deg":round(error,1)} if error>=45 else {})}
    return context


def overview(config, image, output):
    annotated=image.copy(); d=ImageDraw.Draw(annotated)
    for section in config["sections"]:
        color=section["color"]
        gates=[]
        for region in section["regions"]:
            pts=[tuple(p) for p in region["polygon"]]
            d.line(pts+[pts[0]],fill=color,width=3)
            x,y=region["gate_point"]; gates.append((x,y))
            # Region label close to gate, away from most upstream pavement arrows.
            d.text((x+2,y+5),region["id"],font=font(14),fill=color,stroke_width=1,stroke_fill="black")
        x=sum(p[0] for p in gates)/len(gates); y=sum(p[1] for p in gates)/len(gates)
        d.text((x-45,y-28),section["id"],font=font(20),fill=color,stroke_width=2,stroke_fill="black")
    annotated.save(Path(output)/"intersection_sections.png")
    annotated.resize((1600,1600),Image.Resampling.LANCZOS).save(Path(output)/"intersection_sections_preview.png")
    return annotated


def draw_movements(config, movements, background, output):
    from .legs import approaches
    sections={s["id"]:s for s in config["sections"]}
    colors={"left":"#ffb74d","through":"#52dcff","right":"#d29aff","u_turn":"#ff7373"}
    centre=[background.width/2,background.height/2]  # the satellite image is centred on the junction
    for direction in [s["direction"] for s in approaches(config["sections"])]:
        im=background.copy(); d=ImageDraw.Draw(im)
        for m in movements:
            if m["from_direction"]!=direction or m["status"]!="candidate": continue
            ins={r["id"]:r for r in sections[m["in_section_id"]]["regions"]}
            outs={r["id"]:r for r in sections[m["out_section_id"]]["regions"]}
            for pair in m["lane_pairs"]:
                a=ins[pair["in_region_id"]]["gate_point"]; b=outs[pair["out_region_id"]]["gate_point"]
                control=[(a[0]+b[0])/2,(a[1]+b[1])/2]
                if m["turn"]!="through": control=centre
                pts=[((1-t)**2*a[0]+2*(1-t)*t*control[0]+t*t*b[0],
                      (1-t)**2*a[1]+2*(1-t)*t*control[1]+t*t*b[1]) for t in [i/40 for i in range(41)]]
                # Dashed lines explicitly denote hypotheses, not surveyed trajectories.
                for j in range(0,39,3): d.line(pts[j:j+2],fill=colors[m["turn"]],width=5)
                angle=math.atan2(pts[-1][1]-pts[-3][1],pts[-1][0]-pts[-3][0])
                tip=pts[-1]
                for da in (-.5,.5): d.line([tip,(tip[0]-16*math.cos(angle+da),tip[1]-16*math.sin(angle+da))],fill=colors[m["turn"]],width=5)
        d.rectangle((20,20,1100,95),fill="#161a22")
        d.text((35,30),f"{direction} candidate movements / dashed hypotheses / REVIEW REQUIRED",font=font(24),fill="white")
        im.save(Path(output)/f"movements_{direction}.png")


def export(config, output, lanes, movements, yolo, network, stage_client, geometry_audit=None, contexts=None):
    output=Path(output)
    lane_rows=[]
    for section in lanes:
        for region in section["regions"]:
            hint=yolo.get(section["section_id"],{}).get("regions",{}).get(region["region_id"],{})
            lane_rows.append({"section_id":section["section_id"],"section_kind":section["kind"],"direction":section["direction"],
                "region_id":region["region_id"],"motor_lane_index":region["motor_lane_index"],"surface_type":region["surface_type"],
                "observed_arrows":"|".join(region["observed_arrows"]),"basis":region["basis"],"visibility":region["visibility"],
                "confidence":region["confidence"],"yolo_bbox_fraction":hint.get("bbox_union_fraction"),"evidence":region["evidence"],
                "width_px":region.get("width_metrics",{}).get("median_width_px"),
                "width_ratio":region.get("width_metrics",{}).get("relative_width"),
                "endpoint_eligible":region.get("endpoint_eligible"),
                "geometry_flags":"|".join(region.get("width_metrics",{}).get("flags",[])),
                "user_annotation":str(region.get("review_annotation",{})),
                "allowed_turns":"|".join(region.get("turn_constraint",{}).get("allowed_turns") or []),
                "turn_constraint_source":region.get("turn_constraint",{}).get("source"),
                "reference_surface_type":"","reference_arrows":"","reviewer":"","review_note":""})
    lane_rows=preserve_review_fields(output/"lane_review.csv",lane_rows,["section_id","region_id"],
                                    ["reference_surface_type","reference_arrows","reviewer","review_note"])
    write_csv(output/"lane_review.csv",lane_rows,list(lane_rows[0]))
    pair_rows=[]; decision_rows=[]
    for i,m in enumerate(movements,1):
        code={"left":"L","through":"T","right":"R","u_turn":"U"}[m["turn"]]
        decision_id=m["from_direction"]+code
        decision_rows.append({"candidate_id":decision_id,"status":m["status"],"from_direction":m["from_direction"],
            "to_direction":m["to_direction"],"turn":m["turn"],"pairs":";".join(f"{p['ib_lane']}->{p['ob_lane']}" for p in m["lane_pairs"]),
            "through_offsets_lane_widths":";".join(str(p.get("alignment",{}).get("lateral_in_lane_widths","")) for p in m["lane_pairs"]),
            "basis":m["basis"],"confidence":m["confidence"],"flags":"|".join(m["validation_flags"]),"reason":m["reason"],
            "assumptions":" | ".join(m.get("assumptions",[])),"review_status":"needs_review","reviewer":"","approved":""})
        for pair in m["lane_pairs"]:
            pair_rows.append({"candidate_id":decision_id,"node_id":config["node_id"],
                "ib_link_id":config["legs"][m["from_direction"]]["in_link_id"],
                "ob_link_id":config["legs"][m["to_direction"]]["out_link_id"],
                "start_ib_lane":pair["ib_lane"],"end_ib_lane":pair["ib_lane"],
                "start_ob_lane":pair["ob_lane"],"end_ob_lane":pair["ob_lane"],"lanes":1,
                "type":{"through":"thru","u_turn":"uturn"}.get(m["turn"],m["turn"]),"mvmt_txt_id":decision_id,
                "in_region_id":pair["in_region_id"],"out_region_id":pair["out_region_id"],
                "basis":m["basis"],"confidence":m["confidence"],"review_status":"needs_review","auto_apply":False,
                "straight_lateral_px":pair.get("alignment",{}).get("lateral_px"),
                "straight_lateral_lane_widths":pair.get("alignment",{}).get("lateral_in_lane_widths"),
                "validation_flags":"|".join(m["validation_flags"]),"evidence_refs":"|".join(m["evidence_refs"])})
    pair_fields=["candidate_id","node_id","ib_link_id","ob_link_id","start_ib_lane","end_ib_lane","start_ob_lane","end_ob_lane","lanes","type","mvmt_txt_id","in_region_id","out_region_id","basis","confidence","review_status","auto_apply","straight_lateral_px","straight_lateral_lane_widths","validation_flags","evidence_refs"]
    write_csv(output/"movement_candidates.csv",pair_rows,pair_fields)
    decision_rows=preserve_review_fields(output/"movement_review.csv",decision_rows,["candidate_id"],["reviewer","approved"])
    write_csv(output/"movement_review.csv",decision_rows,list(decision_rows[0]))
    if not (output/"review_template.json").exists():
        write_json(output/"review_template.json",{"reviewer":None,"section_counts":{s["section_id"]:None for s in lanes},
            "movement_decisions":{r["candidate_id"]:{"approved":False,"comment":""} for r in decision_rows},
            "warning":"Template fields are not reference truth until independently reviewed"})
    summary={"pipeline_version":PIPELINE_VERSION,"pilot_id":config["pilot_id"],"lane_sections":lanes,
             "movement_decisions":movements,"candidate_movements":sum(m["status"]=="candidate" for m in movements),
             "lane_pair_rows":len(pair_rows),"unresolved_movements":sum(m["status"]=="unresolved" for m in movements),
             "yolo_enabled":any(v.get("enabled") for v in yolo.values()),
             "forward_contexts":contexts or [],"geometry_checks":geometry_audit,
             "gsv_sampling":audit_sampling(config),
             "yolo_unique_sources":len({v.get("source_id") for v in yolo.values() if v.get("enabled")}),
             "actual_model":{"provider":stage_client.provider,"model":stage_client.model},
             "vlm_calls_this_run":stage_client.calls,"vlm_cache_hits_this_run":stage_client.hits,
             "human_reference_status":config["reference"]["status"],"network_modified":False,
             "far_upstream":config["far_upstream"],"network_context":network}
    write_json(output/"summary.json",summary)
    exclusions={s["id"]:s["excluded_regions"] for s in config["sections"] if s.get("excluded_regions")}
    write_json(output/"excluded_regions.json",exclusions)
    lines=["# 路口混合识别 pipeline 运行报告","",
           "本次流程包括：进口停止线前区域、下游接收区域、宽度与直行对齐检查、正向 GSV 主视图、反向 GSV 区域辅助、YOLO 疑似遮挡提示，以及 VLM 候选 movement 推断。所有连接仍需审核，未写回原路网。","",
           "## 车道断面","","下表为模型在所提供几何区域中的分类统计。区域边界为辅助描绘，卫星日期未知；不是已核实的现状车道总数。","",
           "| 行驶方向 | 进口机动车候选数 | 进口未分类区域 | 出口机动车候选数 | 出口未分类区域 |","|---|---:|---:|---:|---:|"]
    section_map={s["section_id"]:s for s in lanes}
    for direction in DIRECTIONS:
        a,b=section_map[f"in_{direction}"]["count"],section_map[f"out_{direction}"]["count"]
        lines.append(f"| {direction} | {a['model_motor_count']} | {a['unclassified_regions']} | {b['model_motor_count']} | {b['unclassified_regions']} |")
    lines += ["","远上游断面与进口停止线断面分开：本试点没有另行标注更远上游，不把 link 的 lanes 属性或停止线数当成远上游真值。","",
              "## VLM movement 候选","","源车道与接收车道均按各自行驶方向从左到右编号，自行车道等非机动车区域已排除。表中箭头代表编号对应，不是已验证的行车路径。","",
              "| 候选 | 状态 | 进口→出口车道编号 | 依据 | 置信度 | 校验提醒 |","|---|---|---|---|---|---|"]
    for r in decision_rows: lines.append(f"| {r['candidate_id']} | {r['status']} | {r['pairs'] or '—'} | {r['basis']} | {r['confidence']} | {r['flags'] or '—'} |")
    lines += ["","## 车道存在与用途约束","",
        "先检查车道存在及道路包络，再检查用户确认的用途或可见路面箭头，最后检查几何宽度和连接对齐。宽度正常不等于车道存在，直行对齐不能覆盖右转专用等限制。","",
        "| 断面 | 已移除的旧区域 ID |","|---|---|"]
    for sid,regions in exclusions.items(): lines.append(f"| {sid} | "+", ".join(r["id"] for r in regions)+" |")
    lines += ["","移除区域仅保留在 excluded_regions.json 中追溯，不再绘制成车道或作为连接端点。区域 ID 与机动车道编号不同。","",
        "| 断面/区域 | 允许转向 | 约束来源 |","|---|---|---|"]
    for section in lanes:
        if section["kind"]!="inbound_stopbar": continue
        for r in section["regions"]:
            rule=r.get("turn_constraint",{})
            if rule.get("allowed_turns"):
                lines.append(f"| {section['section_id']}:{r['region_id']} | "+", ".join(rule["allowed_turns"])+f" | {rule['source']} |")
    lines += ["","用户确认的 EB/WB 外侧右转专用道与模型直接观测箭头分开保存；不伪造 observed_arrows。无法确定适用车道的 GSV 标志仍保留不确定性，不能用一条假设抵消可能的专用转向冲突。"]
    sampling=summary["gsv_sampling"]
    if sampling.get("checked"):
        position_order=sampling["position_order"]
        lines += ["","## GSV 多档位置覆盖","",
            "每个方向使用全部已配置距离档位的正向视图。实际距离由全景坐标计算，参照路口中心；各档必须来自不同拍摄点。近档采用 95° FOV、-12° pitch，中/远档采用 60° FOV、0° pitch，各方向在同档使用相同取景参数。","",
            "| 方向 | "+" | ".join(p+" 实际距离 / 日期" for p in position_order)+" | 日期复核 |",
            "|---|"+"---|"*(len(position_order)+1)]
        for direction in DIRECTIONS:
            selected=[next(r for r in sampling["positions"] if r["direction"]==direction and r["position"]==p) for p in position_order]
            cells=[f"{r['actual_distance_m']} m / {r['capture_date'] or '未知'}" for r in selected]
            warning="跨日期，需复核" if sampling["date_comparisons"][direction]["review_required"] else "同一标注月份"
            lines.append(f"| {direction} | "+" | ".join(cells)+f" | {warning} |")
        lines += ["",f"{len(sampling['positions'])} 个位置均参与 YOLO 检测和 VLM 场景分析；同一全景的反向补充视图不增加独立拍摄点数量。远/中段与停止线间可能出现增道或转向专用道，不能直接等同车道编号和数量。",
            "跨日期图像只提供带时间限定的证据；不能将历史布局当作较新图像的确认，也不能把时间上的变化直接解释成沿道路的变化。"]
        for direction in DIRECTIONS:
            notes=next((m.get("gsv_alignment_notes",[]) for m in movements if m["from_direction"]==direction),[])
            if notes:
                lines += ["",f"### {direction} 多档证据对应",""]+["- "+n for n in notes]
    lines += ["","## 几何约束","",
              "车道宽度与同一卫星图内清晰单车道的像素宽度比较；不假定米/像素，也不将透视街景宽度直接套入该检查。可能合并多条车道或过窄的机动车区域不能直接作为单车道端点。自行车道、缓冲区等窄带单独分类。",
              "直行候选按道路纵向/横向投影检查；EB 的横向差即原始北朝上图像中的竖直偏差。优先小偏移组合，大幅偏移被标记或拒绝。左/右转曲线不受直行条件约束。阈值为可配置的试点检查值，非道路设计标准。",
              "用户指出的 EB 左转车道作为单独的 review_annotation 保留。外部修订方向不自动变成模型观测到的箭头。","",
              "| 直行候选 | 横向偏移／参考车道宽度 |","|---|---|"]
    for r in decision_rows:
        if r["turn"]=="through": lines.append(f"| {r['candidate_id']} | {r['through_offsets_lane_widths'] or '未建立'} |")
    lines += ["","VLM 可基于车道连续性和接收车道容量推断未标箭头车道的候选连接；这些记录保留 inferred/mixed 标识。U-turn 没有明确依据时保留 unresolved/not_proposed。","",
              "## YOLO 辅助","","YOLO 在 GSV 源图上检测车辆、行人等潜在遮挡物，计算预测矩形并集与每个区域的交叠比例。它是粗略覆盖提示，不是真实实例遮挡率；没有检测框不证明无遮挡。不使用通用 YOLO 的卫星检测结果。","",
              "| GSV 视图 | 角色 | YOLO 开启 | 潜在高交叠区域 |","|---|---|---|---|"]
    for key,value in yolo.items():
        high=[r for r,h in value.get("regions",{}).items() if h["hint"]=="high_bbox_overlap"]
        lines.append(f"| {key} | {value.get('camera_role','未指定')} | {value.get('enabled')} | {', '.join(high) or '无区域交叠提示'} |")
    lines += ["","正向上下文覆盖四个方向；反向视图保留用于补读上游标记。相同源图只执行/缓存一份 YOLO 检测，不将不同裁剪当成独立证据。正向上下文没有车道区域时，仅提供目标框和场景观察，不伪造车道遮挡率。"]
    lines += ["","## 校验与复核","",
              "- 校验了模型返回身份、区域 ID 完整性、机动车端点、转向和出口方向的一致性。",
              "- 交叉车道顺序、箭头不一致、汇入、低置信度端点、暂定边界和 U-turn 需要额外复核，见 validation_flags。",
              "- 模型数量范围仅将未分类区域计为可变项，并非统计置信区间；低置信度的已有分类同样可能变化。",
              "- lane_review.csv 和 movement_review.csv 保留人工参考与审核字段，不将模型结果自动当成真值。",
              "- movement_candidates.csv 是逐车道对的 GMNS 字段候选预览，不是可直接覆盖的正式 movement.csv。",
              "- 卫星/街景日期和断面对齐仍有不确定性；未宣称自动检测了全部几何或恢复了合法性。","",
              "## 文件","","- intersection_sections.png：进口和出口区域总图。",
              "- sections/：每个断面的原始区域、掩膜和上下文图。",
              "- yolo/：自动检测框、GPU 身份与逐区域粗略交叠提示。",
              "- geometry_checks.json：宽度参照、包络缺口和所有直行端点组合的偏移矩阵。",
              "- excluded_regions.json：用户确认不存在的旧车道区域，供追溯，已排除在活动几何之外。",
              "- gsv_context.json：四个方向全部已配置距离档位的正向主视图观察。",
              "- gsv_sampling.json：各档实际坐标、距离、全景来源、日期差异和覆盖检查。",
              "- gsv_positions_*.png：各方向所有档位的正向 YOLO 视图对照。",
              "- vlm/：内容哈希缓存、完整请求、响应及实际模型身份。",
              "- movements_*.png：各进口的虚线候选连接示意，不是测绘轨迹。",
              "- lanes.json、movements.json、summary.json：结构化结果。",
              "- lane_review.csv、movement_review.csv、review_template.json：待人工确认的参考与审核表。","",
              f"本轮实际模型：{stage_client.provider}/{stage_client.model}。本次 API 请求 {stage_client.calls} 次，复用 {stage_client.hits} 个 VLM 阶段缓存。"]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def run(config_path, root, output, *, prepare_only=False, cache_only=False, disable_yolo=False, reuse_cache_from=()):
    root,output=Path(root).resolve(),Path(output).resolve()
    require(output!=root,"Output must be a dedicated run directory")
    config=load_config(config_path,root)
    if config.get("inference_mode")=="visual_evidence":
        from .evidence_pipeline import run_evidence
        return run_evidence(config_path,root,output,prepare_only=prepare_only,cache_only=cache_only,
            disable_yolo=disable_yolo,reuse_cache_from=reuse_cache_from)
    client=StageClient(root,output,config["vlm"],cache_only,reuse_cache_from)
    input_hashes={}
    for sid,source in config["sources"].items():
        path=resource_path(root,source["path"] if source["kind"]=="image" else source["pano_path"])
        input_hashes[sid]=file_hash(path)
    for kind,path in config["network"].items(): input_hashes[kind]=file_hash(resource_path(root,path))
    manifest={"pipeline_version":PIPELINE_VERSION,"config":config,"input_sha256":input_hashes,
              "model":[client.provider,client.model],"disable_yolo":disable_yolo}
    signature=digest(manifest)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"run_manifest.json").exists():
        old=read_json(output/"run_manifest.json")
        require(old["signature"]==signature,"Run inputs/config changed. Choose a new output directory; old evidence is preserved.")
    else: write_json(output/"run_manifest.json",{"signature":signature,**manifest})
    write_json(output/"status.json",{"state":"running","started_utc":datetime.now(timezone.utc).isoformat()})
    network=network_context(config,root); write_json(output/"network_context.json",network)
    geometry_audit=audit_geometry(config); write_json(output/"geometry_checks.json",geometry_audit)
    sampling=audit_sampling(config); write_json(output/"gsv_sampling.json",sampling)
    images={}; masks={}
    all_sections=config["sections"]+config["gsv_views"]
    for sid,source in config["sources"].items():
        images[sid]=load_source(root,source)
        (output/"sources").mkdir(exist_ok=True)
        images[sid].save(output/"sources"/(sid+".png"))
    for section in all_sections:
        records,section_masks=render_section(images[section["source_id"]],section,output/"sections"/section["id"])
        masks[section["id"]]=section_masks
        write_json(output/"sections"/section["id"]/"geometry.json",records)
    background=overview(config,images["satellite"],output)
    if prepare_only:
        write_json(output/"status.json",{"state":"prepared","sections":len(all_sections)})
        return {"state":"prepared","output":str(output)}
    yolo=run_yolo(config,root,output,images,masks,disable_yolo)
    if sampling.get("checked") and not disable_yolo:
        for direction in DIRECTIONS:
            panel=Image.new("RGB",(800*len(sampling["position_order"]),650),"#161a22"); draw=ImageDraw.Draw(panel)
            for col,position in enumerate(sampling["position_order"]):
                row=next(r for r in sampling["positions"] if r["direction"]==direction and r["position"]==position)
                with Image.open(output/"yolo"/(row["source_id"]+"_detections.png")) as img:
                    view=img.convert("RGB"); view.thumbnail((790,530)); panel.paste(view,(col*800+(800-view.width)//2,105))
                draw.text((col*800+12,14),f"{direction} {position.upper()} | {row['actual_distance_m']} m from center",font=font(23),fill="white")
                warning=" | MIXED CAPTURE DATES" if sampling["date_comparisons"][direction]["review_required"] else ""
                draw.text((col*800+12,51),str(row["capture_date"] or "unknown date")+warning,font=font(21),fill="#ffc870" if warning else "white")
            panel.save(output/f"gsv_positions_{direction}.png")
    lanes=[]
    for section in all_sections:
        source=config["sources"][section["source_id"]]
        hints=yolo.get(section["id"],{}).get("regions",{})
        result=client.run("lanes_"+section["id"],output/"sections"/section["id"]/"region_panels.png",
            lane_prompt(section,source,hints,geometry_audit["sections"].get(section["id"])),
            lambda value,s=section:validate_lanes(value,s,geometry_audit["sections"].get(s["id"])))
        lanes.append(result)
        write_json(output/"lanes.partial.json",lanes)
    write_json(output/"lanes.json",lanes)
    contexts=[]
    for view in config.get("context_views",[]):
        source=config["sources"][view["source_id"]]
        result=client.run("context_"+view["id"],output/"sources"/(view["source_id"]+".png"),
            context_prompt(view,source,yolo.get(view["id"],{})),lambda value,v=view,s=source:validate_context(value,v,s))
        contexts.append(result)
    write_json(output/"gsv_context.json",contexts)
    dates={sid:{"capture_date":s.get("capture_date"),"role":s["role"],"reverse_view":s.get("reverse_view",False),
                "camera_role":s.get("camera_role"),"direction":s.get("direction"),"pano_id":s.get("pano_id"),
                "sampling_position":s.get("sampling_position"),"distance_to_center_m":s.get("distance_to_center_m")} for sid,s in config["sources"].items()}
    movements=[]
    qualities={s["id"]:{r["id"]:r.get("quality","") for r in s["regions"]} for s in all_sections}
    for direction in DIRECTIONS:
        relevant=[s for s in lanes if s["section_id"]==f"in_{direction}" or s["kind"]=="outbound_receiving" or s["section_id"]==f"gsv_{direction}"]
        relevant += [s for s in contexts if s["direction"]==direction]
        notes={s["id"]:s.get("notes","") for s in all_sections if s["id"] in {v["section_id"] for v in relevant}}
        result=client.run("movements_"+direction,output/"intersection_sections.png",
            movement_prompt(direction,relevant,config["legs"],dates,notes,geometry_audit),
            lambda value,d=direction,ss=relevant:validate_movement_output(value,d,ss,qualities,geometry_audit))
        movements.extend(result); write_json(output/"movements.partial.json",movements)
    write_json(output/"movements.json",movements)
    draw_movements(config,movements,background,output)
    summary=export(config,output,lanes,movements,yolo,network,client,geometry_audit,contexts)
    write_json(output/"status.json",{"state":"complete","finished_utc":datetime.now(timezone.utc).isoformat(),
        "candidate_movements":summary["candidate_movements"],"lane_pair_rows":summary["lane_pair_rows"],"network_modified":False})
    return summary
