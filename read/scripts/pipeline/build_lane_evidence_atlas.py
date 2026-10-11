"""Build P0/P1 and a minimal P2/P4 lane evidence atlas from cached inputs."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.fusion.bundle import build_bundle
from movement_fixer.hybrid.common import write_json


def viewport_lines(g):
    if g.get("relative_checks"):
        checks=g["relative_checks"];names=[k for k in checks if "_to_" in k]
        return ["## 视口问题","",f"主图请求范围宽约 {g['primary_requested_bbox_width_m']:.2f} m，但根据与更大范围请求的图像配准，实际图像宽约 {g['primary_effective_width_m']:.2f} m。",
            "；".join(f"{k.replace('_to_','→')} 像素尺度比 {checks[k]['scale']:.6f}" for k in names)+"。",
            "请求 bbox 和图片的实际视口不同。坐标恢复以最大范围请求的四角拟合为地理锚，使用跨图像配准传递到候选几何源图。该方法的绝对定位还需要独立控制点核验。",""]
    return ["## 视口","",f"卫星图按中心点和缩放级别请求，视口由瓦片公式直接确定，宽约 {g['primary_effective_width_m']:.2f} m，无需配准。影像本身的绝对定位仍未用独立控制点核验。",""]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--downstream-views",type=Path)
    parser.add_argument("--downstream-audit",type=Path)
    parser.add_argument("--legacy-gsv-manifest",type=Path,help="tile-client _manifest.json, for street views that do not record their own pose")
    parser.add_argument("--legacy-pano-yaw",type=Path,help="pano_yaw.json from the same tile client")
    args=parser.parse_args()
    legacy={"manifest":args.legacy_gsv_manifest,"pano_yaw":args.legacy_pano_yaw}
    data,freeze=build_bundle(ROOT,args.baseline,args.output,args.downstream_views,args.downstream_audit,legacy)
    g=data["georeference"]
    points=g.get("primary_request_bbox")
    if points:
        from movement_fixer.fusion.coordinates import Viewport
        atlas=Viewport(**g["atlas_viewport"])
        g["request_footprint_atlas_px"]=[list(atlas.to_pixel(*p)) for p in ((points[0],points[1]),(points[2],points[1]),(points[2],points[3]),(points[0],points[3]))]
    write_json(args.output/"atlas_data.json",data)
    rendered=[v for v in data["views"] if not v.get("in_baseline") and v.get("sampling_domain")!="outbound"]
    window=data.get("target_window")
    lines=["# 拍摄点 mapping 与车道证据关联","",
        "已完成：冻结当前基线、恢复卫星视口、落图实际全景拍摄点与取景方向、建立观测和来源关系、生成可点击证据视图。",
        "尚未证实：绝对地理定位达到车道级精度、GSV 对象的精确地面位置、观测—车道关联正确性。", "",
        *viewport_lines(g),
        "## 当前交付","",f"- {len(data['views'])} 张视图、{len(set(v['pano_id'] for v in data['views']))} 个全景拍摄点。",
        *(["- 补渲染近路口反向视图 "+"、".join(v["id"] for v in rendered)+"，供对应出口检查；未据此重新改写 VLM 结果。"] if rendered else []),
        f"- {len(data['regions'])} 个区域登记记录：{len(data['lanes'])} 个模型支持机动车区域、{len(data['facilities'])} 个其他交通设施、{len(data['rejected_regions'])} 个已否定区域。",
        f"- {len(data['observations'])} 条观测、{len(data['associations'])} 条模型关联。",
        "- 所有 GSV 标记目前只有视线方向和候选重叠区域，没有伪造地面坐标。",
        "- 同一全景的多个裁剪归为一个原始来源组，模型分数不等于校准概率。",
        "- 图对象支持带证据记录的新增、修订、拆分、合并接口；本轮没有自动修改候选边界。", "",
        "## 文件","","- atlas_data.json：交互核对界面的数据；用 dashboard/build_standalone_dashboard.py 打包为单文件页面。",
        "- baseline_freeze.json / baseline/：基线与来源快照。",
        "- georeference.json：卫星视口及其依据。",
        "- sources.json / coverage_matrix.json：真实坐标、朝向、日期和各断面覆盖。",
        "- lane_graph.json：候选车道段、观测、关联假设和冲突保留。",
        "- atlas.geojson：地理数据导出。",
        "- ground_registration_status.json：精细地面配准尚未启用的原因。", "",
        (f"目标影像窗口为 {window}（最新一期街景及其前 6 个月内的影像）；更早的图像标为历史，卫星日期未知。" if window else "街景没有拍摄日期，未设目标影像窗口。")
        +"观测时间与道路状态有效期分开；不宣称恢复了当前合法路网。",
        "坐标公式参考 [Mapbox Static Images 官方说明](https://docs.mapbox.com/api/maps/static-images/)。"]
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if data.get("downstream_acquisition"):
        a=data["downstream_acquisition"]
        exits=sorted({v["target_section"] for v in data["views"] if v.get("sampling_domain")=="outbound"})
        with (args.output/"report.md").open("a",encoding="utf-8") as f:
            f.write(f"\n## 下游证据与区域状态\n\n{len(exits)} 个出口按约25/50/100 m采样，使用向外及回望视图，共{a['views']}幅。新位置查询成功{a['new_panorama_queries_succeeded']}次；名义出口同向侧视图{a['nominal_outgoing_views']}幅，对向侧辅助视图{a['opposing_auxiliary_views']}幅，共{a['unique_panoramas']}个全景来源。同一全景不重复计独立来源；相机所在侧依据元数据和名义道路轴判断，尚未独立标定。\n\n新出口复核结果与基线分别保留，未用人工答案固定区域类别。非道路及已否定区域从有效车道图移出，默认仅显示车道和交通设施；否定区域可在审计层查看。high是模型对该分类的自报置信，不是车道存在的置信。\n\n当前movement为冻结基线，尚未据本轮出口证据重新生成。\n")
            official=[v for v in data['views'] if v.get('sampling_domain')=='outbound' and v.get('carriageway_status')=='nominal_outgoing_side' and v.get('view_role')=='away']
            if official:
                f.write("\n| 出口 | 档位 | 实际距中心 m | 拍摄日期 | 时间关系 |\n|---|---|---:|---|---|\n")
                for v in official:
                    f.write(f"| {v['target_section']} | {v['target_distance_m']} m | {v['distance_to_center_m']:.1f} | {v['capture_date']} | {v['time_relation']} |\n")
                f.write("\n目标窗口以外的同向侧影像仅作历史参考；目标时期证据和历史影像须按各区域的引用分别检查。原始候选多边形仍未自动调整，表面分类通过不等于其全部几何边界精确。\n")
    print({"views":len(data["views"]),"lanes":len(data["lanes"]),"rejected":len(data["rejected_regions"]),"observations":len(data["observations"]),
        "source_width_m":round(g["primary_effective_width_m"],2),"output":str(args.output)})

if __name__=="__main__":main()
