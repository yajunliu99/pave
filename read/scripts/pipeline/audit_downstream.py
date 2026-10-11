"""Reassess all exit candidate regions using dedicated downstream view packets."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.hybrid.common import read_json,write_json,file_hash,digest,resource_path
from movement_fixer.hybrid.evidence_config import assert_no_answer_fields
from movement_fixer.hybrid.evidence_visuals import sheet,transport_jpeg
from movement_fixer.hybrid.evidence_reasoning import surface_prompt,validate_surface
from movement_fixer.hybrid.inference import StageClient
from movement_fixer.fusion.downstream import camera_context
from movement_fixer.fusion.bundle import target_window
from movement_fixer.hybrid.evidence_reasoning import satellite_date


def window_note(cfg):
    """The run's target period for the prompt: an image outside it is context, not confirmation of the target-period state."""
    window=target_window(cfg)
    if not window:return "No target observation window is set; treat every dated image as describing its own capture date only."
    return (f"The target observation window is {window[0]} through {window[1]}. An image captured outside it is historical or later "
            "context, not direct confirmation of the target-period state.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--views",type=Path,required=True,help="exit-view manifest (downstream-views-1)")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache-only",action="store_true")
    parser.add_argument("--reuse-cache-from",type=Path,action="append",default=[],help="Reuse content-identical VLM stages from an earlier exit audit");args=parser.parse_args()
    cfg=read_json(args.baseline/"inference_config.json");assert_no_answer_fields(cfg)
    manifest=read_json(args.views);output=args.output;output.mkdir(parents=True,exist_ok=True)
    signature=digest({"config":cfg,"view_manifest_hash":file_hash(args.views),"version":"downstream-audit-2-mixed-carriageways"})
    if (output/"input_manifest.json").exists() and read_json(output/"input_manifest.json")["signature"]!=signature:raise ValueError("Inputs changed; choose new output")
    write_json(output/"input_manifest.json",{"signature":signature,"baseline":str(args.baseline),"view_manifest":str(args.views),
        "view_manifest_sha256":file_hash(args.views),"reviewer_answers_used":False})
    write_json(output/"status.json",{"state":"running","started_utc":datetime.now(timezone.utc).isoformat()})
    client=StageClient(ROOT,output,cfg["vlm"],cache_only=args.cache_only,reuse_cache_from=args.reuse_cache_from)
    results=[]
    try:
        for section in cfg["sections"]:
            if section["kind"]!="outbound_receiving":continue
            folder=output/section["id"];folder.mkdir(exist_ok=True)
            raw=Image.open(args.baseline/"surface_audits"/section["id"]/"scene_raw.png").convert("RGB")
            ann=Image.open(args.baseline/"surface_audits"/section["id"]/"scene_candidates.png").convert("RGB")
            scene_id="scene_"+section["id"]
            entries=[(scene_id,"Unmarked satellite; actual outgoing travel UP",raw),(scene_id,"Same satellite with hypotheses; not lane truth",ann)]
            metadata={scene_id:{"role":"satellite_scene",**satellite_date(cfg["sources"][section["source_id"]])}}
            views=[v for v in manifest["views"] if v["target_section"]==section["id"]]
            for v in views:
                path=resource_path(ROOT,v["path"])
                if file_hash(path)!=v["image_sha256"]:raise ValueError("View file changed")
                image=Image.open(path).convert("RGB")
                context=camera_context(v)
                entries.append((v["id"],f"{v['view_role']} / {v['distance_to_center_m']:.1f}m / {v['capture_date']} / {context['side_label']}",image))
                metadata[v["id"]]={k:v[k] for k in ("direction","target_section","view_role","capture_date","pano_id","actual_lonlat","distance_to_center_m","carriageway_status","compass_heading_deg")}
                metadata[v["id"]]["target_carriageway_relation"]=context["target_relation"]
            canvas=sheet(entries,folder/"input.png",cell=(900,780),columns=3)
            path=transport_jpeg(canvas,folder/"input.png")
            prompt=surface_prompt(section,None,metadata)+"""\nThis packet specifically samples the EXIT road segment at near/mid/far positions, with paired views along outgoing travel (away) and back toward the intersection (toward). Cameras can be on EITHER carriageway: follow each image's carriageway_status and target_carriageway_relation, never assume all are opposite-side views. These are nominal metadata poses, not measured lane localization. Verify using the center separation, curb, bicycle strip and building frontage. A target region is not confirmed merely because a vehicle or lane appears somewhere in a view.\n"""+window_note(cfg)+""" Explicitly describe which dates support each conclusion and whether temporal change or uncertain correspondence prevents confirmation. Same panorama / two headings count as one original source; multiple locations are not automatically statistically independent. Check every candidate, including candidates outside the true curb, and retain uncertainty if the exact region cannot be matched. Do not assume prior model decisions are correct.\nobserved_arrows values, if any, must be left, through, right or u_turn (use through for straight).\n"""
            result=client.run("downstream_"+section["id"],path,prompt,lambda v,s=section,keys=set(metadata):validate_surface(v,s,keys))
            result["audit_source"]="downstream_satellite_gsv"
            result["view_ids"]=[v["id"] for v in views]
            results.append(result);write_json(output/"predictions.partial.json",results)
        write_json(output/"predictions.json",results)
        write_json(output/"summary.json",{"state":"complete","sections":len(results),"views":len(manifest["views"]),
            "api_calls":client.calls,"cache_hits":client.hits,"acquisition_mode":manifest["acquisition_mode"],
            "reviewer_answers_used":False,"counts":{s["section_id"]:s["count"] for s in results}})
        write_json(output/"status.json",{"state":"complete","finished_utc":datetime.now(timezone.utc).isoformat()})
        print({"sections":len(results),"api_calls":client.calls,"cache_hits":client.hits})
    except Exception as e:
        write_json(output/"status.json",{"state":"failed","error_type":type(e).__name__})
        raise

if __name__=="__main__":main()
