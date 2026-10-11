from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import ImageDraw
import numpy as np

from .common import digest, file_hash, read_json, write_json, resource_path, require
from .geometry import bbox_union_coverage, font


def run_yolo(config, root, output, images, masks, disabled=False):
    options=config["yolo"]
    views=config.get("context_views",[])+config["gsv_views"]
    if disabled:
        return {v["id"]:{"enabled":False,"regions":{},"limitation":"No detector evidence; not a visibility claim"} for v in views}
    cache_root=Path.home()/".cache/net2cell-vlm/ultralytics-config"
    cache_root.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(cache_root)
    os.environ["YOLO_OFFLINE"]="true"; os.environ["YOLO_AUTOINSTALL"]="false"
    import torch
    import ultralytics
    from ultralytics import YOLO,settings
    settings.update({"sync":False})
    device=options.get("device",0)
    if device=="auto":device=0 if torch.cuda.is_available() else "cpu"
    require(device=="cpu" or torch.cuda.is_available(),"YOLO GPU requested but CUDA unavailable")
    weights=resource_path(root,options["weights"])
    require(file_hash(weights)==options["expected_sha256"],"YOLO checkpoint checksum mismatch")
    folder=Path(output)/"yolo"; folder.mkdir(parents=True,exist_ok=True)
    model=None; results={}; reported=set()
    for view in views:
        sid=view["source_id"]; image=images[sid]
        signature=digest({"image":file_hash(Path(output)/"sources"/(sid+".png")),"model_hash":options["expected_sha256"],
                          "options":options,"ultralytics":ultralytics.__version__})
        path=folder/(sid+"_"+signature+".json")
        if path.exists(): data=read_json(path)
        else:
            if model is None: model=YOLO(str(weights),task="detect")
            start=time.perf_counter()
            result=model.predict(np.array(image)[:,:,::-1].copy(),device=device,imgsz=options.get("imgsz",1280),
                conf=options.get("confidence",.25),quantize=32,rect=False,verbose=False,save=False)[0]
            actual=str(next(model.predictor.model.backend.model.parameters()).device)
            require(actual.startswith("cuda") if device!="cpu" else actual=="cpu","YOLO device mismatch")
            detections=[]
            if result.boxes is not None:
                for box,conf,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                    detections.append({"class":result.names[int(cls)],"confidence":round(conf,5),"bbox_xyxy":box})
            data={"enabled":True,"model":"yolo26n","device":actual,"weights_sha256":options["expected_sha256"],
                  "ultralytics":ultralytics.__version__,"elapsed_s":round(time.perf_counter()-start,3),"detections":detections}
            write_json(path,data)
        selected=[d for d in data["detections"] if d["class"] in options["occluder_classes"]]
        hints={}
        for region in view.get("regions",[]):
            fraction=bbox_union_coverage(masks[view["id"]][region["id"]],selected)
            hints[region["id"]]={"bbox_union_fraction":round(fraction,4),
                "hint":"high_bbox_overlap" if fraction>=options.get("high_bbox_coverage",.25) else "some_bbox_overlap" if fraction>0 else "no_detected_overlap",
                "not_true_occlusion_mask":True,"zero_does_not_prove_clear":True}
        overlay=image.copy(); draw=ImageDraw.Draw(overlay)
        for detection in selected:
            x1,y1,x2,y2=detection["bbox_xyxy"]
            draw.rectangle((x1,y1,x2,y2),outline="#ff7c99",width=3)
            draw.text((x1,max(0,y1-18)),detection["class"],font=font(17),fill="#ff7c99",stroke_width=1,stroke_fill="black")
        camera_role=config["sources"][sid].get("camera_role","unspecified")
        draw.rectangle((8,8,min(image.width-8,820),45),fill="#161a22")
        draw.text((15,14),sid+" | "+camera_role.upper(),font=font(20),fill="white")
        overlay.save(folder/(sid+"_detections.png"))
        results[view["id"]]={**data,"source_id":sid,"camera_role":camera_role,
            "reverse_view":config["sources"][sid].get("reverse_view",False),"regions":hints,
            "limitation":"YOLO rectangles are coarse overlap hints, not lane geometry or calibrated occlusion masks. No satellite YOLO detections are used."}
        if sid not in reported: print(f"YOLO {sid} ({camera_role}): {len(selected)} potential occluders",flush=True)
        reported.add(sid)
    write_json(folder/"summary.json",results)
    return results
