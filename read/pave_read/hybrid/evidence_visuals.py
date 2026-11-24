"""Pixel-preserving context and model-requested detail panels for evidence audits."""
from pathlib import Path
from PIL import Image,ImageDraw
from .geometry import rotate,font
from .common import write_json,require


def scene_views(image,section,output):
    points=[p for r in section["regions"] for p in r["polygon"]]
    margin=160
    box=(max(0,int(min(p[0] for p in points))-margin),max(0,int(min(p[1] for p in points))-margin),
         min(image.width,int(max(p[0] for p in points))+margin),min(image.height,int(max(p[1] for p in points))+margin))
    raw=rotate(image.crop(box),section.get("rotation_ccw",0))
    annotated=image.copy();d=ImageDraw.Draw(annotated)
    for r in section["regions"]:
        pts=[tuple(p) for p in r["polygon"]]
        d.line(pts+[pts[0]],fill="#16d5ec",width=2)
        x,y=r["gate_point"]
        d.text((x+3,y+4),r["id"],font=font(16),fill="white",stroke_width=2,stroke_fill="black")
    annotated=rotate(annotated.crop(box),section.get("rotation_ccw",0))
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    raw.save(folder/"scene_raw.png");annotated.save(folder/"scene_candidates.png")
    meta={"source_id":"scene_"+section["id"],"source_crop_xyxy":box,"rotation_ccw":section.get("rotation_ccw",0),
          "image_size":list(raw.size),"context_margin_px":margin,"pixel_values_preserved":True}
    write_json(folder/"scene_geometry.json",meta)
    return raw,annotated,meta


def sheet(entries,path,cell=(1500,1250),columns=2):
    width,height=cell
    canvas=Image.new("RGB",(width*columns,height*((len(entries)+columns-1)//columns)),"#222831")
    d=ImageDraw.Draw(canvas);manifest=[]
    for i,(key,title,image) in enumerate(entries):
        x=(i%columns)*width;y=(i//columns)*height
        display=image.copy();display.thumbnail((width-20,height-74))
        # Enlarge a small satellite context only for legibility, using nearest pixels.
        if max(image.size)<800:
            scale=min((width-20)/image.width,(height-74)/image.height,2)
            display=image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.NEAREST)
        px=x+(width-display.width)//2;py=y+68
        canvas.paste(display,(px,py));d.text((x+10,y+8),key,font=font(22),fill="white")
        d.text((x+10,y+36),title,font=font(18),fill="#aebdcb")
        manifest.append({"image_id":key,"native_size":list(image.size),"sheet_box_xyxy":[px,py,px+display.width,py+display.height]})
    canvas.save(path);write_json(Path(path).with_suffix(".panels.json"),manifest)
    return canvas


def detail_sheet(requests,sources,context,path):
    entries=[("context","Previous context, reduced overview",context)]
    records=[]
    for i,request in enumerate(requests):
        original=sources[request["source_id"]];a,b,c,d=request["bbox_xyxy"]
        box=(int(a*original.width),int(b*original.height),max(int(c*original.width),int(a*original.width)+1),max(int(d*original.height),int(b*original.height)+1))
        crop=original.crop(box)
        entries.append((f"detail_{i+1}",request["source_id"]+" "+str(request["bbox_xyxy"]),crop))
        records.append({"image_id":f"detail_{i+1}","source_id":request["source_id"],"source_bbox_xyxy":box,
            "requested_bbox_normalized":request["bbox_xyxy"],"pixel_values_preserved":True})
    result=sheet(entries,path,cell=(1200,1000));write_json(Path(path).with_suffix(".crops.json"),records)
    return result


def transport_jpeg(image,png_path):
    """Keep the full PNG, use same-size high-quality JPEG for bounded API payloads."""
    path=Path(png_path).with_suffix(".jpg")
    image.save(path,quality=95,subsampling=0)
    write_json(path.with_suffix(".transport.json"),{"original_png":Path(png_path).name,"transport_image":path.name,
        "size":list(image.size),"quality":95,"subsampling":0,"lossy_transport":True,
        "note":"Only the transport image is JPEG; requested detail crops come from original source pixels."})
    return path
