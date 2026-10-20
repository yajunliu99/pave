"""Coordinate transforms with explicit viewport and pose provenance.

Image coordinates are continuous top-left-origin pixel coordinates. Geographic
positions use WGS84 degrees; projected units are EPSG:3857 meters. Local meters
use a tangent-scale approximation at the specified origin, valid for this small
intersection footprint. A reversible transform does not prove absolute accuracy.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np

R=6378137.0
WORLD=2*math.pi*R


def mercator(lon,lat):
    if not (-180<=lon<=180 and abs(lat)<85.051129): raise ValueError("Outside Web Mercator domain")
    return R*math.radians(lon), R*math.log(math.tan(math.pi/4+math.radians(lat)/2))


def lonlat(x,y):
    return math.degrees(x/R),math.degrees(2*math.atan(math.exp(y/R))-math.pi/2)


def local_xy(lon,lat,origin):
    x,y=mercator(lon,lat);a,b=mercator(*origin);scale=math.cos(math.radians(origin[1]))
    return (x-a)*scale,(y-b)*scale


def local_lonlat(east,north,origin):
    a,b=mercator(*origin);scale=math.cos(math.radians(origin[1]))
    return lonlat(a+east/scale,b+north/scale)


@dataclass
class Viewport:
    width:int
    height:int
    mercator_bounds:tuple
    provenance:str
    absolute_error_m:float|None=None

    @classmethod
    def fit_bbox(cls,bbox,width,height,padding=(0,0,0,0),provenance="metadata_bbox_fit_assumption"):
        a,b,c,d=bbox;left,bottom=mercator(a,b);right,top=mercator(c,d)
        pt,pr,pb,pl=padding
        if width<=pl+pr or height<=pt+pb:raise ValueError("Padding consumes viewport")
        mpp=max((right-left)/(width-pl-pr),(top-bottom)/(height-pt-pb))
        cx=(left+right)/2;cy=(bottom+top)/2
        x0=cx-(pl+(width-pl-pr)/2)*mpp
        y1=cy+(pt+(height-pt-pb)/2)*mpp
        return cls(width,height,(x0,y1-height*mpp,x0+width*mpp,y1),provenance)

    def to_pixel(self,lon,lat):
        x,y=mercator(lon,lat);a,b,c,d=self.mercator_bounds
        return (x-a)/(c-a)*self.width,(d-y)/(d-b)*self.height

    def to_lonlat(self,x,y):
        a,b,c,d=self.mercator_bounds
        return lonlat(a+x/self.width*(c-a),d-y/self.height*(d-b))

    def ground_mpp(self,latitude):
        a,b,c,d=self.mercator_bounds
        return (c-a)/self.width*math.cos(math.radians(latitude))

    def serialize(self):return asdict(self)


def affine_points(points,matrix):
    p=np.asarray(points,dtype=float).reshape(-1,2);m=np.asarray(matrix,dtype=float)
    return p@m[:,:2].T+m[:,2]


def inverse_affine(matrix):
    m=np.vstack([np.asarray(matrix,dtype=float),[0,0,1]])
    if abs(np.linalg.det(m))<1e-12:raise ValueError("Singular image transform")
    return np.linalg.inv(m)[:2]


def compass_from_projection(projection_heading,pano_yaw):
    return (projection_heading-pano_yaw)%360


def pixel_bearing(x,y,width,height,hfov,pitch,heading):
    """Horizontal compass bearing of a native-image ray; no depth assumed."""
    f=.5*width/math.tan(math.radians(hfov)/2)
    cx=x-width/2;cy=y-height/2;t=math.radians(pitch)
    z=cy*math.sin(t)+f*math.cos(t)
    return (heading+math.degrees(math.atan2(cx,z)))%360


def bearing_interval(bbox,size,hfov,pitch,heading):
    x1,y1,x2,y2=bbox;w,h=size
    values=[pixel_bearing(x*w,y*h,w,h,hfov,pitch,heading) for x,y in ((x1,y1),(x1,y2),(x2,y1),(x2,y2))]
    offsets=[(a-heading+180)%360-180 for a in values]
    return heading+min(offsets),heading+max(offsets)


def sector_polygon(center,start_deg,end_deg,radius,steps=20):
    x,y=center
    return [[x,y]]+[[x+radius*math.sin(math.radians(start_deg+(end_deg-start_deg)*i/steps)),
                       y+radius*math.cos(math.radians(start_deg+(end_deg-start_deg)*i/steps))] for i in range(steps+1)]


def ground_intersection(pixel,pose):
    """Optional flat-ground ray intersection: height must be explicitly calibrated."""
    if pose.get("calibrated_height_m") is None:raise ValueError("Calibrated camera height is required; GPS alone is insufficient")
    w,h=pose["image_size"];x,y=pixel;t=math.radians(pose["pitch_deg"])
    f=.5*w/math.tan(math.radians(pose["hfov_deg"])/2);cx=x-w/2;cy=y-h/2
    down=cy*math.cos(t)-f*math.sin(t);forward=cy*math.sin(t)+f*math.cos(t)
    if down<=0:return None
    length=pose["calibrated_height_m"]/down;angle=math.radians(pose["compass_heading_deg"])
    return (pose["local_xy"][0]+length*(cx*math.cos(angle)+forward*math.sin(angle)),
            pose["local_xy"][1]+length*(-cx*math.sin(angle)+forward*math.cos(angle)))
