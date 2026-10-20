"""Relative raster registration and optional ground-plane correspondence."""
import cv2
import numpy as np
from .coordinates import affine_points


def residual_summary(values):
    values=np.asarray(values,float)
    if not len(values):return {"count":0}
    return {"count":len(values),"median_px":float(np.median(values)),"p90_px":float(np.percentile(values,90)),
            "p95_px":float(np.percentile(values,95)),"max_px":float(np.max(values))}


def register_rasters(source_path,target_path,seed=351):
    cv2.setRNGSeed(seed)
    a=cv2.imread(str(source_path),0);b=cv2.imread(str(target_path),0)
    if a is None or b is None:raise ValueError("Registration image missing")
    detector=cv2.SIFT_create(nfeatures=4000)
    ka,da=detector.detectAndCompute(a,None);kb,db=detector.detectAndCompute(b,None)
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[x for x,y in matches if x.distance<.7*y.distance]
    src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good])
    # Spatial holdout: matches in these blocks never participate in the fit.
    hold=np.array([(int(x/320)+2*int(y/320))%5==0 for x,y in src])
    if np.sum(~hold)<12 or np.sum(hold)<12:raise ValueError("Insufficient fit/holdout correspondences")
    matrix,inliers=cv2.estimateAffinePartial2D(src[~hold],dst[~hold],method=cv2.RANSAC,ransacReprojThreshold=2)
    if matrix is None:raise ValueError("Image registration failed")
    residual=np.linalg.norm(affine_points(src,matrix)-dst,axis=1);checked=residual[hold]
    accepted=checked<=2
    return {"source_to_target":matrix.tolist(),"scale":float(np.hypot(matrix[0,0],matrix[1,0])),
        "rotation_deg":float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))),
        "matched_features":len(good),"fit_count":int((~hold).sum()),"fit_inliers":int(inliers.sum()),
        "holdout_all":residual_summary(checked),"holdout_within_2px":residual_summary(checked[accepted]),
        "holdout_fraction_within_2px":float(accepted.mean()),
        "holdout_examples":[{"source_xy":s.tolist(),"target_xy":d.tolist(),"error_px":float(e)} for s,d,e in zip(src[hold][:30],dst[hold][:30],checked[:30])],
        "meaning":"Relative same-imagery registration only; not surveyed absolute geolocation accuracy."}


def fit_ground_correspondence(fit_image,fit_map,check_image,check_map):
    if len(fit_image)<4 or len(check_image)<2:raise ValueError("Need at least four fit and two independent check landmarks")
    a=np.asarray(fit_image,np.float32);b=np.asarray(fit_map,np.float32)
    if len(cv2.convexHull(a))<3 or abs(cv2.contourArea(cv2.convexHull(a)))<1:raise ValueError("Degenerate calibration landmarks")
    H,_=cv2.findHomography(a,b,method=0)
    if H is None:raise ValueError("Ground-plane fit failed")
    checks=np.asarray(check_image,np.float32).reshape(-1,1,2)
    predicted=cv2.perspectiveTransform(checks,H).reshape(-1,2)
    residual=np.linalg.norm(predicted-np.asarray(check_map,float),axis=1)
    return {"homography":H.tolist(),"independent_check_residual":residual_summary(residual),
        "valid_image_hull":cv2.convexHull(a).reshape(-1,2).tolist(),
        "applicability":"Calibrated local ground plane only; not signs, signals, building tops or unrestricted extrapolation."}
