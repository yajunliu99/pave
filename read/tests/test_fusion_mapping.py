import copy
import math
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from movement_fixer.fusion.coordinates import mercator,lonlat,local_xy,local_lonlat,Viewport,affine_points,inverse_affine,compass_from_projection,pixel_bearing,bearing_interval,ground_intersection
from movement_fixer.fusion.registration import fit_ground_correspondence
from movement_fixer.fusion.graph import LaneGraph,time_relation,overlap_fraction
from movement_fixer.fusion.bundle import scene_to_primary


class CoordinateTests(unittest.TestCase):
    def test_mercator_round_trip(self):
        for p in ((-111.94,33.422),(0,0),(170,80),(-70,-50)):
            self.assertTrue(np.allclose(lonlat(*mercator(*p)),p,atol=1e-10))

    def test_local_round_trip(self):
        origin=(-111.939995,33.421935)
        for p in ((0,0),(100,-100),(-120,130)):
            self.assertTrue(np.allclose(local_xy(*local_lonlat(*p,origin),origin),p,atol=1e-7))

    def test_bbox_fit_preserves_aspect_and_padding(self):
        v=Viewport.fit_bbox((-1,-1,1,1),1200,600,(20,40,30,50))
        a,b=v.to_pixel(-1,1),v.to_pixel(1,-1)
        self.assertGreaterEqual(a[0],50);self.assertGreaterEqual(a[1],20-1e-8)
        self.assertLessEqual(b[0],1160);self.assertLessEqual(b[1],570+1e-8)
        self.assertTrue(np.allclose(v.to_lonlat(*v.to_pixel(.7,.4)),(.7,.4),atol=1e-10))

    def test_retina_changes_density_not_geographic_extent(self):
        a=Viewport.fit_bbox((-111.941,33.421,-111.939,33.423),1280,1280)
        b=Viewport.fit_bbox((-111.941,33.421,-111.939,33.423),2560,2560)
        self.assertTrue(np.allclose(a.mercator_bounds,b.mercator_bounds))
        self.assertTrue(np.allclose(np.asarray(a.to_pixel(-111.94,33.422))*2,b.to_pixel(-111.94,33.422)))

    def test_affine_round_trip(self):
        m=[[.663,-.002,430],[.002,.663,431]];p=[[0,0],[1000,1600],[2559,2559]]
        self.assertTrue(np.allclose(affine_points(affine_points(p,m),inverse_affine(m)),p,atol=1e-10))
        with self.assertRaises(ValueError):inverse_affine([[0,0,1],[0,0,2]])

    def test_panorama_projection_heading_is_not_compass_heading(self):
        self.assertEqual(compass_from_projection(1,271),90)
        self.assertEqual(compass_from_projection(181,271),270)
        self.assertAlmostEqual(pixel_bearing(800,500,1600,1000,95,-12,90),90)

    def test_bbox_bearing_wrap(self):
        a,b=bearing_interval([.4,.4,.6,.6],[1280,1280],60,0,359)
        self.assertLess(a,359);self.assertGreater(b,359);self.assertLess(b-a,20)

    def test_full_image_fov_contains_pitched_corner_rays(self):
        a,b=bearing_interval([0,0,1,1],[1600,1000],95,-12,90)
        for x in (0,800,1600):
            for y in (0,500,1000):
                ray=pixel_bearing(x,y,1600,1000,95,-12,90)
                self.assertLessEqual(a,ray+1e-9);self.assertGreaterEqual(b,ray-1e-9)

    def test_ground_projection_requires_calibrated_height(self):
        pose={"image_size":[1000,1000],"hfov_deg":90,"pitch_deg":-45,"compass_heading_deg":0,"local_xy":[0,0]}
        with self.assertRaises(ValueError):ground_intersection((500,500),pose)
        pose["calibrated_height_m"]=2
        self.assertTrue(np.allclose(ground_intersection((500,500),pose),(0,2)))

    def test_scene_crop_inverse_rotation(self):
        meta={"source_crop_xyxy":[100,200,500,800],"rotation_ccw":90}
        self.assertEqual(scene_to_primary([[0,0],[10,20]],meta),[[499,200],[479,210]])

    def test_homography_checks_points_not_used_in_fit(self):
        fit=np.array([[0,0],[100,0],[100,100],[0,100]],float);target=fit*.2+[20,-10]
        check=np.array([[30,20],[80,70]],float)
        result=fit_ground_correspondence(fit,target,check,check*.2+[20,-10])
        self.assertLess(result["independent_check_residual"]["max_px"],1e-4)
        with self.assertRaises(ValueError):fit_ground_correspondence(fit,target,[],[])


class GraphTests(unittest.TestCase):
    def test_time_is_not_validity(self):
        window=("2025-10","2025-11")
        self.assertEqual(time_relation("2024-06",window),"historical")
        self.assertEqual(time_relation("2025-11",window),"target_window")
        self.assertEqual(time_relation("2026-06",window),"later")
        self.assertEqual(time_relation(None,window),"unknown_date")
        self.assertEqual(time_relation("2020-11",None),"no_target_window")

    def test_overlap_is_geometric_only(self):
        self.assertAlmostEqual(overlap_fraction([[0,0],[10,0],[10,10],[0,10]],[[5,0],[15,0],[15,10],[5,10]]),.5)

    def test_split_preserves_original_and_lineage(self):
        original={"id":"lane_a","geometry":[[0,0],[10,0]]};graph=LaneGraph([original],[{"id":"obs_1"}])
        graph.replace_segments("split",["lane_a"],[{"id":"lane_b"},{"id":"lane_c"}],["obs_1"])
        self.assertNotIn("superseded",original)
        self.assertTrue(graph.segments["lane_a"]["superseded"])
        self.assertEqual(graph.segments["lane_b"]["parents"],["lane_a"])

    def test_no_change_without_evidence_or_unique_ids(self):
        graph=LaneGraph([{"id":"a"}])
        with self.assertRaises(ValueError):graph.replace_segments("reject",["a"],[],[])
        with self.assertRaises(ValueError):graph.replace_segments("revise",["a"],[{"id":"a"}],["o1"])
        self.assertFalse(graph.segments["a"].get("superseded",False))

    def test_unknown_evidence_cannot_authorize_revision(self):
        graph=LaneGraph([{"id":"a"}],[{"id":"known"}])
        with self.assertRaises(ValueError):graph.replace_segments("reject",["a"],[],["invented"])


class SatelliteMetadataTests(unittest.TestCase):
    def test_new_cache_marks_bbox_as_request_not_actual_extent(self):
        from movement_fixer.satellite_client import MapboxSatelliteClient,NodeCoord
        from movement_fixer.hybrid.common import read_json
        with tempfile.TemporaryDirectory() as tmp:
            client=MapboxSatelliteClient("pk.test",Path(tmp))
            node=NodeCoord(1,33.4,-111.9)
            client._save_cache("example",b"test-bytes",(-111.9000011,33.4,-111.89999,33.40001),"test",[node])
            metadata=read_json(Path(tmp)/"example.json")
            self.assertEqual(metadata["rendered_viewport_status"],"unverified")
            self.assertEqual(metadata["request_bbox_rounded"][0],-111.900001)
            self.assertEqual(metadata["metadata_version"],2)


if __name__=="__main__":unittest.main()
