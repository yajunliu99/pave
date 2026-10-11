"""Street-view projection geometry and pose fitting from boundary readings. No model, no network."""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image
from movement_fixer.hybrid.common import ValidationError
from movement_fixer.autoloop.gsv_projection import (boundaries, camera_xyz, choose_rows, crossings, dense, fit_group, fit_pose, left_to_right, validate_check,
                                                    project_polygon, render_for_model, validate_readings, visible_ids)

VIEW = {"id": "v", "local_xy": [0., 0.], "compass_heading_deg": 0., "pitch_deg": 0., "hfov_deg": 60., "image_size": [1280, 1280],
        "capture_date": "2025-11", "direction": "NB"}


def strip(left, right, start=4., end=40.):
    """A northbound strip between two eastings, as a region polygon in local metres."""
    down = [[left, y] for y in np.linspace(start, end, 5)]
    up = [[right, y] for y in np.linspace(end, start, 5)]
    return {"polygon_local_m": down + up}


class ProjectionTests(unittest.TestCase):
    def test_point_straight_ahead(self):
        f = 640 / math.tan(math.radians(30))
        (u, v), = project_polygon([[0, 10], [0, 10.0001], [0.0001, 10]], VIEW)[:1]
        self.assertAlmostEqual(u, 640, places=3)
        self.assertAlmostEqual(v, 640 + f * 2.5 / 10, places=3)

    def test_sideways_shift_moves_the_camera_right(self):
        right, _, _ = camera_xyz([[2., 10.]], VIEW, dx=2.)
        self.assertAlmostEqual(right[0], 0., places=9)  # a point 2 m to the right is now dead ahead
        east = dict(VIEW, compass_heading_deg=90.)
        right, _, forward = camera_xyz([[10., -2.]], east, dx=2.)
        self.assertAlmostEqual(right[0], 0., places=9)  # facing east, the camera's right is south
        self.assertAlmostEqual(forward[0], 10., places=9)

    def test_ground_behind_the_camera_is_clipped(self):
        polygon = project_polygon(strip(-1, 1, -10, 10)["polygon_local_m"], VIEW)
        self.assertTrue(polygon and all(v > 640 for _, v in polygon))

    def test_shared_edges_are_one_boundary(self):
        edges, sides = boundaries([strip(-1.8, 1.8), strip(1.8, 5.4)])
        self.assertEqual((len(edges), sides), (3, [(0, 1), (1, 2)]))


class FitTests(unittest.TestCase):
    def setUp(self):
        edges, _ = boundaries([strip(-1.8, 1.8), strip(1.8, 5.4), strip(5.4, 7.0)])
        self.samples = [dense(e) for e in edges]
        self.rows = choose_rows(self.samples, VIEW)

    def readings(self, pose):
        xs = crossings(self.samples, VIEW, self.rows, pose)
        return {f"b{i + 1}": [None if not np.isfinite(x) else float(x) for x in row] for i, row in enumerate(xs)}

    def test_recovers_a_known_pose(self):
        truth = (1.4, 1.5, 2.7)
        fit = fit_pose(self.samples, VIEW, self.rows, self.readings(truth))
        p = fit["pose"]
        self.assertAlmostEqual(p["lateral_shift_m"], truth[0], delta=.1)
        self.assertAlmostEqual(p["heading_shift_deg"], truth[1], delta=.2)
        self.assertAlmostEqual(p["camera_height_m"], truth[2], delta=.1)
        self.assertLess(fit["rms_px_after"], 3)
        self.assertGreater(fit["rms_px_before"], 50)
        self.assertFalse(fit["at_search_limit"])

    def test_follows_the_readings_across_a_whole_lane(self):
        # Readings place every boundary one lane to the left; the fit must not settle on the nearer repeat.
        fit = fit_pose(self.samples, VIEW, self.rows, self.readings((3.6, 0., 2.5)))
        self.assertAlmostEqual(fit["pose"]["lateral_shift_m"], 3.6, delta=.1)

    def test_noise_and_missing_readings(self):
        readings = self.readings((-.8, -1., 2.4))
        rng = np.random.RandomState(3)
        noisy = {k: [None if x is None else x + rng.normal(0, 4) for x in v] for k, v in readings.items()}
        noisy["b3"] = [None, None, None]
        fit = fit_pose(self.samples, VIEW, self.rows, noisy)
        self.assertAlmostEqual(fit["pose"]["lateral_shift_m"], -.8, delta=.3)
        self.assertLess(fit["rms_px_after"], 10)

    def test_too_few_readings(self):
        with self.assertRaises(ValidationError):
            fit_pose(self.samples, VIEW, self.rows, {"b1": [600., None, None], "b2": [None, None, None]})


class ReadingTests(unittest.TestCase):
    def answer(self, *rows, ids=("b1", "b2")):
        return {"boundaries": [{"id": i, "feature": "lane_line", "xs": list(xs), "evidence": "paint"} for i, xs in zip(ids, rows)]}

    def test_valid_and_invalid_readings(self):
        ok = validate_readings(self.answer([100, 200, None], [300, 400, 500]), ["b1", "b2"], [600, 800, 1000], 1280)
        self.assertEqual(ok["readings"]["b1"], [100., 200., None])
        bad = [self.answer([100, 200, 300]), self.answer([100, 200, 300], [90, 400, 500]), self.answer([100, 200, 1300], [300, 400, 500]),
               self.answer([100, 200], [300, 400]), {**self.answer([1, 2, 3], [4, 5, 6]), "layout_mismatches": "none"},
               self.answer([100, 200, 300], [300, 400, 500], ids=("b1", "b9"))]
        for value in bad:
            with self.assertRaises(ValidationError):
                validate_readings(value, ["b1", "b2"], [600, 800, 1000], 1280)

    def test_render_and_visible_ids(self):
        edges, sides = boundaries([strip(-1.8, 1.8), strip(1.8, 5.4), strip(30, 33)])
        samples = [dense(e) for e in edges]
        rows = choose_rows(samples, VIEW)
        self.assertEqual(visible_ids(samples, VIEW, rows, (0., 0., 2.5)), ["b1", "b2", "b3"])  # the far strip is out of frame
        canvas = render_for_model(Image.new("RGB", (1280, 1280), "gray"), VIEW, samples, sides, [("L1", "#16d5ec")] * 3, rows, (0., 0., 2.5))
        self.assertEqual(canvas.size, (1280 + 52 + 12, 1280 + 60))


if __name__ == "__main__":
    unittest.main()


class OrderingTests(unittest.TestCase):
    def test_boundaries_are_numbered_left_to_right_and_extended(self):
        edges, sides = boundaries([strip(1.8, 5.4), strip(-5.4, -1.8), strip(-1.8, 1.8)])
        samples, sides = left_to_right([dense(e) for e in edges], sides, VIEW)
        firsts = [s[0][0] for s in samples]
        self.assertEqual(firsts, sorted(firsts))
        self.assertEqual(sides, [(2, 3), (0, 1), (1, 2)])
        longer = dense(edges[0], extend=10.)
        self.assertAlmostEqual(longer[0][1], 4. - 10., places=6)
        self.assertAlmostEqual(longer[-1][1], 40. + 10., places=6)


class GroupTests(unittest.TestCase):
    def entry(self, view, readings_pose):
        edges, _ = boundaries([strip(-1.8, 1.8), strip(1.8, 5.4), strip(5.4, 7.0)])
        samples = [dense(e, extend=10.) for e in edges]
        rows = choose_rows(samples, view)
        xs = crossings(samples, view, rows, readings_pose)
        readings = {f"b{i + 1}": [None if not np.isfinite(x) else float(x) for x in r] for i, r in enumerate(xs)}
        return {"view": view, "samples": samples, "rows": rows, "readings": readings, "fit": fit_pose(samples, view, rows, readings)}

    def test_shared_offset_and_an_outlier_that_matched_the_wrong_lanes(self):
        views = [dict(VIEW, id=f"v{k}", local_xy=[0., -float(k * 5)]) for k in range(3)]
        entries = [self.entry(views[0], (1.6, .8, 2.3)), self.entry(views[1], (1.6, -1.2, 2.3)),
                   self.entry(views[2], (1.6 + 3.6, .3, 2.3))]  # read one lane over
        group = fit_group(entries)
        self.assertEqual((group["inliers"], group["outliers"]), (["v0", "v1"], ["v2"]))
        self.assertAlmostEqual(group["shared"]["leg_shift_m"], 1.6, delta=.1)
        self.assertAlmostEqual(group["shared"]["camera_height_m"], 2.3, delta=.1)
        self.assertAlmostEqual(group["views"]["v1"]["heading_shift_deg"], -1.2, delta=.2)
        self.assertLess(group["views"]["v0"]["rms_px"], 3)

    def test_two_views_that_disagree_have_no_shared_pose(self):
        views = [dict(VIEW, id=f"v{k}", local_xy=[0., -float(k * 5)]) for k in range(2)]
        group = fit_group([self.entry(views[0], (0., 0., 2.5)), self.entry(views[1], (3.6, 0., 2.5))])
        self.assertIsNone(group["shared"])


class CheckTests(unittest.TestCase):
    def test_consistent_check_carries_no_readings_and_inconsistent_needs_them(self):
        strips, ids, rows = ["NB L1", "NB BIKE"], ["b1", "b2", "b3"], [600, 800, 1000]
        ok = validate_check({"strip_checks": [{"strip": "NB L1", "verdict": "consistent", "evidence": "left arrow"}], "boundaries": []},
                            ids, rows, 1280, strips)
        self.assertEqual((ok["inconsistent"], ok["corrected"]), (False, None))
        fixed = {"strip_checks": [{"strip": "NB BIKE", "verdict": "inconsistent", "evidence": "right arrow inside"}],
                 "boundaries": [{"id": i, "feature": "lane_line", "xs": [100 * k, 110 * k, 120 * k], "evidence": "paint"} for k, i in enumerate(ids, 1)]}
        self.assertTrue(validate_check(fixed, ids, rows, 1280, strips)["inconsistent"])
        for bad in ({**fixed, "boundaries": []}, {"strip_checks": [{"strip": "SB R5", "verdict": "consistent", "evidence": "x"}]},
                    {"strip_checks": [{"strip": "NB L1", "verdict": "consistent", "evidence": "x"}], "boundaries": fixed["boundaries"]},
                    {"strip_checks": []}):
            with self.assertRaises(ValidationError):
                validate_check(bad, ids, rows, 1280, strips)


class ArmTests(unittest.TestCase):
    def region(self, section, east, north):
        return {"section_id": section, "polygon_local_m": [[east - 1, north - 1], [east + 1, north - 1], [east + 1, north + 1], [east - 1, north + 1]]}

    def test_arms_from_geometry_at_any_angle(self):
        from movement_fixer.autoloop.gsv_projection import arm_of, facing, road_arms
        regions = [self.region("in_link_7", -5, -30), self.region("out_link_8", 5, -30),          # south arm
                   self.region("in_link_9", 30 * math.sin(math.radians(70)), 30 * math.cos(math.radians(70))),   # oblique arm at 70 deg
                   self.region("out_link_3", -25, 20)]                                              # north-west arm, outbound only
        arms = road_arms(regions)
        self.assertEqual(sorted(a["name"] for a in arms), ["link_3", "link_7", "link_9"])
        south = next(a for a in arms if a["name"] == "link_7")
        self.assertEqual(sorted(south["sections"]), ["in_link_7", "out_link_8"])
        self.assertAlmostEqual(south["inbound_heading"], 0., delta=1.)                              # travel toward the centre is north
        camera = {"local_xy": [3., -45.], "compass_heading_deg": 2.}
        self.assertEqual(arm_of(camera, arms)["name"], "link_7")
        self.assertEqual((facing(camera, south), facing(dict(camera, compass_heading_deg=181.), south)), (1., -1.))

    def test_look_back_view_shares_the_offset_with_a_reversed_sign(self):
        forward = dict(VIEW, id="f", local_xy=[0., -5.])
        back = dict(VIEW, id="b", local_xy=[0., 45.], compass_heading_deg=180.)
        g = GroupTests()
        entries = [dict(g.entry(forward, (1.6, .5, 2.4)), facing=1.), dict(g.entry(back, (-1.6, -.8, 2.4)), facing=-1.)]
        group = fit_group(entries)
        self.assertEqual(group["outliers"], [])
        self.assertAlmostEqual(group["shared"]["leg_shift_m"], 1.6, delta=.1)


class RealignTests(unittest.TestCase):
    def test_readings_two_boundaries_off_are_rematched_after_the_lane_shift(self):
        from movement_fixer.autoloop.gsv_projection import realign_group
        edges, _ = boundaries([strip(x, x + 3.5) for x in (-7., -3.5, 0., 3.5, 7.)])
        truth = (1.2, .6, 2.4)
        entries = []
        for k, y in enumerate((-4., -8.)):
            view = dict(VIEW, id=f"v{k}", local_xy=[0., y])
            samples = [dense(e, extend=10.) for e in edges]
            rows = choose_rows(samples, view)
            xs = crossings(samples, view, rows, truth)
            # The model read every real line correctly but named it two boundaries to the left.
            raw = {f"b{i - 1}": [None if not np.isfinite(x) else float(x) for x in xs[i]] for i in range(2, len(xs))}
            fit = fit_pose(samples, view, rows, raw)
            entries.append({"view": view, "samples": samples, "rows": rows, "facing": 1., "step": 3.5, "raw_readings": raw,
                            "readings": raw, "fit": fit, "pose": tuple(fit["pose"].values())})
        self.assertGreater(abs(entries[0]["pose"][0] - truth[0]), 5.)       # the first fit lands two lanes off
        group = realign_group(entries, round((truth[0] - entries[0]["pose"][0]) / 3.5))
        self.assertAlmostEqual(group["shared"]["leg_shift_m"], truth[0], delta=.15)
        self.assertAlmostEqual(entries[1]["pose"][1], truth[1], delta=.3)

    def test_refinement_moves_from_a_start_beyond_six_metres(self):
        from movement_fixer.autoloop.gsv_projection import refine_pose
        edges, _ = boundaries([strip(x, x + 3.5) for x in (-10.5, -7., -3.5, 0., 3.5)])
        samples = [dense(e, extend=10.) for e in edges]
        view = dict(VIEW, local_xy=[0., -4.])
        rows = choose_rows(samples, view, (-7.5, 0., 2.5))
        truth = (-7.9, -1.2, 2.3)
        xs = crossings(samples, view, rows, truth)
        readings = {f"b{i + 1}": [None if not np.isfinite(x) else float(x) for x in r] for i, r in enumerate(xs)}
        fit = refine_pose(samples, view, rows, readings, (-7.1, 0., 2.6))
        self.assertAlmostEqual(fit["pose"]["lateral_shift_m"], truth[0], delta=.1)
        self.assertAlmostEqual(fit["pose"]["heading_shift_deg"], truth[1], delta=.2)
        self.assertLess(fit["rms_px_after"], 2)

    def test_vote_rule(self):
        from movement_fixer.autoloop.gsv_projection import lane_offset
        self.assertEqual(lane_offset({"a": (-2, "high"), "b": (-2, "medium"), "c": (0, "low")}), (-2, "group_vote"))
        self.assertEqual(lane_offset({"a": (-2, "high")}), (-2, "single_vote"))
        self.assertEqual(lane_offset({"a": (-2, "high"), "b": (1, "high")}), (None, None))
        self.assertEqual(lane_offset({"a": (-2, "low")}), (None, None))


class OffsetTests(unittest.TestCase):
    def test_lane_width_and_offset_validation(self):
        from movement_fixer.autoloop.gsv_projection import lane_width, validate_offset
        regions = [(strip(-1.8, 1.8), ["NB", "L1"], "#16d5ec"), (strip(1.8, 5.2), ["NB", "2"], "#16d5ec"), (strip(5.2, 6.7), ["NB", "BIKE"], "#35c94a")]
        self.assertAlmostEqual(lane_width(regions), 3.5, places=6)  # the bicycle strip is not a lane step
        self.assertEqual(validate_offset({"panel": "B", "confidence": "high", "reasons": ["L1 on the left-arrow lane"]})["panel"], "B")
        for bad in ({"panel": "F", "confidence": "high", "reasons": ["x"]}, {"panel": "A", "confidence": "sure", "reasons": ["x"]},
                    {"panel": "A", "confidence": "high", "reasons": []}):
            with self.assertRaises(ValidationError):
                validate_offset(bad)
