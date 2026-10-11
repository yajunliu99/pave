"""Strip geometry, edits, coordinate checks, crop windows and rendering. No model, no network."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image
from movement_fixer.hybrid.common import ValidationError
from movement_fixer.hybrid.geometry import rotate
from movement_fixer.autoloop import render
from movement_fixer.autoloop.checks import check_strips
from movement_fixer.autoloop.frames import frames_from_network, parse_linestring
from movement_fixer.autoloop.geometry_stage import draft_prompt, review_prompt
from movement_fixer.autoloop.strips import (Frame, apply_edits, change_px, region_widths, section_regions,
                                            signature, validate_strips)

STATIONS = [20., 110., 200., 290., 380.]


def strips(*xs):
    return {"stations": list(STATIONS), "edges": [[float(x)] * len(STATIONS) for x in xs]}


def codes(issues):
    return {(i["code"], i["target"]) for i in issues}


class FrameTests(unittest.TestCase):
    def test_round_trip_at_any_heading(self):
        for heading in (0, 90, 180, 270, 37.5, 251.2):
            frame = Frame((812.5, 640.0), heading, 240, 400)
            for point in ((0, 0), (240, 400), (57.3, 311.8)):
                back = frame.to_local(*frame.to_source(*point))
                self.assertAlmostEqual(back[0], point[0], places=6)
                self.assertAlmostEqual(back[1], point[1], places=6)

    def test_travel_points_up_and_driver_right_is_image_right(self):
        east = Frame((500, 500), 90, 200, 300)  # travelling toward +x in the source image
        self.assertAlmostEqual(east.to_source(100, 0)[0], 650)   # top of the crop is further along the travel direction
        self.assertAlmostEqual(east.to_source(200, 150)[1], 600)  # driver-right of eastbound travel is +y (south)

    def test_cardinal_crops_keep_source_pixels(self):
        pixels = np.random.RandomState(7).randint(0, 255, (160, 200, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        boxes = {0: (70, 30, 130, 130), 180: (70, 30, 130, 130), 90: (50, 50, 150, 110), 270: (50, 50, 150, 110)}
        for heading, box in boxes.items():
            frame = Frame((100, 80), heading, 60, 100)
            expected = rotate(image.crop(box), heading)
            self.assertTrue(frame.pixels_preserved())
            self.assertEqual(render.crop(image, frame).tobytes(), expected.tobytes(), heading)

    def test_oblique_crop_is_marked_resampled(self):
        self.assertFalse(Frame((100, 80), 33, 60, 100).pixels_preserved())


class StripTests(unittest.TestCase):
    def test_structural_validation(self):
        validate_strips(strips(10, 60, 110), 200)
        for bad in (strips(10, 60, 58), strips(10, 60, 62), strips(-1, 60), strips(10, 260), strips(10)):
            with self.assertRaises(ValidationError):
                validate_strips(bad, 200)
        ragged = strips(10, 60)
        ragged["edges"][1] = [60., 60.]
        with self.assertRaises(ValidationError):
            validate_strips(ragged, 200)

    def test_regions_use_source_pixels_and_gate_at_junction_end(self):
        frame = Frame((1000, 1500), 0, 200, 400)
        s = strips(40, 90, 150)
        inbound = section_regions(s, frame, "inbound_stopbar")
        outbound = section_regions(s, frame, "outbound_receiving")
        self.assertEqual([r["id"] for r in inbound], ["r1", "r2"])
        self.assertEqual(inbound[0]["polygon"][0], [940.0, 1320.0])     # left boundary at the top station
        self.assertEqual(inbound[0]["gate_point"], [965.0, 1320.0])     # stop bar is the top station
        self.assertEqual(outbound[0]["gate_point"], [965.0, 1680.0])    # receiving gate is nearest the junction
        southbound = section_regions(s, Frame((1000, 600), 180, 200, 400), "inbound_stopbar")
        self.assertEqual(southbound[0]["gate_point"], [1035.0, 780.0])  # rotated: driver-left is image-right

    def test_each_edit_operation(self):
        base = strips(10, 60, 110, 160)
        moved, applied, rejected = apply_edits(base, [{"op": "move_edge", "edge": "e2", "dx": 5}], 200)
        self.assertEqual(moved["edges"][1][0], 65)
        self.assertEqual((len(applied), rejected), (1, []))
        split, _, _ = apply_edits(base, [{"op": "split_region", "region": "r2"}], 200)
        self.assertEqual([e[0] for e in split["edges"]], [10, 60, 85, 110, 160])
        merged, _, _ = apply_edits(base, [{"op": "merge_regions", "regions": ["r2", "r1"]}], 200)
        self.assertEqual([e[0] for e in merged["edges"]], [10, 110, 160])
        added, _, _ = apply_edits(base, [{"op": "add_region", "side": "right", "xs": [190] * 5}], 200)
        self.assertEqual(len(region_widths(added)), 4)
        dropped, _, _ = apply_edits(base, [{"op": "drop_region", "region": "r1"}], 200)
        self.assertEqual([e[0] for e in dropped["edges"]], [60, 110, 160])
        self.assertEqual(split["meta"][2]["origin"], "review")

    def test_applied_edits_are_described_by_position(self):
        base = strips(10, 60, 110, 160)
        edits = [{"op": "merge_regions", "regions": ["r1", "r2"]}, {"op": "move_edge", "edge": "e3", "dx": 6},
                 {"op": "split_region", "region": "r3"}, {"op": "drop_region", "region": "r3"},
                 {"op": "add_region", "side": "left", "xs": [2] * 5}]
        _, applied, _ = apply_edits(base, edits[:3] + edits[4:], 200)
        self.assertEqual([e["effect"] for e in applied],
                         ["removed the boundary at x≈60", "moved the boundary at x≈110 to x≈116", "added a boundary at x≈138",
                          "added an outer boundary on the left at x≈2"])
        _, applied, _ = apply_edits(base, [edits[3]], 200)
        self.assertEqual(applied[0]["effect"], "removed the outer boundary at x≈160")

    def test_ids_refer_to_the_reviewed_geometry_across_a_batch(self):
        base = strips(10, 60, 110, 160)
        edits = [{"op": "split_region", "region": "r1"}, {"op": "move_edge", "edge": "e4", "xs": [170] * 5},
                 {"op": "merge_regions", "regions": ["r2", "r3"]}]
        result, applied, rejected = apply_edits(base, edits, 200)
        self.assertEqual([e[0] for e in result["edges"]], [10, 35, 60, 170])  # e4 is still the original fourth boundary
        self.assertEqual((len(applied), rejected), (3, []))

    def test_unusable_edits_are_rejected_not_guessed(self):
        base = strips(10, 60, 110, 160)
        edits = [{"op": "move_edge", "edge": "e2", "xs": [120] * 5},           # would cross e3
                 {"op": "drop_region", "region": "r2"},                         # interior
                 {"op": "merge_regions", "regions": ["r1", "r3"]},              # not neighbours
                 {"op": "move_edge", "edge": "e9", "dx": 1},                    # unknown ID
                 {"op": "merge_regions", "regions": ["r1", "r2"]},              # fine: removes e2
                 {"op": "move_edge", "edge": "e2", "dx": 3},                    # e2 is gone
                 {"op": "paint", "edge": "e1"}]
        result, applied, rejected = apply_edits(base, edits, 200)
        self.assertEqual([e[0] for e in result["edges"]], [10, 110, 160])
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(rejected), 6)
        self.assertIn("removed by an earlier edit", rejected[4]["reason"])

    def test_change_and_signature(self):
        a, b = strips(10, 60, 110), strips(10, 63, 110)
        self.assertEqual(change_px(a, b), 3)
        self.assertIsNone(change_px(a, strips(10, 60)))
        self.assertNotEqual(signature(a), signature(b))
        self.assertEqual(signature(a), signature(strips(10, 60.3, 110)))  # sub-pixel jitter is the same state


class CheckTests(unittest.TestCase):
    def test_regular_strips_raise_nothing(self):
        self.assertEqual(check_strips(strips(20, 70, 120, 170, 220), 300), [])

    def test_merged_and_narrow_use_the_existing_thresholds(self):
        self.assertIn(("possible_merged_lanes", "r3"), codes(check_strips(strips(20, 70, 120, 220, 270), 300)))
        self.assertIn(("narrow_region_check_facility", "r3"), codes(check_strips(strips(20, 70, 120, 145, 195), 300)))

    def test_two_regions_have_no_relative_reference(self):
        self.assertEqual(check_strips(strips(20, 70, 220), 300), [])

    def test_absolute_width_needs_ground_scale(self):
        wide = strips(20, 70, 220)
        self.assertIn(("possible_merged_lanes", "r2"), codes(check_strips(wide, 300, mpp=.06)))  # 9 m
        self.assertIn(("sliver_region", "r1"), codes(check_strips(strips(20, 30, 90), 300, mpp=.06)))  # 0.6 m

    def test_taper_jagged_and_border(self):
        taper = strips(20, 70, 120, 170)
        taper["edges"][1] = [70., 62., 54., 46., 38.]
        self.assertIn(("width_varies_along_section", "r1"), codes(check_strips(taper, 300)))
        jagged = strips(20, 70, 120, 170)
        jagged["edges"][2] = [120., 120., 150., 120., 120.]
        found = [i for i in check_strips(jagged, 300) if i["code"] == "jagged_edge"]
        self.assertEqual((found[0]["target"], found[0]["detail"]["station"]), ("e3", "S3"))
        border = codes(check_strips(strips(2, 60, 110, 297), 300))
        self.assertTrue({("touches_crop_border", "e1"), ("touches_crop_border", "e4")} <= border)

    def test_issue_key_follows_the_flagged_geometry(self):
        first = check_strips(strips(20, 70, 120, 220, 270), 300)[0]
        again = check_strips(strips(20, 70, 120, 220, 270), 300)[0]
        moved = check_strips(strips(20, 70, 120, 230, 270), 300)
        self.assertEqual(first["key"], again["key"])
        self.assertNotIn(first["key"], {i["key"] for i in moved})


def link(link_id, start, end, node=1, inbound=True, lanes=2, biway=1, uses="auto"):
    a, b = (start, end) if inbound else (end, start)
    return {"link_id": link_id, "from_node_id": 9 if inbound else node, "to_node_id": node if inbound else 9,
            "geometry": f"LINESTRING ({a[0]} {a[1]}, {b[0]} {b[1]})", "lanes": lanes, "allowed_uses": uses, "from_biway": biway}


def four_way():
    c = (600, 600)
    return [link(1, (600, 1100), c), link(2, (600, 100), c), link(3, (100, 600), c), link(4, (1100, 600), c),
            link(5, (600, 100), c, inbound=False), link(6, (600, 1100), c, inbound=False),
            link(7, (1100, 600), c, inbound=False), link(8, (100, 600), c, inbound=False)]


def identity(lon, lat):
    return lon, lat


class FrameFromNetworkTests(unittest.TestCase):
    def test_four_way_shared_axis(self):
        specs = {s["id"]: s for s in frames_from_network(1, (600, 600), four_way(), identity, .1, (1200, 1200))}
        self.assertEqual(set(specs), {"in_NB", "in_SB", "in_EB", "in_WB", "out_NB", "out_SB", "out_EB", "out_WB"})
        nb = specs["in_NB"]
        self.assertEqual((nb["link_id"], nb["frame"].heading_deg, nb["direction"]), (1, 0, "NB"))
        # 2 crossing lanes x 3.6 m + 5 m crosswalk margin = 12.2 m, then a 28 m window: centre 26.2 m upstream.
        self.assertEqual(nb["crop_prior"]["near_m"], 12.2)
        self.assertAlmostEqual(nb["frame"].center[1], 600 + 262, places=1)
        self.assertAlmostEqual(nb["frame"].center[0], 600 + 36, places=1)   # right half of a shared axis
        self.assertAlmostEqual(specs["in_SB"]["frame"].center[0], 600 - 36, places=1)
        self.assertEqual(specs["in_EB"]["frame"].heading_deg, 90)
        self.assertAlmostEqual(specs["in_EB"]["frame"].center[1], 600 + 36, places=1)
        out = specs["out_NB"]
        self.assertEqual((out["link_id"], out["kind"], out["frame"].heading_deg), (5, "outbound_receiving", 0))
        self.assertAlmostEqual(out["frame"].center[1], 600 - 262, places=1)
        self.assertTrue(all(s["usable"] and s["reference_kind"] == "estimated_junction_envelope" for s in specs.values()))
        self.assertEqual(len(nb["stations"]), 5)
        self.assertEqual((nb["stations"][0], nb["stations"][-1]), (20, nb["frame"].height - 20))

    def test_directed_carriageway_is_not_shifted_again(self):
        links = four_way()
        links[0]["from_biway"] = 0
        links[5]["geometry"] = "LINESTRING (750 600, 750 1100)"  # the opposite direction runs on its own carriageway
        nb = next(s for s in frames_from_network(1, (600, 600), links, identity, .1, (1200, 1200)) if s["id"] == "in_NB")
        self.assertEqual((nb["frame"].center[0], nb["centerline_reference"]), (600, "directed_carriageway"))

    def test_two_directed_links_on_one_centreline_share_the_axis(self):
        # Some networks draw a two-way road as two directed links on the same line and leave from_biway at 0.
        links = four_way()
        for l in links:
            l["from_biway"] = 0
        nb = next(s for s in frames_from_network(1, (600, 600), links, identity, .1, (1200, 1200)) if s["id"] == "in_NB")
        self.assertEqual(nb["centerline_reference"], "shared_road_axis")
        self.assertGreater(nb["frame"].center[0], 600)  # moved onto the northbound half

    def test_consolidated_node_with_sub_node_stubs(self):
        # Dual carriageways whose links start at former sub-nodes 8 m either side of the merged node
        # and reach their own axis through a diagonal stub, as net2cell's macronet does.
        def stub(link_id, far, bend, sub):
            return {"link_id": link_id, "from_node_id": 9, "to_node_id": 1, "lanes": 2, "allowed_uses": "auto", "from_biway": 0,
                    "geometry": "LINESTRING (" + ", ".join(f"{x} {y}" for x, y in (far, bend, sub)) + ")"}
        links = [link(1, (680, 1150), (680, 600), biway=0), link(2, (520, 50), (520, 600), biway=0),
                 stub(3, (1150, 550), (730, 550), (680, 600)), stub(4, (50, 650), (470, 650), (520, 600))]
        specs = {s["id"]: s for s in frames_from_network(1, (600, 600), links, identity, .1, (1200, 1200))}
        wb = specs["in_WB"]
        # Crossing road reaches 8 m (axis offset) + 3.6 m (half of two lanes); + 5 m margin = 16.6 m from the node.
        self.assertEqual((wb["crop_prior"]["near_m"], wb["crop_prior"]["axis_offset_m"], wb["frame"].heading_deg), (16.6, 5.0, 270))
        self.assertAlmostEqual(wb["frame"].center[0], 600 + 166 + 140, places=1)  # from the node, not from the sub-node
        self.assertAlmostEqual(wb["frame"].center[1], 550, places=1)              # on its own carriageway axis
        nb = specs["in_NB"]
        self.assertEqual(nb["crop_prior"]["near_m"], 13.6)                         # 5 m offset + 3.6 m + 5 m
        self.assertAlmostEqual(nb["frame"].center[1], 600 + 136 + 140, places=1)
        self.assertAlmostEqual(nb["frame"].center[0], 680, places=1)

    def test_oblique_three_leg_and_duplicate_cardinals(self):
        c = (600, 600)
        links = [link(1, (350, 1000), c), link(2, (850, 1000), c), link(3, (600, 150), c),
                 link(4, (600, 150), c, inbound=False), link(5, (300, 600), c, uses="bike")]
        specs = frames_from_network(1, (600, 600), links, identity, .1, (1200, 1200))
        ids = [s["id"] for s in specs]
        # Two northbound-ish legs get eight-point names; no bike link.
        self.assertEqual(sorted(ids), ["in_NEB", "in_NWB", "in_SB", "out_NB"])
        self.assertEqual({s["id"]: s["arm"] for s in specs}, {"in_NEB": "SW", "in_NWB": "SE", "in_SB": "N", "out_NB": "N"})
        one = next(s for s in specs if s["id"] == "in_NEB")
        self.assertAlmostEqual(one["frame"].heading_deg, 32.005, places=2)
        self.assertFalse(one["frame"].pixels_preserved())

    def test_window_outside_the_image_is_reported(self):
        specs = frames_from_network(1, (600, 600), four_way(), identity, .1, (800, 800))
        blocked = [s for s in specs if not s["usable"]]
        self.assertTrue(blocked and all(s["unusable_reason"] == "crop_window_outside_satellite_image" for s in blocked))

    def test_short_link_is_extrapolated_and_flagged(self):
        links = four_way()
        links[0]["geometry"] = "LINESTRING (600 700, 600 600)"
        nb = next(s for s in frames_from_network(1, (600, 600), links, identity, .1, (1200, 1200)) if s["id"] == "in_NB")
        self.assertTrue(nb["extrapolated_beyond_link"])
        self.assertAlmostEqual(nb["frame"].center[1], 862, places=1)

    def test_wkt_parsing(self):
        self.assertEqual(parse_linestring("LINESTRING (-111.94 33.42, -111.93 33.43)"), [(-111.94, 33.42), (-111.93, 33.43)])
        with self.assertRaises(ValidationError):
            parse_linestring("POINT (1 2)")


class PromptIsolationTests(unittest.TestCase):
    def test_network_lane_prior_never_reaches_a_prompt(self):
        prompts = []
        for lanes in (2, 5):
            spec = next(s for s in frames_from_network(1, (600, 600), [dict(l, lanes=lanes) for l in four_way()], identity, .1, (1200, 1200))
                        if s["id"] == "in_NB")
            self.assertEqual(spec["crop_prior"]["lanes"], lanes)
            s = strips(20, 70, 120)
            s["stations"] = spec["stations"]
            prompts.append(draft_prompt(spec) + review_prompt(spec, s, [], 1, []))
        self.assertEqual(prompts[0], prompts[1])
        for word in ("crop_prior", "link_id", "near_m", "declared"):
            self.assertNotIn(word, prompts[0])


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.raw = Image.new("RGB", (240, 400), (90, 90, 90))
        self.strips = strips(30, 80, 130, 200)

    def test_annotations_stay_off_the_pavement_except_dashed_lines(self):
        ruled = np.asarray(render.ruled(self.raw, STATIONS, scale=2))
        body = ruled[render.TOP:render.TOP + 800, render.LEFT:render.LEFT + 480]
        changed = np.any(body != 90, axis=2)
        rows = set(np.flatnonzero(changed.any(axis=1)))
        self.assertEqual(len(rows), 2 * len(STATIONS))                              # two display rows per station
        self.assertTrue(all(min(abs(r - round(y * 2)) for y in STATIONS) <= 1 for r in rows))
        self.assertLess(changed.sum() / (2 * len(STATIONS) * 480), .5)              # dashes leave most of each row visible
        self.assertEqual(ruled.shape[:2], (render.TOP + 800 + render.BOTTOM, render.LEFT + 480 + render.RIGHT))

    def test_overlay_and_panels(self):
        over = np.asarray(render.overlay(self.raw, self.strips, scale=2))
        cyan = np.all(over == (0x16, 0xd5, 0xec), axis=2)[render.TOP + 40:render.TOP + 760]
        for x in (30, 80, 130, 200):
            column = cyan[:, render.LEFT + 2 * x - 2:render.LEFT + 2 * x + 2].any(axis=1)
            self.assertTrue(.5 < column.mean() < .8, column.mean())                 # dashed: paint shows through the gaps
        self.assertFalse(cyan[:, render.LEFT + 2 * 100 - 6:render.LEFT + 2 * 100 + 6].any())
        panel = render.panels(self.raw, self.strips, scale=2)
        self.assertEqual(panel.height, 40 + 2 * (380 - 20) + 12 + 2)
        self.assertGreater(panel.width, 2 * (50 + 50 + 70))


if __name__ == "__main__":
    unittest.main()


class PartialBoundaryTests(unittest.TestCase):
    def test_null_stations_are_filled_and_marked(self):
        from movement_fixer.autoloop.geometry_stage import fill_partial, validate_draft
        self.assertEqual(fill_partial([None, None, 116, 112, 107]), [116., 116., 116., 112., 107.])
        self.assertEqual(fill_partial([100, None, 120, None, None]), [100., 110., 120., 120., 120.])
        spec = next(s for s in frames_from_network(1, (600, 600), four_way(), identity, .1, (1200, 1200)) if s["id"] == "in_NB")
        draft = {"boundaries": [{"xs": [None, None, 16, 12, 7], "kind": "centre_or_median_edge", "visibility": "partial", "evidence": "divider"},
                                {"xs": [146, 98, 94, 89, 82], "kind": "kerb_or_pavement_edge", "visibility": "partial", "evidence": "kerb"}]}
        result = validate_draft(draft, spec)
        self.assertEqual(result["strips"]["edges"][0], [16., 16., 16., 12., 7.])
        self.assertEqual(result["strips"]["meta"][0]["interpolated_stations"], ["S1", "S2"])
        with self.assertRaises(ValidationError):
            validate_draft({"boundaries": [{**draft["boundaries"][0], "xs": [None, None, None, None, 107]}, draft["boundaries"][1]]}, spec)


class ShadowTests(unittest.TestCase):
    def test_only_wholly_shaded_crops_are_stretched(self):
        from PIL import Image
        from movement_fixer.autoloop.render import lit
        dark = Image.new("RGB", (60, 120), (40, 45, 55))
        dark.paste((52, 57, 66), (28, 0, 31, 120))  # a faint lane line in shadow
        stretched = lit(dark)
        self.assertIsNotNone(stretched)
        self.assertGreater(stretched.getpixel((29, 60))[0] - stretched.getpixel((10, 60))[0], 12)
        sunlit = dark.copy()
        sunlit.paste((180, 180, 180), (0, 0, 60, 40))  # part of the crop in sun
        self.assertIsNone(lit(sunlit))
        self.assertIsNone(lit(Image.new("RGB", (60, 120), (120, 120, 125))))
