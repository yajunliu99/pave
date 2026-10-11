"""Network Wrangler interop. The export tests need only the standard library; the round-trip
tests run when network_wrangler is installed:

  uv run --python 3.13 --with network-wrangler==1.0b4 python -m unittest tests.test_wrangler
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from vlm4scenario.gmns import Network
from vlm4scenario.pipeline import run
from vlm4scenario.vlm import MockProposer
from vlm4scenario.wrangler import LANES_PLACEHOLDER, SitePlacement, link_record, project_card, to_gmns, write_card

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "sr347_i10_ddi"
HAVE_WRANGLER = importlib.util.find_spec("network_wrangler") is not None


def pipeline_run(target=None) -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory()
    out = Path(tmp.name) / "run"
    with contextlib.redirect_stdout(io.StringIO()):
        code = run(SCEN, out, MockProposer(SCEN / "mock_proposal.json"), target=target)
    assert code == 0
    return tmp, out


class ExportTests(unittest.TestCase):
    def setUp(self):
        sc = json.loads((SCEN / "scenario.json").read_text())
        self.place = SitePlacement.for_scenario(sc)
        self.base = Network.load(SCEN / "base")

    def test_placement_is_invertible(self):
        for n in self.base.nodes.values():
            lon, lat = self.place.to_lonlat(n["x_coord"], n["y_coord"])
            x, y = self.place.to_xy(lon, lat)
            self.assertAlmostEqual(x, n["x_coord"], delta=0.002)
            self.assertAlmostEqual(y, n["y_coord"], delta=0.002)

    def test_x_axis_points_along_queen_creek_rd(self):
        lon0, lat0 = self.place.to_lonlat(0, 0)
        lon1, lat1 = self.place.to_lonlat(100, 0)
        self.assertGreater(lon1, lon0)      # NE: east and north
        self.assertGreater(lat1, lat0)

    def test_unknown_lanes_are_flagged_not_zero(self):
        rec = link_record(self.base.links[201])
        self.assertEqual((rec["lanes"], rec["lanes_known"]), (LANES_PLACEHOLDER, False))
        back = to_gmns([rec], [{"model_node_id": n, "X": -111.95, "Y": 33.25} for n in (1, 21)], self.place)
        self.assertEqual(back.links[201]["lanes"], "")

    def test_ddi_card_structure(self):
        tmp, out = pipeline_run()
        with tmp:
            after = Network.load(out / "after_network")
            card = project_card(self.base, after, self.place, project="t", tags=[], notes="")
            kinds = [next(iter(c)) for c in card["changes"]]
            self.assertEqual(kinds, ["roadway_property_change"] * 4 + ["roadway_deletion", "roadway_addition", "roadway_property_change"])
            self.assertEqual(card["changes"][4]["roadway_deletion"]["links"]["model_link_id"], [401, 402, 411, 412])
            added = card["changes"][5]["roadway_addition"]
            self.assertEqual([n["model_node_id"] for n in added["nodes"]], [41, 42, 43, 44])
            self.assertTrue((out / "wrangler" / "project_card.yml").exists())


@unittest.skipUnless(HAVE_WRANGLER, "network_wrangler is not installed")
class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        from wrangler_roundtrip import roundtrip
        cls.roundtrip = staticmethod(roundtrip)

    def test_every_target_round_trips(self):
        for target in (None, "spui", "roundabout_diamond"):
            with self.subTest(target=target or "ddi"):
                tmp, out = pipeline_run(target)
                with tmp:
                    rep = self.roundtrip(out)
                    self.assertTrue(rep["match"], json.dumps(rep, indent=1)[:2000])

    def test_a_missing_change_is_caught(self):
        import yaml
        tmp, out = pipeline_run()
        with tmp:
            card = yaml.safe_load((out / "wrangler" / "project_card.yml").read_text())
            card = copy.deepcopy(card)
            del card["changes"][0]                       # forget to move node 23
            bad = write_card(card, out / "wrangler" / "bad_card.yml")
            rep = self.roundtrip(out, bad)
            self.assertFalse(rep["match"])
            self.assertTrue(any(m.startswith("23:") for m in rep["nodes"]["mismatches"]))


if __name__ == "__main__":
    unittest.main()
