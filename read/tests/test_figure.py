import json
import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pave_read.harness import figure  # noqa: E402

RUN = ROOT / "examples/broadway_ames_2021/run"


class FigureTest(unittest.TestCase):
    def test_labels(self):
        lane = {"region_id": "r2", "surface_type": "motor_vehicle_lane", "existence": "supported", "motor_lane_index": 2,
                "lane_use_prediction": {"allowed_turns": ["right", "through"]}}
        self.assertEqual(figure.label(lane, True), ("TR2", figure.MOTOR))
        self.assertEqual(figure.label(lane, False), ("OUT2", figure.MOTOR))
        self.assertEqual(figure.label({**lane, "existence": "rejected_by_lane_use"}, True)[0], "NOT A LANE")
        bike = {"region_id": None, "surface_type": "bicycle_lane", "existence": "not_motor"}
        self.assertEqual(figure.label(bike, True), ("BIKE", figure.BIKE))

    def test_gate_is_the_junction_end(self):
        # a strip from the junction (y = 0) away to y = 100: first and last points sit at the junction end
        polygon = [[0, 0], [0, 50], [0, 100], [10, 100], [10, 50], [10, 0]]
        self.assertEqual(figure.gate(polygon), (5, 0))
        self.assertEqual(figure.axis(polygon), (0, -1))

    def test_example_figure(self):
        lanes = json.loads((RUN / "lanes.json").read_text())
        movements = json.loads((RUN / "movements.json").read_text())
        image = Image.new("RGB", (2560, 2560), "gray")
        out = figure.overview(image, lanes, movements)
        self.assertLess(out.width, 2560)                 # cropped to what it shows
        self.assertGreater(out.height, out.width * 0.5)
        self.assertEqual(out.getpixel((5, 5)), (0x16, 0x1a, 0x22))   # legend band on top


if __name__ == "__main__":
    unittest.main()
