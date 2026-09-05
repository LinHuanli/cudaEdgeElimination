#!/usr/bin/env python3
"""原始坐标语义和标签隔离辅助函数测试，不依赖下载或 GPU。"""

from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_hs2014_data import distance, load_instance, load_tour, records


class DataTests(unittest.TestCase):
    def test_half_rounding(self):
        origin = (Decimal(0), Decimal(0))
        self.assertEqual(distance(origin, (Decimal("0.5"), Decimal(0)), "EUC_2D"), 1)
        self.assertEqual(distance(origin, (Decimal("0.5"), Decimal("0.5")), "CEIL_2D"), 1)
        self.assertEqual(distance(origin, (Decimal("1.5"), Decimal(0)), "EUC_2D"), 2)
        self.assertEqual(distance(origin, (Decimal("3"), Decimal(4)), "CEIL_2D"), 5)

    def test_original_not_normalized(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as name:
            tsp = Path(name) / "half.tsp"
            tsp.write_text("DIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
                           "1 1.88955e+04 1.13300e+04\n2 0 0\n3 -0.5 3\nEOF\n")
            points, metric = load_instance(tsp)
            self.assertEqual(points[0][0], Decimal("18895.5"))
            self.assertEqual(points[2][0], Decimal("-0.5"))
            self.assertEqual(metric, "EUC_2D")
            tour = Path(name) / "tour"
            tour.write_text("TOUR_SECTION\n1 2 2\n-1\n")
            with self.assertRaises(ValueError):
                load_tour(tour, 3)

    def test_frozen_suite(self):
        self.assertEqual([x["name"] for x in records()], ["pr1002", "vm1084", "pcb1173", "pcb3038"])
        self.assertEqual([x["paper_edges"] for x in records()], [4521, 4610, 6084, 14869])


if __name__ == "__main__":
    unittest.main()
