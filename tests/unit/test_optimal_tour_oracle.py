#!/usr/bin/env python3
"""全最优解 oracle 自身的并集／交集独立性门禁，无需 GPU。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exhaustive"))
from check_fgpu_solve import optimal_tour_facts


class OptimalTourOracleTests(unittest.TestCase):
    def test_two_optima(self):
        optimum, union, mandatory, pairs = optimal_tour_facts(
            [(0, 5), (9, 3), (5, 2), (5, 11), (1, 5)]
        )
        self.assertEqual(optimum, 27)
        self.assertEqual(union, {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)})
        self.assertEqual(mandatory, {(0, 4), (1, 2), (1, 3)})
        self.assertEqual(len(pairs), 9)
        self.assertIsNot(union, mandatory)

    def test_all_tours_optimal(self):
        optimum, union, mandatory, pairs = optimal_tour_facts([(0, 0)] * 4)
        self.assertEqual(optimum, 0)
        self.assertEqual(len(union), 6)
        self.assertFalse(mandatory)
        self.assertEqual(len(pairs), 12)


if __name__ == "__main__":
    unittest.main()
