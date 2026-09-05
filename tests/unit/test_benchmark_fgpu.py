#!/usr/bin/env python3
"""不访问 GPU 的基准工具边界、身份漂移和设备过滤测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fgpu_bench", ROOT / "tools/benchmark_fgpu.py")
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkTests(unittest.TestCase):
    def test_host_identity(self) -> None:
        identity = BENCH.host_identity()
        self.assertTrue(identity["hostname"])
        self.assertTrue(identity["machine"])
        self.assertGreater(identity["logical_cpus"], 0)

    def test_hybrid_does_not_supply_labels(self) -> None:
        inputs = {"instance": ROOT / "tests/data/pr299.tsp",
                  "tour": ROOT / "tests/data/pr299.opt.tour"}
        command = BENCH.solver_command(ROOT / ".tmp/solver", "hybrid-e2e", inputs,
                                       ROOT / "artifacts/test", 48191, ["--lp-backend", "off"])
        self.assertIn("hybrid-e2e", command)
        self.assertNotIn(str(inputs["tour"]), command)
        for argument in ("--tour", "--tour-role", "--expected-cost", "--input-edges", "48191"):
            self.assertNotIn(argument, command)
        inputs["input_edges"] = ROOT / "tests/data/pr299.edg"
        with self.assertRaises(ValueError):
            BENCH.solver_command(ROOT / ".tmp/solver", "hybrid-e2e", inputs,
                                 ROOT / "artifacts/test", 48191, [])
        legacy = BENCH.solver_command(ROOT / ".tmp/solver", "legacy", inputs,
                                      ROOT / "artifacts/test", 48191, [])
        self.assertIn("--tour", legacy)
        self.assertIn("--input-edges", legacy)
        self.assertNotIn("--profile", legacy)

    def test_boundary(self) -> None:
        self.assertEqual(BENCH.inside(ROOT / "artifacts/test"), ROOT / "artifacts/test")
        for path in [ROOT, ROOT.parent / "escaped", ROOT / ".." / "escaped"]:
            with self.assertRaises(ValueError):
                BENCH.inside(path)

    def test_device_identity_filter(self) -> None:
        output = ["GPU-other, card, 100, 200, 90\nGPU-id, card, 0, 200, 0\n",
                  "GPU-other, 10, training, 100\n"]
        with patch.object(BENCH.subprocess, "check_output", side_effect=output):
            self.assertEqual(BENCH.gpu_snapshot("GPU-id")["processes"], [])
        with patch.object(BENCH.subprocess, "check_output", side_effect=output):
            with self.assertRaises(ValueError):
                BENCH.gpu_snapshot("GPU-missing")

    def test_identity_and_worktree(self) -> None:
        base = ROOT / "build/test-benchmark"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="identity-", dir=base) as temporary:
            directory = Path(temporary)
            executable = directory / "fgpu-elim"
            executable.write_bytes(b"abc")
            digest = {"binary": BENCH.sha256(executable)}
            self.assertEqual(digest["binary"],
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            BENCH.unchanged({"binary": executable}, digest)
            executable.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                BENCH.unchanged({"binary": executable}, digest)
            cache = directory / "CMakeCache.txt"
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={directory}\n")
            self.assertEqual(BENCH.variant_source_root(executable), directory)
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={ROOT.parent}\n")
            with self.assertRaises(ValueError):
                BENCH.variant_source_root(executable)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
