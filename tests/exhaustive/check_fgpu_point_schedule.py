#!/usr/bin/env python3
"""密图 Point 首轮计数、推迟与最终完整回扫回归；零成本图的所有 tour 都最优。"""

import json
from pathlib import Path
import subprocess
import sys

from check_fgpu_solve import edge_set, nonpair_set


def main():
    executable, output = (Path(x).resolve() for x in sys.argv[1:3])
    root = Path(__file__).resolve().parents[2]
    if output == root or not output.is_relative_to(root):
        raise ValueError("测试输出必须位于项目内")
    for n in (18, 19, 24, 40):
        directory = output / f"n{n}"
        directory.mkdir(parents=True, exist_ok=True)
        instance = directory / "input.tsp"
        instance.write_text(f"NAME: point-schedule-{n}\nTYPE: TSP\nDIMENSION: {n}\n"
                            "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n" +
                            "".join(f"{i+1} 0 0\n" for i in range(n)) + "EOF\n")
        expected_edges = {(a, b) for a in range(n) for b in range(a + 1, n)}
        hashes = set()
        for adaptive, prime in ((0, 0), (1, 0), (1, 1)):
            target = directory / f"adaptive{adaptive}-prime{prime}"
            target.mkdir(exist_ok=True)
            command = [str(executable), "solve", "--profile", "hybrid-e2e",
                       "--instance", str(instance), "--device", "0", "--lp-backend", "off",
                       "--leaf-permutation-cache", "1", "--point-near-first", "1",
                       "--point-adaptive-start", str(adaptive),
                       "--point-prime-near", str(prime),
                       # n=40 用于验证 32 点预热前缀不是最终扫描上限；隔离掉昂贵 metric。
                       "--full-metric", "0" if n == 40 else "1",
                       "--output-edges", str(target / "out.edg"),
                       "--fixed", str(target / "out.fix"),
                       "--nonpairs", str(target / "out.nonpairs"),
                       "--manifest", str(target / "out.json")]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(f"n={n}, adaptive={adaptive}: {process.stderr}")
            report = json.loads((target / "out.json").read_text())
            assert report["termination"] == "fixed-point" and report["gpu_replayed"]
            assert report["proof_rejected"] == 0 and not report["input_optimum_labels"]
            assert report["point_adaptive_start"] == bool(adaptive)
            assert report["point_prime_near"] == bool(prime)
            assert report["point_initial_pairs"] == n * (n - 1) * (n - 2) // 2
            assert report["point_initial_edge_frontier"] == len(expected_edges) * 16
            deferred = bool(adaptive) and n > 18
            assert report["point_deferred_initially"] == deferred
            assert report["point_deferred_sweeps"] == int(deferred)
            assert report["point_service_sweeps"] >= 1, "推迟后缺失最终完整 Point 回扫"
            assert report["point_prime_sweeps"] == int(bool(prime) and deferred)
            assert edge_set(target / "out.edg") == expected_edges, "误删零成本最优边"
            assert not edge_set(target / "out.fix"), "零成本完全图没有必选边"
            assert not nonpair_set(target / "out.nonpairs", n), "零成本完全图每个邻边对均合法"
            hashes.add(report["final_state_hash"])
        assert len(hashes) == 1, "执行次序改变了零成本图的终态"
        print(f"point schedule n={n}: exact frontier and full final sweep verified", flush=True)


if __name__ == "__main__":
    main()
