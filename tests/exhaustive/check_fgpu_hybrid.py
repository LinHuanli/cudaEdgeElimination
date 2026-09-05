#!/usr/bin/env python3
"""无标签完整图入口的独立全最优解门禁，以及 LP/cache 消融一致性。"""

import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

from check_fgpu_solve import distance, edge_set, nonpair_set, optimal_tour_facts


def main():
    exe, output = (Path(x).resolve() for x in sys.argv[1:3])
    root = Path(__file__).resolve().parents[2]
    if output == root or not output.is_relative_to(root):
        raise ValueError("测试输出必须位于项目内")
    rng = random.Random(20142023)
    fixtures = [[(rng.randrange(40), rng.randrange(40)) for _ in range(n)] for n in range(3, 10)]
    fixtures += [[(0, 0)] * 5, [(0, 5), (9, 3), (5, 2), (5, 11), (1, 5)],
                 [(10 * x, 10 * y) for x in range(5) for y in range(2)],
                 [(0, 0), (0.5, 0), (2.5, 3), (4, 2.5), (5, 3), (8.5, 0)]]
    fixtures.append(fixtures[-1])
    nonpair_off = 0
    for case, points in enumerate(fixtures):
        directory = output / f"case-{case}"
        directory.mkdir(parents=True, exist_ok=True)
        n = len(points)
        metric = "CEIL_2D" if case == len(fixtures) - 1 else "EUC_2D"
        tsp = directory / "input.tsp"
        tsp.write_text(f"NAME: hybrid-{case}\nTYPE: TSP\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: {metric}\nNODE_COORD_SECTION\n" +
                       "".join(f"{i+1} {x} {y}\n" for i, (x, y) in enumerate(points)) + "EOF\n")
        optimum, optimal_edges, mandatory, optimal_pairs = optimal_tour_facts(points, metric)
        hashes = {}
        variants = [(lp, cache, "1", "1") for lp, cache in
                    itertools.product(("off", "primal-dual-sec"), ("0", "1"))]
        variants += [(lp, "1", pair_cache, "0") for lp, pair_cache in
                     itertools.product(("off", "primal-dual-sec"), ("0", "1"))]
        for lp, cache, pair_cache, full_metric in variants:
            target = directory / f"{lp}-cache{cache}-pair{pair_cache}-metric{full_metric}"
            target.mkdir(exist_ok=True)
            command = [str(exe), "solve", "--profile", "hybrid-e2e", "--instance", str(tsp),
                       "--device", "0", "--lp-backend", lp, "--distance-cache", cache,
                       "--main-pair-cache", pair_cache, "--full-metric", full_metric,
                       "--output-edges", str(target / "out.edg"), "--fixed", str(target / "out.fix"),
                       "--nonpairs", str(target / "out.nonpairs"), "--manifest", str(target / "out.json")]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(f"{case}/{lp}/{cache}: {result.stdout}\n{result.stderr}")
            report = json.loads((target / "out.json").read_text())
            assert report["termination"] == "fixed-point" and report["gpu_replayed"] and not report["proof_rejected"]
            assert report["initial_edges"] == n * (n - 1) // 2
            assert not report["input_optimum_labels"] and report["tour_sha256"] is None
            assert not optimal_edges - edge_set(target / "out.edg"), "误删最优边"
            assert not edge_set(target / "out.fix") - mandatory, "错误固定"
            assert not nonpair_set(target / "out.nonpairs", n) & optimal_pairs, "误删最优邻边对"
            if lp == "off":
                assert report["lp_backend"] == "off" and report["pdhg_iterations"] == 0
                assert report["lp_solver_ms"] == 0 and report["incumbent_starts"] == 0
                nonpair_off += report["point_nonpairs"] + report["fixed_anchor_nonpairs"]
            else:
                assert report["incumbent_origin"] == "gpu-nn-2opt-oropt" and report["incumbent_cost"] >= optimum
                assert report["incumbent_starts"] == min(128, n)
            assert report["main_pair_cache"] == (pair_cache == "1")
            assert report["full_degree_metric"] == (full_metric == "1")
            key = (lp, full_metric)
            if key in hashes:
                assert hashes[key] == report["final_state_hash"], "距离/条件 pair 缓存改变了终态"
            hashes[key] = report["final_state_hash"]
        # 普通非最优 incumbent 可被合法删除，不能因 final nonpair 冲突报错。
        bad_tour = directory / "nonoptimal.tour"
        candidate = next((p for p in itertools.permutations(range(n)) if
                          sum(distance(points[p[i]], points[p[(i+1) % n]], metric) for i in range(n)) > optimum), None)
        if candidate is not None:
            bad_tour.write_text(f"NAME: heuristic\nTYPE: TOUR\nDIMENSION: {n}\nTOUR_SECTION\n" +
                                "\n".join(str(v + 1) for v in candidate) + "\n-1\nEOF\n")
            legacy = command.copy()
            legacy_directory = directory / "legacy-heuristic"
            for flag, filename in (("--output-edges", "out.edg"), ("--fixed", "out.fix"),
                                   ("--nonpairs", "out.nonpairs"), ("--manifest", "out.json")):
                legacy[legacy.index(flag) + 1] = str(legacy_directory / filename)
            legacy[legacy.index("--profile") + 1] = "legacy"
            for flag in ("--distance-cache", "--main-pair-cache", "--full-metric"):
                offset = legacy.index(flag)
                del legacy[offset:offset+2]
            legacy += ["--tour", str(bad_tour), "--tour-role", "incumbent"]
            subprocess.run(legacy, capture_output=True, text=True, check=True)
            # Hybrid 拒绝标签或 tour，不能悄悄退回 legacy 路径。
            rejected = subprocess.run(command + ["--tour", str(bad_tour)], capture_output=True, text=True)
            assert rejected.returncode != 0
        print(f"hybrid case={case} n={n} optimum={optimum}: no-label/all-opt/cache verified", flush=True)
    assert nonpair_off > 0, "LP off 测试没有实际触发 pair 服务"


if __name__ == "__main__":
    main()
