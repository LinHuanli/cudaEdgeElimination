#!/usr/bin/env python3
"""独立穷举全部最优解；三个 Opt34 后端必须给出相同完整终态。"""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

from check_fgpu_solve import distance, edge_set, nonpair_set, optimal_tour_facts, run


def main() -> None:
    executable, output = (Path(arg).resolve() for arg in sys.argv[1:3])
    root = Path(__file__).resolve().parents[2]
    if output == root or not output.is_relative_to(root):
        raise ValueError("穷举测试输出必须在项目内")
    generator = random.Random(0xF6A34)
    fixtures = [[(x * 10, y * 10) for x in range(4) for y in range(2)]]
    fixtures += [[(generator.randrange(80), generator.randrange(80)) for _ in range(6 + i % 3)]
                 for i in range(5)]
    # 两条等成本 tour 的边并集为 7、必选边交集为 3，不能只保留 incumbent。
    fixtures.append([(0, 5), (9, 3), (5, 2), (5, 11), (1, 5)])
    backends = ("permutation", "prescreen-permutation", "prescreen-subset-dp")
    for index, points in enumerate(fixtures):
        directory = output / f"case-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        tsp, edges, tour = (directory / name for name in ("input.tsp", "input.edg", "input.tour"))
        count = len(points)
        optimum, optimal_edges, mandatory, optimal_pairs = optimal_tour_facts(points)
        tsp.write_text(f"NAME: exhaustive-{index}\nTYPE: TSP\nDIMENSION: {count}\n"
                       "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n" +
                       "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(points)) + "EOF\n")
        edges.write_text(f"{count} {count * (count - 1) // 2}\n" +
                         "".join(f"{a} {b} {distance(points[a], points[b])}\n"
                                 for a in range(count) for b in range(a + 1, count)))
        # 仅测试代码在 CPU 枚举 incumbent；正式主链不调用任何 CPU oracle。
        for suffix in itertools.permutations(range(1, count)):
            candidate = (0, *suffix)
            cost = sum(distance(points[candidate[i]], points[candidate[(i + 1) % count]])
                       for i in range(count))
            if cost == optimum:
                tour.write_text(f"NAME: optimum-{index}\nTYPE: TOUR\nDIMENSION: {count}\n"
                                "TOUR_SECTION\n" +
                                "\n".join(str(v + 1) for v in candidate) + "\n-1\nEOF\n")
                break
        expected_hash = None
        for backend, cta in itertools.product(backends, (2, 4)):
            target = directory / f"{backend}-cta{cta}"
            report = run(executable, tsp, edges, tour, target, optimum,
                         ("--point-leaf-kernel", backend, "--point-cta-blocks", str(cta)))
            if optimal_edges - edge_set(target / "out.edg"):
                raise RuntimeError(f"{index}/{backend}: 最优 tour 边被误删")
            if edge_set(target / "out.fix") - mandatory:
                raise RuntimeError(f"{index}/{backend}: 非必选边被错误固定")
            if nonpair_set(target / "out.nonpairs", count) & optimal_pairs:
                raise RuntimeError(f"{index}/{backend}: 最优 tour 的邻边对被排除")
            if report["proof_rejected"] != 0 or not report["gpu_replayed"]:
                raise RuntimeError("GPU replay 不通过")
            if expected_hash is not None and report["final_state_hash"] != expected_hash:
                raise RuntimeError("等价 Opt34 后端的终态不一致")
            if report["point_leaf_kernel"] != backend or report["point_path_end_branches"] != 4:
                raise RuntimeError("测试没有使用请求的后端与完整端点 OR")
            if report["point_cta_blocks"] != cta or report["point_active_blocks_per_sm"] < cta:
                raise RuntimeError("测试没有使用请求的 CTA 驻留策略")
            expected_hash = report["final_state_hash"]
        print(f"case={index} n={count} optimum={optimum} all-optima/6-variants verified", flush=True)


if __name__ == "__main__":
    main()
