#!/usr/bin/env python3
"""在 8 点实例上枚举全部最优 tour，检查无保护 one-shot 不删最优边。"""

from __future__ import annotations

import itertools
import math
import pathlib
import subprocess
import sys


def parse_coordinates(path: pathlib.Path) -> list[tuple[int, int]]:
    coordinates: list[tuple[int, int]] = []
    in_section = False
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if line == "NODE_COORD_SECTION":
            in_section = True
            continue
        if not in_section or line in {"EOF", ""}:
            continue
        fields = line.split()
        coordinates.append((int(fields[1]), int(fields[2])))
    return coordinates


def euc_2d(first: tuple[int, int], second: tuple[int, int]) -> int:
    squared = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
    root = math.isqrt(squared)
    return root + int(squared - root * root > root)


def all_optimal_edges(coordinates: list[tuple[int, int]]) -> tuple[int, set[tuple[int, int]]]:
    dimension = len(coordinates)
    distances = [
        [euc_2d(coordinates[u], coordinates[v]) for v in range(dimension)]
        for u in range(dimension)
    ]
    optimum: int | None = None
    edges: set[tuple[int, int]] = set()
    # 固定起点 0，并只保留一种反向，8 点仅需 7!/2 次精确枚举。
    for suffix in itertools.permutations(range(1, dimension)):
        if suffix[0] > suffix[-1]:
            continue
        tour = (0, *suffix)
        cost = sum(distances[tour[index]][tour[(index + 1) % dimension]]
                   for index in range(dimension))
        tour_edges = {
            tuple(sorted((tour[index], tour[(index + 1) % dimension])))
            for index in range(dimension)
        }
        if optimum is None or cost < optimum:
            optimum = cost
            edges = tour_edges
        elif cost == optimum:
            edges.update(tour_edges)
    if optimum is None:
        raise RuntimeError("没有枚举到 tour")
    return optimum, edges


def read_edges(path: pathlib.Path) -> set[tuple[int, int]]:
    rows = path.read_text(encoding="ascii").splitlines()[1:]
    return {tuple(sorted((int(row.split()[0]), int(row.split()[1])))) for row in rows}


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_fgpu_oneshot.py EXE TSP TOUR OUTPUT_DIR")
    executable = pathlib.Path(sys.argv[1]).resolve()
    tsp = pathlib.Path(sys.argv[2]).resolve()
    tour = pathlib.Path(sys.argv[3]).resolve()
    output_dir = pathlib.Path(sys.argv[4]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optimum, optimal_edges = all_optimal_edges(parse_coordinates(tsp))
    if optimum != 251:
        raise RuntimeError(f"锁定实例最优值变化: {optimum}")

    output_edges = output_dir / "recursive-point.oneshot.edg"
    command = [
        str(executable), "resident-oneshot", "--instance", str(tsp), "--tour", str(tour),
        "--tour-role", "known-optimum", "--expected-cost", str(optimum), "--device", "0",
        "--potential-candidates", "6", "--main-edge-potentials", "6",
        "--main-edge-positions", "3", "--quick-hs-candidates", "8",
        "--quick-hs-pair-trials", "0", "--quick-hs-two-hop", "1",
        "--pdlp-iterations", "500", "--max-pdlp-epochs", "0", "--max-hs-epochs", "0",
        "--max-jv-rounds", "0", "--enable-geometry", "1", "--enable-pdlp", "1",
        "--enable-quick-hs", "1", "--enable-jv", "1", "--protect-tour", "0",
        "--output-edges", str(output_edges), "--fixed", str(output_dir / "recursive-point.fix"),
        "--nonpairs", str(output_dir / "recursive-point.nonpairs"),
        "--manifest", str(output_dir / "recursive-point.manifest"),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if "mode=resident-oneshot" not in completed.stdout or "converged=1" not in completed.stdout:
        raise RuntimeError(f"one-shot 未收敛: {completed.stdout} {completed.stderr}")

    complete_edges = {
        (u, v) for u in range(len(parse_coordinates(tsp))) for v in range(u + 1, len(parse_coordinates(tsp)))
    }
    removed = complete_edges - read_edges(output_edges)
    unsafe = removed & optimal_edges
    if unsafe:
        raise RuntimeError(f"one-shot 删除了至少一个最优 tour 使用的边: {sorted(unsafe)}")
    if not removed:
        raise RuntimeError("one-shot 没有触发删除，测试未覆盖强度路径")
    print(f"FGPU one-shot exhaustive optimal-edge check passed; removed={len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
