#!/usr/bin/env python3
"""小图 LP 删除后验证 non-pair stable-edge 重映射、确定性与最优 tour 安全。"""

from __future__ import annotations

import itertools
import math
import pathlib
import subprocess
import sys


def coordinates(path: pathlib.Path) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    active = False
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if line == "NODE_COORD_SECTION":
            active = True
        elif active and line and line != "EOF":
            fields = line.split()
            points.append((int(fields[1]), int(fields[2])))
    return points


def distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    squared = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
    root = math.isqrt(squared)
    return root + int(squared - root * root > root)


def optimal_facts(points: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]]]:
    optimum: int | None = None
    edge_union: set[tuple[int, int]] = set()
    pair_union: set[tuple[int, int, int]] = set()
    for suffix in itertools.permutations(range(1, len(points))):
        if suffix[0] > suffix[-1]:
            continue
        tour = (0, *suffix)
        cost = sum(distance(points[tour[index]], points[tour[(index + 1) % len(tour)]])
                   for index in range(len(tour)))
        edges = {tuple(sorted((tour[index], tour[(index + 1) % len(tour)])))
                 for index in range(len(tour))}
        pairs = {(tour[index], *sorted((tour[index - 1], tour[(index + 1) % len(tour)])))
                 for index in range(len(tour))}
        if optimum is None or cost < optimum:
            optimum = cost
            edge_union = edges
            pair_union = pairs
        elif cost == optimum:
            edge_union.update(edges)
            pair_union.update(pairs)
    return edge_union, pair_union


def read_edges(path: pathlib.Path) -> set[tuple[int, int]]:
    return {tuple(sorted((int(row.split()[0]), int(row.split()[1]))))
            for row in path.read_text(encoding="ascii").splitlines()[1:]}


def read_nonpairs(path: pathlib.Path, dimension: int) -> set[tuple[int, int, int]]:
    tokens = iter(path.read_text(encoding="ascii").split())
    if int(next(tokens)) != dimension:
        raise RuntimeError("non-pair 维度错误")
    declared = int(next(tokens))
    result: set[tuple[int, int, int]] = set()
    for expected_center in range(dimension):
        center = int(next(tokens))
        count = int(next(tokens))
        if center != expected_center:
            raise RuntimeError("non-pair 中心顺序错误")
        for _ in range(count):
            first, second = sorted((int(next(tokens)), int(next(tokens))))
            result.add((center, first, second))
    if len(result) != declared:
        raise RuntimeError("non-pair 声明计数与唯一记录不一致")
    return result


def read_manifest(path: pathlib.Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            rows[fields[0]] = fields[1]
    return rows


def run(executable: pathlib.Path, tsp: pathlib.Path, tour: pathlib.Path,
        output: pathlib.Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), "resident", "--instance", str(tsp), "--tour", str(tour),
        "--tour-role", "known-optimum", "--expected-cost", "251", "--device", "0",
        "--enable-geometry", "0", "--enable-pdlp", "1", "--enable-quick-hs", "0",
        "--enable-jv", "0", "--enable-main-edge", "0", "--enable-extra-edge", "0",
        "--pdlp-iterations", "5", "--max-pdlp-epochs", "1", "--max-hs-epochs", "0",
        "--max-jv-rounds", "0", "--cpu-audit", "0", "--protect-tour", "0",
        "--output-edges", str(output / "out.edg"), "--fixed", str(output / "out.fix"),
        "--nonpairs", str(output / "out.nonpairs"), "--manifest", str(output / "out.manifest"),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if "final_edges=10" not in completed.stdout or "nonpairs=1" not in completed.stdout:
        raise RuntimeError(f"LP/non-pair 锁定强度变化: {completed.stdout}")
    manifest = read_manifest(output / "out.manifest")
    if (manifest.get("lp_connectivity_cuts") != "8" or
            manifest.get("lp_strong_snapshots") != "1"):
        raise RuntimeError("connectivity SEC 未进入 strong LP 快照")
    return manifest


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_fgpu_nonpair_persistence.py EXE TSP TOUR OUTPUT_DIR")
    executable, tsp, tour, output = (pathlib.Path(value).resolve() for value in sys.argv[1:])
    points = coordinates(tsp)
    optimal_edges, optimal_pairs = optimal_facts(points)
    first = run(executable, tsp, tour, output / "first")
    second = run(executable, tsp, tour, output / "second")
    if first.get("final_hash") != second.get("final_hash"):
        raise RuntimeError("LP/non-pair 重复运行 hash 不一致")
    final_edges = read_edges(output / "first" / "out.edg")
    nonpairs = read_nonpairs(output / "first" / "out.nonpairs", len(points))
    if not optimal_edges.issubset(final_edges):
        raise RuntimeError("LP 删除了至少一条最优 tour 边")
    invalid_pairs = nonpairs & optimal_pairs
    if invalid_pairs:
        raise RuntimeError(f"LP 排除了最优 tour 邻边对: {sorted(invalid_pairs)}")
    if len(nonpairs) != 1 or first.get("nonpair_count") != "1":
        raise RuntimeError("CSR compact 后 non-pair 数量未保持")
    print("FGPU non-pair persistence check passed; final_edges=10 nonpairs=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
