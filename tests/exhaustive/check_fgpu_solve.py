#!/usr/bin/env python3
"""穷举小图全部最优 tour，验证 gpu-safe 主命令和确定性。"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import subprocess
import sys


def coordinates(path: pathlib.Path) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    active = False
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if line == "NODE_COORD_SECTION":
            active = True
            continue
        if active and line and line != "EOF":
            fields = line.split()
            result.append((int(fields[1]), int(fields[2])))
    return result


def distance(first: tuple[int, int], second: tuple[int, int], metric: str = "EUC_2D") -> int:
    # 整数/半整数的独立 oracle；先精确放大坐标差，最后才做 TSPLIB 舍入。
    dx, dy = 2 * (first[0] - second[0]), 2 * (first[1] - second[1])
    if dx != int(dx) or dy != int(dy):
        raise ValueError("测试 oracle 仅支持整数或半整数坐标")
    squared = int(dx) ** 2 + int(dy) ** 2
    root = math.isqrt(squared)
    if metric == "CEIL_2D":
        return root // 2 + int(squared != (root // 2 * 2) ** 2)
    if metric != "EUC_2D":
        raise ValueError("测试 oracle 不支持该距离类型")
    return (root + 1) // 2


def optimal_tour_facts(
    points: list[tuple[int, int]], metric: str = "EUC_2D",
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int, int]]]:
    optimum: int | None = None
    union: set[tuple[int, int]] = set()
    intersection: set[tuple[int, int]] = set()
    pair_union: set[tuple[int, int, int]] = set()
    for suffix in itertools.permutations(range(1, len(points))):
        if suffix[0] > suffix[-1]:
            continue
        tour = (0, *suffix)
        cost = sum(distance(points[tour[i]], points[tour[(i + 1) % len(tour)]], metric)
                   for i in range(len(tour)))
        edges = {tuple(sorted((tour[i], tour[(i + 1) % len(tour)])))
                 for i in range(len(tour))}
        pairs = {
            (tour[i], *sorted((tour[i - 1], tour[(i + 1) % len(tour)])))
            for i in range(len(tour))
        }
        if optimum is None or cost < optimum:
            optimum = cost
            # 并集与交集必须持有独立集合；共享同一对象会让 intersection_update
            # 悄悄缩小 union，把多最优解测试退化成只保留最后一条 tour。
            union = set(edges)
            intersection = set(edges)
            pair_union = set(pairs)
        elif cost == optimum:
            union.update(edges)
            intersection.intersection_update(edges)
            pair_union.update(pairs)
    if optimum is None:
        raise RuntimeError("未找到 Hamilton tour")
    return optimum, union, intersection, pair_union


def edge_set(path: pathlib.Path) -> set[tuple[int, int]]:
    lines = path.read_text(encoding="ascii").splitlines()
    return {tuple(sorted((int(line.split()[0]), int(line.split()[1]))))
            for line in lines[1:]}


def nonpair_set(path: pathlib.Path, dimension: int) -> set[tuple[int, int, int]]:
    tokens = iter(path.read_text(encoding="ascii").split())
    file_dimension = int(next(tokens))
    declared = int(next(tokens))
    if file_dimension != dimension:
        raise RuntimeError("nonpair 文件维度错误")
    result: set[tuple[int, int, int]] = set()
    for expected_center in range(dimension):
        center = int(next(tokens))
        count = int(next(tokens))
        if center != expected_center:
            raise RuntimeError("nonpair 文件中心顺序错误")
        for _ in range(count):
            first, second = sorted((int(next(tokens)), int(next(tokens))))
            result.add((center, first, second))
    if len(result) != declared:
        raise RuntimeError("nonpair 文件计数或去重错误")
    try:
        next(tokens)
    except StopIteration:
        return result
    raise RuntimeError("nonpair 文件含多余字段")


def run(executable: pathlib.Path, tsp: pathlib.Path, edge_input: pathlib.Path,
        tour: pathlib.Path, output: pathlib.Path, optimum: int,
        extra_args: tuple[str, ...] = ()) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), "solve", "--instance", str(tsp), "--input-edges", str(edge_input),
        "--tour", str(tour), "--tour-role", "known-optimum", "--expected-cost", str(optimum),
        "--mode", "gpu-safe", "--device", "auto", "--output-edges", str(output / "out.edg"),
        "--fixed", str(output / "out.fix"), "--nonpairs", str(output / "out.nonpairs"),
        "--manifest", str(output / "out.json"), *extra_args,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"solve failed ({completed.returncode}): {completed.stderr}\n{completed.stdout}")
    if "termination=fixed-point" not in completed.stdout or "gpu_replayed=1" not in completed.stdout:
        raise RuntimeError(f"solve 未进入 GPU-safe 不动点: {completed.stdout}")
    return json.loads((output / "out.json").read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) < 6:
        raise SystemExit(
            "usage: check_fgpu_solve.py EXE TSP EDGES TOUR OUTPUT_DIR [REQUIRED_COUNTER ...]"
        )
    executable, tsp, edges, tour, output = map(
        lambda value: pathlib.Path(value).resolve(), sys.argv[1:6]
    )
    required_counters = sys.argv[6:]
    points = coordinates(tsp)
    optimum, required, mandatory, optimal_pairs = optimal_tour_facts(points)
    first = run(executable, tsp, edges, tour, output / "first", optimum)
    second = run(executable, tsp, edges, tour, output / "second", optimum)
    if first["final_state_hash"] != second["final_state_hash"]:
        raise RuntimeError("gpu-safe 重复运行的完整状态 hash 不一致")
    for filename in ("out.edg", "out.fix", "out.nonpairs"):
        if (output / "first" / filename).read_bytes() != (output / "second" / filename).read_bytes():
            raise RuntimeError(f"gpu-safe 重复运行的 {filename} 不一致")
    if int(first["proof_rejected"]) != 0 or not bool(first["gpu_replayed"]):
        raise RuntimeError("manifest 的 GPU replay 状态非法")
    for required_counter in required_counters:
        if int(first.get(required_counter, 0)) <= 0:
            raise RuntimeError(f"正式回归未实际触发 {required_counter}")
    missing = required - edge_set(output / "first" / "out.edg")
    if missing:
        raise RuntimeError(f"gpu-safe 删除了某个最优 tour 边: {sorted(missing)}")
    invalid_fixed = edge_set(output / "first" / "out.fix") - mandatory
    if invalid_fixed:
        raise RuntimeError(f"gpu-safe 错误固定了非必选边: {sorted(invalid_fixed)}")
    invalid_nonpairs = nonpair_set(output / "first" / "out.nonpairs", len(points)) & optimal_pairs
    if invalid_nonpairs:
        raise RuntimeError(f"gpu-safe 排除了某个最优 tour 邻边对: {sorted(invalid_nonpairs)}")
    if int(first["nonpairs"]) != len(nonpair_set(output / "first" / "out.nonpairs", len(points))):
        raise RuntimeError("manifest 与 nonpair 文件计数不一致")
    print("FGPU solve exhaustive check passed; "
          f"final_state_hash={first['final_state_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
