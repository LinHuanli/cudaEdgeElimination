#!/usr/bin/env python3
"""对 n=6..12 的确定性实例求出所有最优 tour 的边并检查消元安全性。"""

from __future__ import annotations

import math
import pathlib
import random
import subprocess
import sys


def euc_2d(a: tuple[int, int], b: tuple[int, int]) -> int:
    squared = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    root = math.isqrt(squared)
    return root + int(squared - root * root > root)


def all_optimal_edges(distances: list[list[int]]) -> tuple[int, set[tuple[int, int]]]:
    """Held–Karp 动态规划并回溯所有最优状态 DAG 的边并集。"""
    n = len(distances)
    dp: dict[tuple[int, int], int] = {}
    predecessors: dict[tuple[int, int], list[int]] = {}
    for last in range(1, n):
        mask = 1 << last
        dp[mask, last] = distances[0][last]
        predecessors[mask, last] = [0]

    for subset_size in range(2, n):
        for mask in range(1 << n):
            if mask & 1 or mask.bit_count() != subset_size:
                continue
            for last in range(1, n):
                if not (mask & (1 << last)):
                    continue
                previous_mask = mask ^ (1 << last)
                best = None
                best_previous: list[int] = []
                for previous in range(1, n):
                    if not (previous_mask & (1 << previous)):
                        continue
                    value = dp[previous_mask, previous] + distances[previous][last]
                    if best is None or value < best:
                        best = value
                        best_previous = [previous]
                    elif value == best:
                        best_previous.append(previous)
                if best is not None:
                    dp[mask, last] = best
                    predecessors[mask, last] = best_previous

    full_mask = ((1 << n) - 1) ^ 1
    optimum = min(dp[full_mask, last] + distances[last][0] for last in range(1, n))
    optimal_edges: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()

    def backtrack(mask: int, last: int) -> None:
        state = (mask, last)
        if state in visited:
            return
        visited.add(state)
        for previous in predecessors[state]:
            optimal_edges.add(tuple(sorted((previous, last))))
            if previous != 0:
                backtrack(mask ^ (1 << last), previous)

    for last in range(1, n):
        if dp[full_mask, last] + distances[last][0] == optimum:
            optimal_edges.add((0, last))
            backtrack(full_mask, last)
    return optimum, optimal_edges


def write_instance(directory: pathlib.Path, n: int) -> tuple[pathlib.Path, pathlib.Path,
                                                              set[tuple[int, int]]]:
    generator = random.Random(0xC0DA0000 + n)
    coordinates: list[tuple[int, int]] = []
    while len(coordinates) < n:
        point = (generator.randrange(0, 2001), generator.randrange(0, 2001))
        if point not in coordinates:
            coordinates.append(point)
    distances = [[euc_2d(coordinates[u], coordinates[v]) for v in range(n)] for u in range(n)]
    optimum, optimal_edges = all_optimal_edges(distances)

    tsp = directory / f"small-{n}.tsp"
    tsp.write_text(
        "\n".join([
            f"NAME : cudaee-small-{n}", "TYPE : TSP", f"COMMENT : exact optimum {optimum}",
            f"DIMENSION : {n}", "EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION",
            *[f"{index + 1} {point[0]} {point[1]}" for index, point in enumerate(coordinates)],
            "EOF", "",
        ]),
        encoding="ascii",
    )
    edges = directory / f"small-{n}.edg"
    edge_rows = [f"{u} {v} {distances[u][v]}" for u in range(n) for v in range(u + 1, n)]
    edges.write_text(f"{n} {len(edge_rows)}\n" + "\n".join(edge_rows) + "\n", encoding="ascii")
    return tsp, edges, optimal_edges


def read_edges(path: pathlib.Path) -> set[tuple[int, int]]:
    lines = path.read_text(encoding="ascii").splitlines()
    return {(int(row.split()[0]), int(row.split()[1])) for row in lines[1:]}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_small_instances.py CUDAEE OUTPUT_DIR")
    executable = pathlib.Path(sys.argv[1]).resolve()
    output_dir = pathlib.Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_removed = 0
    first_nonempty_proof: pathlib.Path | None = None
    first_tsp: pathlib.Path | None = None
    first_edges: pathlib.Path | None = None

    for n in range(6, 13):
        tsp, original, optimal_edges = write_instance(output_dir, n)
        filtered = output_dir / f"small-{n}.filtered.edg"
        proof = output_dir / f"small-{n}.proof"
        subprocess.run(
            [str(executable), "gpu-eliminate", "--tsp", str(tsp), "--edges", str(original),
             "--output", str(filtered), "--proof", str(proof), "--backend", "cpu",
             "--max-rounds", "100"],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            [str(executable), "verify", "--tsp", str(tsp), "--edges", str(original),
             "--proof", str(proof)],
            check=True,
            text=True,
            capture_output=True,
        )
        removed = read_edges(original) - read_edges(filtered)
        unsafe = removed & optimal_edges
        if unsafe:
            raise RuntimeError(f"n={n} 删除了某个最优 tour 的边: {sorted(unsafe)}")
        total_removed += len(removed)
        if removed and first_nonempty_proof is None:
            first_nonempty_proof, first_tsp, first_edges = proof, tsp, original

    if total_removed == 0:
        raise RuntimeError("测试实例没有触发任何删除，无法覆盖安全断言")

    # 篡改首条见证为边端点，验证器必须 fail closed。
    if first_nonempty_proof is not None and first_tsp is not None and first_edges is not None:
        lines = first_nonempty_proof.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("record "):
                fields = line.split()
                fields[-1] = fields[4]
                lines[index] = " ".join(fields)
                break
        tampered = output_dir / "tampered.proof"
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [str(executable), "verify", "--tsp", str(first_tsp), "--edges", str(first_edges),
             "--proof", str(tampered)],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            raise RuntimeError("验证器错误接受了篡改证明")

    print(f"n=6..12 exact optimal-edge checks passed; removed={total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
