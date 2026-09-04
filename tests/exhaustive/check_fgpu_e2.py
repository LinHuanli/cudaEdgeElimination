#!/usr/bin/env python3
"""在 10 点完整图上穷举最优 tour，验证纯 GPU `-e2` 固定点。"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from check_fgpu_solve import coordinates, edge_set, optimal_tour_facts


def parse_manifest(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line or line == "END":
            continue
        fields = line.split(maxsplit=1)
        result[fields[0]] = fields[1] if len(fields) == 2 else ""
    return result


def run(executable: pathlib.Path, tsp: pathlib.Path, tour: pathlib.Path,
        output: pathlib.Path, optimum: int) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), "resident", "--instance", str(tsp), "--tour", str(tour),
        "--tour-role", "known-optimum", "--expected-cost", str(optimum),
        "--device", "-1", "--enable-pdlp", "0", "--enable-geometry", "0",
        "--enable-jv", "0", "--enable-quick-hs", "0", "--enable-main-edge", "0",
        "--enable-extra-edge", "1", "--extra-edge-depth", "2", "--protect-tour", "0",
        "--output-edges", str(output / "out.edg"), "--fixed", str(output / "out.fix"),
        "--nonpairs", str(output / "out.nonpairs"),
        "--manifest", str(output / "out.manifest"),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if "converged=1" not in completed.stdout or "extra_edge_depth=2" not in completed.stdout:
        raise RuntimeError(f"纯 -e2 未到达固定点: {completed.stdout}")
    return parse_manifest(output / "out.manifest")


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_fgpu_e2.py EXE TSP TOUR OUTPUT_DIR")
    executable, tsp, tour, output = map(
        lambda value: pathlib.Path(value).resolve(), sys.argv[1:]
    )
    optimum, required, _, _ = optimal_tour_facts(coordinates(tsp))
    first = run(executable, tsp, tour, output / "first", optimum)
    second = run(executable, tsp, tour, output / "second", optimum)
    if int(first["extra_edge_committed"]) <= 0:
        raise RuntimeError("10 点回归没有实际触发 -e2 stage 删除")
    if first["final_state_hash"] != second["final_state_hash"]:
        raise RuntimeError("纯 -e2 重复运行的完整状态 hash 不一致")
    for filename in ("out.edg", "out.fix", "out.nonpairs"):
        if (output / "first" / filename).read_bytes() != (output / "second" / filename).read_bytes():
            raise RuntimeError(f"纯 -e2 重复运行的 {filename} 不一致")
    missing = required - edge_set(output / "first" / "out.edg")
    if missing:
        raise RuntimeError(f"纯 -e2 删除了某个最优 tour 边: {sorted(missing)}")
    print(
        "FGPU e2 exhaustive check passed; "
        f"edges={first['final_edges']} hash={first['final_state_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
