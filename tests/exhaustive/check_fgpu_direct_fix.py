#!/usr/bin/env python3
"""在 pr299 正式路径上验证 endpoint-product fixing 与独立 GPU replay。"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: check_fgpu_direct_fix.py EXE TSP EDGES TOUR OUTPUT_DIR"
        )
    executable, tsp, edges, tour, output = map(
        lambda value: pathlib.Path(value).resolve(), sys.argv[1:]
    )
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable), "solve", "--instance", str(tsp),
        "--input-edges", str(edges), "--tour", str(tour),
        "--tour-role", "known-optimum", "--expected-cost", "48191",
        "--mode", "gpu-safe", "--device", "auto",
        "--output-edges", str(output / "out.edg"),
        "--fixed", str(output / "out.fix"),
        "--nonpairs", str(output / "out.nonpairs"),
        "--manifest", str(output / "out.json"),
    ]
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    manifest = json.loads((output / "out.json").read_text(encoding="utf-8"))
    if "termination=fixed-point" not in completed.stdout:
        raise RuntimeError("Direct Fix 回归未到达联合不动点")
    if int(manifest["direct_fixed_edges"]) <= 0:
        raise RuntimeError("pr299 未实际触发 endpoint-product fixing")
    if int(manifest["proof_rejected"]) != 0 or not bool(manifest["gpu_replayed"]):
        raise RuntimeError("Direct Fix 未通过独立 GPU replay")
    # Opt33 补齐作者的四组必要门禁后，安全基线为 828/38；旧的
    # 797/35 来自绕过门禁的 path-only 谓词，不能再作为正确性阈值。
    if int(manifest["final_edges"]) > 828 or int(manifest["fixed_edges"]) < 38:
        raise RuntimeError("Direct Fix 回归的强度低于已记录基线")
    print(
        "FGPU Direct Fix check passed; "
        f"direct={manifest['direct_fixed_edges']} "
        f"edges={manifest['final_edges']} fixed={manifest['fixed_edges']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
