#!/usr/bin/env python3
"""在同一 pr299 输入上比较 CPU 与 CUDA 的已验证最终图。"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def run(backend: str, executable: pathlib.Path, tsp: pathlib.Path, edges: pathlib.Path,
        output_dir: pathlib.Path) -> str:
    edge_output = output_dir / f"pr299.{backend}.edg"
    proof_output = output_dir / f"pr299.{backend}.proof"
    completed = subprocess.run(
        [str(executable), "gpu-eliminate", "--tsp", str(tsp), "--edges", str(edges),
         "--output", str(edge_output), "--proof", str(proof_output), "--backend", backend,
         "--max-rounds", "100"],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"final_hash=([0-9a-f]{16})", completed.stdout)
    if match is None:
        raise RuntimeError(f"{backend} 输出缺少 final_hash: {completed.stdout}")
    return match.group(1)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: compare_backends.py CUDAEE TSP EDGES OUTPUT_DIR")
    executable, tsp, edges, output_dir = map(pathlib.Path, sys.argv[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_hash = run("cpu", executable, tsp, edges, output_dir)
    cuda_hash = run("cuda", executable, tsp, edges, output_dir)
    if cpu_hash != cuda_hash:
        raise RuntimeError(f"CPU/GPU final hash mismatch: {cpu_hash} != {cuda_hash}")
    print(f"CPU/GPU verified final hash: {cpu_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
