#!/usr/bin/env python3
"""作者串行适配与真实单 worker MPI 路径差分，不是大实例速度验收。"""

import json
import os
from pathlib import Path
import subprocess

from benchmark_fgpu import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = ROOT / "build/hs2014-reference"
    output = ROOT / "artifacts/hs2014-reference-differential"
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"TMPDIR": str(ROOT / ".tmp"), "PRTE_MCA_prte_tmpdir_base": str(ROOT / ".tmp"),
                "OMPI_MCA_orte_tmpdir_base": str(ROOT / ".tmp"), "PRTE_MCA_prte_silence_shared_fs": "1"})
    instances = [ROOT / "tests/data/recursive-point.tsp",
                 ROOT.parent / "references/EdgeElimination/instances/berlin52/berlin52.tsp"]
    results = []
    for tsp in instances:
        for mode in ("serial", "mpi"):
            directory = output / tsp.stem / mode
            directory.mkdir(parents=True)
            common = [str(tsp), str(ROOT / "configs/hs2014_full.options"), str(directory)]
            command = ([str(build / "hs2014-serial"), *common] if mode == "serial" else
                       ["mpirun", "-np", "2", "--bind-to", "none", str(build / "hs2014-mpi"), *common,
                        "nofile", "nofile", "nofile", "nofile"])
            with (directory / "stdout.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        def state(mode):
            path = next((output / tsp.stem / mode).glob("*_final.edges"))
            rows = path.read_text().splitlines()
            n, count = map(int, rows[0].split())
            edges = {tuple(map(int, row.split())) for row in rows[1:]}
            if len(edges) != count:
                raise ValueError("作者边数不匹配")
            return n, edges
        serial, mpi = state("serial"), state("mpi")
        if serial != mpi:
            raise ValueError(f"{tsp.stem} 串行与单 worker 最终边集不同")
        results.append({"instance": tsp.stem, "n": serial[0], "edges": len(serial[1]), "match": True})
        print(f'{tsp.stem}: single-worker/serial identical, edges={len(serial[1])}', flush=True)
    record = {"cases": results, "configuration_sha256": sha256(ROOT / "configs/hs2014_full.options"),
              "executables": {mode: sha256(build / f"hs2014-{mode}") for mode in ("serial", "mpi")}}
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
