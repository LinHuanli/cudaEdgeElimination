#!/usr/bin/env python3
"""锁定作者单核、完整三步骤、无最优标签输入的同机对照。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from benchmark_fgpu import inside, sha256
from benchmark_hybrid import telemetry
from prepare_hs2014_data import CONFIG, DATA, ROOT, records, validate


def postcheck(path, item):
    _, _, _, tour = validate(item)
    rows = path.read_text().splitlines()
    n, count = map(int, rows[0].split())
    edges = {tuple(map(int, row.split())) for row in rows[1:]}
    if n != item["n"] or len(edges) != count or len(rows) != count + 1:
        raise ValueError("作者输出维度/边计数不符")
    if any(len(edge) != 2 or not 0 <= edge[0] < edge[1] < n for edge in edges):
        raise ValueError("作者输出边编号非法")
    optimum_edges = {tuple(sorted((tour[i], tour[(i + 1) % n]))) for i in range(n)}
    if optimum_edges - edges:
        raise ValueError("作者输出与已知最优 tour 冲突")
    return {"known_optimum_conflicts": 0, "active_edges": count,
            "edge_set_sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=ROOT / "build/hs2014-reference/hs2014-serial")
    parser.add_argument("--identity", type=Path, default=ROOT / "build/hs2014-reference/identity.json")
    parser.add_argument("--differential", type=Path,
                        default=ROOT / "artifacts/hs2014-reference-differential/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True, help="记录同一节点资源，不使用此 GPU 求解")
    parser.add_argument("--instances", nargs="+", default=[x["name"] for x in records()])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    if args.cpu_core not in os.sched_getaffinity(0):
        parser.error("CPU 核不在当前进程允许的 affinity 中")
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("预热非负、完整重复次数为正")
    items = {x["name"]: x for x in records()}
    if len(set(args.instances)) != len(args.instances) or any(x not in items for x in args.instances):
        parser.error("实例必须来自冻结的四例且不重复")
    executable, output = inside(args.exe), inside(args.output)
    identity = json.loads(inside(args.identity).read_text())
    differential = json.loads(inside(args.differential).read_text())
    options = ROOT / "configs/hs2014_full.options"
    executable_hash = sha256(executable)
    if (identity["executables"]["serial"] != executable_hash or
            differential["executables"]["serial"] != executable_hash or
            differential["configuration_sha256"] != sha256(options) or
            len(differential["cases"]) < 2 or not all(x["match"] for x in differential["cases"])):
        raise ValueError("当前作者可执行文件/配置未通过串行与单 worker 差分")
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "differential.json").write_text(json.dumps(differential, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update({"TMPDIR": str(ROOT / ".tmp"), "PYTHONDONTWRITEBYTECODE": "1"})
    runs = []
    for iteration in range(args.warmups + args.repetitions):
        order = args.instances if iteration % 2 == 0 else list(reversed(args.instances))
        for name in order:
            validate(items[name])
            if sha256(executable) != executable_hash:
                raise ValueError("作者可执行文件在基准期间发生改变")
            before = telemetry(args.gpu_uuid)
            if before["node_processes"] and not args.allow_busy:
                raise ValueError("节点上存在 GPU 作业，拒绝污染同机对照")
            directory = output / name / f"run-{iteration}"
            directory.mkdir(parents=True)
            command = ["taskset", "-c", str(args.cpu_core), str(executable),
                       str(DATA / f"{name}.tsp"), str(options), str(directory)]
            samples, errors = [before], []
            stop = threading.Event()

            def observe():
                while not stop.wait(1.0):
                    try:
                        samples.append(telemetry(args.gpu_uuid))
                    except Exception as error:
                        errors.append(str(error))

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            start = time.perf_counter()
            try:
                with (directory / "stdout.log").open("x") as log:
                    code = subprocess.run(command, cwd=ROOT, env=environment,
                                          stdout=log, stderr=subprocess.STDOUT, check=False).returncode
                wall = time.perf_counter() - start
            finally:
                stop.set()
                observer.join()
                (directory / "telemetry.json").write_text(json.dumps(samples, indent=2) + "\n")
            record = {"instance": name, "iteration": iteration, "warmup": iteration < args.warmups,
                      "command": command, "cpu_core": args.cpu_core, "returncode": code,
                      "process_wall_seconds": wall, "executable_sha256": executable_hash,
                      "config_sha256": sha256(CONFIG), "options_sha256": sha256(options),
                      "instance_sha256": items[name]["tsp_sha256"],
                      "differential_validated": True, "telemetry_errors": errors,
                      "clean": not (args.allow_busy or errors or any(x["node_processes"] for x in samples))}
            runs.append(record)
            (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if code != 0:
                raise RuntimeError(f"作者全量运行失败: {directory}")
            record["postcheck"] = postcheck(directory / f"{name}_final.edges", items[name])
            if sha256(executable) != executable_hash:
                raise ValueError("作者可执行文件在运行期间改变")
            (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            print(f'{name} serial run={iteration} wall={wall:.3f}s '
                  f'edges={record["postcheck"]["active_edges"]} clean={record["clean"]}', flush=True)


if __name__ == "__main__":
    main()
