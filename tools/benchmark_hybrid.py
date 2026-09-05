#!/usr/bin/env python3
"""四实例、无标签求解、进程墙钟与全程资源记录；标签只在退出后校验。"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from benchmark_fgpu import host_identity, inside, sha256
from prepare_hs2014_data import CONFIG, DATA, ROOT, distance, records, validate


def telemetry(uuid):
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    devices = query(["--query-gpu=uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw"])
    processes = query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"])
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu": next(row for row in devices if row.split(",")[0].strip() == uuid),
            "processes": [row for row in processes if row.split(",")[0].strip() == uuid],
            "node_processes": processes, "loadavg": os.getloadavg()}


def check_outputs(directory, item):
    _, points, metric, tour = validate(item)
    return check_outputs_against_tour(directory, points, metric, tour)


def check_outputs_against_tour(directory, points, metric, tour, stem="out"):
    n = len(points)
    def edges(path):
        rows = path.read_text().splitlines()
        dim, count = map(int, rows[0].split())
        if dim != n or count != len(rows) - 1:
            raise ValueError("输出边数量不符")
        result = set()
        for row in rows[1:]:
            a, b, cost = map(int, row.split())
            if not 0 <= a < b < n or cost != distance(points[a], points[b], metric) or (a, b) in result:
                raise ValueError("输出边编号、去重或距离校验失败")
            result.add((a, b))
        return result
    active, fixed = edges(directory / f"{stem}.edg"), edges(directory / f"{stem}.fix")
    optimum_edges = {tuple(sorted((tour[i], tour[(i + 1) % n]))) for i in range(n)}
    optimum_pairs = {(tour[i], *sorted((tour[i - 1], tour[(i + 1) % n]))) for i in range(n)}
    if optimum_edges - active or fixed - optimum_edges:
        raise ValueError("已知最优 tour 与删边或 fixed 冲突")
    tokens = iter(map(int, (directory / f"{stem}.nonpairs").read_text().split()))
    if next(tokens) != n:
        raise ValueError("nonpair 维度不符")
    expected = next(tokens)
    actual = 0
    seen = set()
    for center in range(n):
        if next(tokens) != center:
            raise ValueError("nonpair 行编号不符")
        row_count = next(tokens)
        if row_count < 0:
            raise ValueError("nonpair 行计数为负")
        for _ in range(row_count):
            a, b = next(tokens), next(tokens)
            pair = (center, a, b)
            if not 0 <= a < b < n or center in (a, b) or pair in seen or pair in optimum_pairs:
                raise ValueError("nonpair 越界、重复或最优标签冲突")
            if tuple(sorted((center, a))) not in active or tuple(sorted((center, b))) not in active:
                raise ValueError("nonpair 不属于最终活动图")
            seen.add(pair)
            actual += 1
    if actual != expected or next(tokens, None) is not None:
        raise ValueError("nonpair 计数不符")
    return {"known_optimum_conflicts": 0, "active_edges": len(active), "fixed_edges": len(fixed), "nonpairs": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", nargs="+", default=[x["name"] for x in records()])
    parser.add_argument("--lp-backend", choices=("off", "primal-dual-sec", "sec-dual"), default="off")
    parser.add_argument("--distance-cache", choices=("0", "1"), default="1")
    parser.add_argument("--main-pair-cache", choices=("0", "1"), default="1")
    parser.add_argument("--full-metric", choices=("0", "1"), default="1")
    parser.add_argument("--leaf-permutation-cache", choices=("0", "1"), default="0")
    parser.add_argument("--point-near-first", choices=("0", "1"), default="0")
    parser.add_argument("--point-adaptive-start", choices=("0", "1"), default="0")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--allow-busy", action="store_true", help="只做开发观测，不能用于正式验收")
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("预热非负、完整重复次数为正")
    if args.full_metric == "1" and args.main_pair_cache == "0":
        parser.error("全度数 metric 需要条件 pair cache")
    all_items = {x["name"]: x for x in records()}
    if len(set(args.instances)) != len(args.instances) or any(x not in all_items for x in args.instances):
        parser.error("实例必须来自冻结的四例且不得重复")
    executable, output = inside(args.exe), inside(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name in args.instances:
        validate(all_items[name])
    identity = sha256(executable)
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": args.gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "TMPDIR": str(ROOT / ".tmp"), "CUDA_CACHE_PATH": str(ROOT / ".tmp/cuda-cache"),
                        "PYTHONDONTWRITEBYTECODE": "1"})
    runs = []
    for iteration in range(args.warmups + args.repetitions):
        order = args.instances if iteration % 2 == 0 else list(reversed(args.instances))
        for name in order:
            item = all_items[name]
            if sha256(executable) != identity:
                raise ValueError("可执行文件在基准期间改变")
            before = telemetry(args.gpu_uuid)
            if before["processes"] and not args.allow_busy:
                raise ValueError("所选 GPU 已有任务，拒绝污染正式测量")
            directory = output / name / f"run-{iteration}"
            directory.mkdir(parents=True)
            # 这里有意不传 tour、expected-cost、input-edges；标签只在进程退出后消费。
            command = [str(executable), "solve", "--profile", "hybrid-e2e", "--mode", "gpu-safe",
                       "--instance", str(DATA / f"{name}.tsp"), "--device", "0",
                       "--lp-backend", args.lp_backend, "--distance-cache", args.distance_cache,
                       "--main-pair-cache", args.main_pair_cache, "--full-metric", args.full_metric,
                       "--leaf-permutation-cache", args.leaf_permutation_cache,
                       "--point-near-first", args.point_near_first,
                       "--point-adaptive-start", args.point_adaptive_start,
                       "--output-edges", str(directory / "out.edg"), "--fixed", str(directory / "out.fix"),
                       "--nonpairs", str(directory / "out.nonpairs"), "--manifest", str(directory / "out.json")]
            stop = threading.Event()
            samples, errors = [before], []
            def observe():
                # Event.wait 可即时结束；不阻塞用户交互或求解线程。
                while not stop.wait(1.0):
                    try:
                        samples.append(telemetry(args.gpu_uuid))
                    except Exception as error:
                        errors.append(str(error))
            thread = threading.Thread(target=observe, daemon=True)
            thread.start()
            start = time.perf_counter()
            try:
                with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                    code = process.wait()
                wall = time.perf_counter() - start
            finally:
                stop.set()
                thread.join()
                (directory / "telemetry.json").write_text(json.dumps(samples, indent=2) + "\n")
            try:
                samples.append(telemetry(args.gpu_uuid))
            except Exception as error:
                errors.append(str(error))
            foreign = any(int(row.split(",")[1].strip()) != process.pid for sample in samples for row in sample["processes"])
            # 其他卡训练也会争用 CPU/内存；保留开发样本但不称为独占节点 clean。
            node_busy = any(int(row.split(",")[1].strip()) != process.pid for sample in samples for row in sample["node_processes"])
            record = {"instance": name, "iteration": iteration, "warmup": iteration < args.warmups,
                      "host_identity": host_identity(),
                      "command": command, "executable_sha256": identity, "config_sha256": sha256(CONFIG),
                      "returncode": code, "process_wall_seconds": wall, "telemetry_errors": errors,
                      "foreign_gpu_process": foreign, "other_node_gpu_process": node_busy,
                      "clean": not (foreign or node_busy or errors or args.allow_busy)}
            (directory / "telemetry.json").write_text(json.dumps(samples, indent=2) + "\n")
            runs.append(record)
            (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if code != 0:
                raise RuntimeError(f"{name} 完整 solve 失败，日志保留于 {directory}")
            report = json.loads((directory / "out.json").read_text())
            if report.get("gpu_identity", {}).get("uuid") != args.gpu_uuid or \
                    report.get("build_identity", {}).get("executable_sha256") != identity:
                raise ValueError("求解的 GPU/可执行文件身份不符")
            if report["profile"] != "hybrid-e2e" or report["initial_edges"] != item["n"] * (item["n"] - 1) // 2 or report["tour_sha256"] is not None:
                raise ValueError("求解不是无标签完整图入口")
            if report["termination"] != "fixed-point" or not report["gpu_replayed"] or report["proof_rejected"]:
                raise ValueError("求解未达到精确 GPU 不动点")
            record["report"] = report
            record["postcheck"] = check_outputs(directory, item)
            if sha256(executable) != identity:
                raise ValueError("运行期间可执行文件被修改")
            (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            print(f'{name} run={iteration} wall={wall:.3f}s edges={report["final_edges"]} fixed={report["fixed_edges"]} clean={record["clean"]}', flush=True)
    summary = {"complete_acceptance": False, "same_machine_reference": "not-yet-measured",
               "configuration": {"lp_backend": args.lp_backend, "distance_cache": args.distance_cache,
                                 "main_pair_cache": args.main_pair_cache, "full_metric": args.full_metric,
                                 "leaf_permutation_cache": args.leaf_permutation_cache,
                                 "point_near_first": args.point_near_first,
                                 "point_adaptive_start": args.point_adaptive_start}, "instances": {}}
    for name in args.instances:
        measured = [x for x in runs if x["instance"] == name and not x["warmup"]]
        clean = [x for x in measured if x["clean"]]
        # 污染样本仅显示 pilot 指标，不混入正式中位数。
        pilot = statistics.median(x["process_wall_seconds"] for x in measured)
        hashes = {x["report"]["final_state_hash"] for x in measured}
        if len(hashes) != 1:
            raise ValueError(f"{name} 重复运行终态不同")
        remaining = measured[0]["report"]["final_edges"]
        item = all_items[name]
        summary["instances"][name] = {"pilot_median_seconds": pilot, "clean_runs": len(clean),
            "formal_median_seconds": statistics.median(x["process_wall_seconds"] for x in clean) if len(clean) >= 3 else None,
            "remaining_edges": remaining, "paper_edges": item["paper_edges"], "edge_gap": remaining - item["paper_edges"],
            "pilot_paper_time_ratio": item["paper_seconds"] / pilot,
            "strict_paper_strength_pass": remaining < item["paper_edges"], "same_machine_speed_pass": None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
