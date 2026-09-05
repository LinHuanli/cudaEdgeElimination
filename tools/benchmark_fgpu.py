#!/usr/bin/env python3
"""同一单 GPU 上交错运行完整 solve，记录真实子进程墙钟与可复现身份。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def inside(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError(f"输出必须在项目目录内: {path}")
    return resolved


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unchanged(paths: dict[str, Path], expected: dict[str, str]) -> None:
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise RuntimeError(f"计时期间文件发生变化，停止比较: {name}: {path}")


def gpu_snapshot(uuid: str) -> dict:
    fields = "uuid,name,memory.used,memory.total,utilization.gpu"
    devices = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True
    )
    device = next((line for line in devices.splitlines() if line.split(",")[0].strip() == uuid), None)
    if device is None:
        raise ValueError(f"找不到 GPU {uuid}")
    return {"device": device, "processes": [line for line in processes.splitlines()
                                             if line.split(",")[0].strip() == uuid]}


def variant_source_root(executable: Path) -> Path:
    cache = executable.parent / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                source = Path(line.split("=", 1)[1]).resolve()
                if source != ROOT and not source.is_relative_to(ROOT):
                    raise ValueError("对照构建的源码目录不在项目内")
                return source
    return ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=EXE")
    parser.add_argument("--variant-args", action="append", default=[], metavar="NAME=JSON_ARRAY")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--input-edges", type=Path)
    parser.add_argument("--tour", type=Path, required=True)
    parser.add_argument("--expected-cost", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--allow-busy", action="store_true", help="只做开发观测，不标记为 clean")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("重复次数必须为正，预热次数不能为负")
    output = inside(args.output_root)
    variants: dict[str, Path] = {}
    for item in args.variant:
        name, executable = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in variants:
            raise ValueError(f"无效或重复的 variant 名称: {name}")
        variants[name] = inside(Path(executable))
    extra = {name: [] for name in variants}
    for item in args.variant_args:
        name, values = item.split("=", 1)
        parsed = json.loads(values)
        if name not in variants or not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("variant-args 必须为已声明 variant 的字符串数组")
        # 变体不得覆盖完整目标、输入、设备或输出，只允许算法/执行后端消融。
        allowed = {"--lp-backend", "--point-leaf-kernel", "--point-cta-blocks"}
        if len(parsed) % 2 or any(parsed[i] not in allowed for i in range(0, len(parsed), 2)):
            raise ValueError("variant-args 只能切换已声明的算法后端")
        extra[name] = parsed
    identity = {name: sha256(exe) for name, exe in variants.items()}
    inputs = {"instance": args.instance.resolve(), "tour": args.tour.resolve()}
    if args.input_edges is not None:
        inputs["input_edges"] = args.input_edges.resolve()
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TMPDIR": str(ROOT / ".tmp"), "CUDA_CACHE_PATH": str(ROOT / ".tmp/cuda-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    records: list[dict] = []
    run_tag = output.name + "-" + hashlib.sha256(str(output).encode()).hexdigest()[:10]
    for repetition in range(args.warmups + args.repetitions):
        # AB / BA 交替，降低温度、时钟或节点负载漂移的顺序偏差。
        order = list(variants)
        if repetition % 2:
            order.reverse()
        for name in order:
            executable = variants[name]
            unchanged({name: executable}, identity)
            unchanged(inputs, input_hashes)
            before = gpu_snapshot(args.gpu_uuid)
            if before["processes"] and not args.allow_busy:
                raise RuntimeError("所选 GPU 有其他计算进程；不把共享卡计时当 clean A/B")
            source_root = variant_source_root(executable)
            run_directory = inside(source_root / "artifacts" / "paired-bench" / run_tag / name / f"run-{repetition}")
            run_directory.mkdir(parents=True, exist_ok=False)
            command = [str(executable), "solve", "--mode", "gpu-safe", "--device", "0",
                       "--instance", str(inputs["instance"]), "--tour", str(inputs["tour"]),
                       "--tour-role", "known-optimum", "--expected-cost", str(args.expected_cost),
                       "--output-edges", str(run_directory / "out.edg"),
                       "--fixed", str(run_directory / "out.fix"),
                       "--nonpairs", str(run_directory / "out.nonpairs"),
                       "--manifest", str(run_directory / "out.json"), *extra[name]]
            if args.input_edges is not None:
                command += ["--input-edges", str(inputs["input_edges"])]
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            begin = time.perf_counter()
            with (run_directory / "stdout.log").open("w") as stdout, (run_directory / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            wall_ms = (time.perf_counter() - begin) * 1000.0
            after = gpu_snapshot(args.gpu_uuid)
            record = {"variant": name, "repetition": repetition, "warmup": repetition < args.warmups,
                      "command": command, "executable_sha256": identity[name], "started_utc": started,
                      "process_wall_ms": wall_ms, "returncode": process.returncode,
                      "gpu_before": before, "gpu_after": after, "run_directory": str(run_directory)}
            records.append(record)
            # 即使随后 manifest 校验失败，也保留失败运行及日志位置。
            (output / "runs.json").write_text(json.dumps({"input_sha256": input_hashes, "records": records}, indent=2) + "\n")
            unchanged({name: executable}, identity)
            unchanged(inputs, input_hashes)
            if process.returncode == 0:
                manifest = json.loads((run_directory / "out.json").read_text())
                if manifest.get("termination") != "fixed-point" or not manifest.get("gpu_replayed") or manifest.get("proof_rejected") != 0:
                    raise RuntimeError("solve 未完成 GPU-safe 固定点")
                embedded = manifest.get("build_identity", {}).get("executable_sha256")
                if embedded is not None and embedded != identity[name]:
                    raise RuntimeError("manifest 与实际可执行文件身份不一致")
                record["solve"] = manifest
            (output / "runs.json").write_text(json.dumps({"input_sha256": input_hashes, "records": records}, indent=2) + "\n")
            if process.returncode != 0:
                raise RuntimeError(f"{name} 运行失败，日志保存在 {run_directory}")
            print(f"{name} run={repetition} warmup={record['warmup']} edges={record['solve']['final_edges']} process_wall_ms={wall_ms:.3f}", flush=True)
    summary = {"input_sha256": input_hashes, "variants": {}}
    for name in variants:
        measured = [r for r in records if r["variant"] == name and not r["warmup"]]
        hashes = {r["solve"]["final_state_hash"] for r in measured}
        if len(hashes) != 1:
            raise RuntimeError(f"{name} 的重复运行不确定")
        walls = [r["process_wall_ms"] for r in measured]
        summary["variants"][name] = {"process_wall_median_ms": statistics.median(walls),
            "process_wall_min_ms": min(walls), "process_wall_max_ms": max(walls),
            "final_edges": measured[0]["solve"]["final_edges"],
            "fixed_edges": measured[0]["solve"]["fixed_edges"],
            "nonpairs": measured[0]["solve"]["nonpairs"], "final_state_hash": next(iter(hashes)),
            "boundary_gpu_checks_clear": all(not r["gpu_before"]["processes"] and not r["gpu_after"]["processes"] for r in measured),
            "development_allow_busy": args.allow_busy}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
