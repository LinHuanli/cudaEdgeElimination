#!/usr/bin/env python3
"""从已完成的 GPU/作者单核原始运行记录检查四例双强度/速度门槛。"""

import argparse
import json
from pathlib import Path
import statistics
import math

from benchmark_fgpu import inside
from prepare_hs2014_data import records


def compare(gpu_runs, cpu_runs, items):
    result = {"complete_acceptance": False, "instances": {}}
    gpu_ids = {x["executable_sha256"] for x in gpu_runs}
    cpu_ids = {x["executable_sha256"] for x in cpu_runs}
    configurations = {(x["report"]["lp_backend"], x["report"]["distance_cache"],
                       x["report"]["main_pair_cache"], x["report"]["full_degree_metric"],
                       x["report"].get("leaf_permutation_cache", False),
                       x["report"].get("point_near_first", False),
                       x["report"].get("point_adaptive_start", False),
                       x["report"].get("point_leaf_kernel", "permutation"),
                       x["report"].get("point_cta_blocks", 4))
                      for x in gpu_runs if "report" in x}
    shared_config = len(gpu_ids) == len(cpu_ids) == len(configurations) == 1
    shared_config = shared_config and len({x.get("config_sha256") for x in gpu_runs + cpu_runs}) == 1
    shared_config = shared_config and len({x.get("options_sha256") for x in cpu_runs}) == 1
    shared_config = shared_config and len({x.get("cpu_core") for x in cpu_runs}) == 1
    # hostname/CPU 型号均须一致；已有但缺失这些身份的 pilot 不提升为同机正式数据。
    host_keys = {(x.get("host_identity", {}).get("hostname"),
                  x.get("host_identity", {}).get("cpu_model")) for x in gpu_runs + cpu_runs}
    same_machine = len(host_keys) == 1 and all(all(key) for key in host_keys)
    gpu_models = {(x.get("report", {}).get("gpu_identity", {}).get("name"),
                   x.get("report", {}).get("gpu_identity", {}).get("compute_major"),
                   x.get("report", {}).get("gpu_identity", {}).get("compute_minor")) for x in gpu_runs}
    same_gpu_model = len(gpu_models) == 1 and all(
        key[0] and key[1] is not None and key[2] is not None for key in gpu_models)
    shared_config = shared_config and same_machine and same_gpu_model
    for item in items:
        name = item["name"]
        gpu = [x for x in gpu_runs if x["instance"] == name]
        cpu = [x for x in cpu_runs if x["instance"] == name]
        valid_gpu = [x for x in gpu if x.get("clean") and not x["warmup"] and
                     x["returncode"] == 0 and x.get("postcheck", {}).get("known_optimum_conflicts") == 0 and
                     x.get("report", {}).get("termination") == "fixed-point" and
                     x["report"].get("gpu_replayed") and not x["report"].get("proof_rejected") and
                     x["report"].get("initial_edges") == item["n"] * (item["n"] - 1) // 2 and
                     x["report"].get("profile") == "hybrid-e2e" and
                     x["report"].get("instance_sha256") == item["tsp_sha256"] and
                     x["report"].get("input_edges_sha256") is None and
                     x["report"].get("final_edges") == x["postcheck"].get("active_edges") and
                     math.isfinite(x["process_wall_seconds"]) and x["process_wall_seconds"] > 0 and
                     x["report"].get("tour_sha256") is None and
                     x["report"].get("input_optimum_labels") is False]
        valid_cpu = [x for x in cpu if x.get("clean") and not x["warmup"] and
                     x["returncode"] == 0 and x.get("postcheck", {}).get("known_optimum_conflicts") == 0 and
                     x.get("differential_validated") and x.get("instance_sha256") == item["tsp_sha256"] and
                     math.isfinite(x["process_wall_seconds"]) and x["process_wall_seconds"] > 0]
        repeatable = (len({x["report"]["final_state_hash"] for x in valid_gpu}) == 1 and
                      len({x["postcheck"]["edge_set_sha256"] for x in valid_cpu}) == 1)
        ready = shared_config and repeatable and len(valid_gpu) >= 3 and len(valid_cpu) >= 3 and \
                any(x["warmup"] and x.get("returncode") == 0 for x in gpu) and \
                any(x["warmup"] and x.get("returncode") == 0 for x in cpu)
        row = {"ready": ready, "same_machine": same_machine, "same_gpu_model": same_gpu_model,
               "gpu_clean_runs": len(valid_gpu), "cpu_clean_runs": len(valid_cpu),
               "passed": False}
        if ready:
            gpu_seconds = statistics.median(x["process_wall_seconds"] for x in valid_gpu)
            cpu_seconds = statistics.median(x["process_wall_seconds"] for x in valid_cpu)
            gpu_edges = valid_gpu[0]["report"]["final_edges"]
            cpu_edges = valid_cpu[0]["postcheck"]["active_edges"]
            row.update({"gpu_seconds": gpu_seconds, "cpu_seconds": cpu_seconds,
                        "gpu_edges": gpu_edges, "cpu_edges": cpu_edges,
                        "paper_edges": item["paper_edges"], "paper_seconds": item["paper_seconds"],
                        "paper_speedup": item["paper_seconds"] / gpu_seconds,
                        "same_machine_speedup": cpu_seconds / gpu_seconds,
                        "passed": gpu_edges < item["paper_edges"] and gpu_edges <= cpu_edges and
                                  gpu_seconds < item["paper_seconds"] and gpu_seconds < cpu_seconds})
        result["instances"][name] = row
    result["complete_acceptance"] = len(result["instances"]) == 4 and all(
        x["passed"] for x in result["instances"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-runs", type=Path, required=True)
    parser.add_argument("--cpu-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(inside(args.gpu_runs).read_text()),
                     json.loads(inside(args.cpu_runs).read_text()), records())
    output = inside(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
