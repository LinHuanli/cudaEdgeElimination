#!/usr/bin/env python3
"""合成运行记录只测试验收门禁，不作为实验数据。"""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from compare_hybrid import compare
from prepare_hs2014_data import records


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.items = records()
        self.gpu, self.cpu = [], []
        for item in self.items:
            for iteration in range(4):
                common = {"instance": item["name"], "iteration": iteration,
                          "warmup": iteration == 0, "clean": True, "returncode": 0,
                          "host_identity": {"hostname": "synthetic-node", "cpu_model": "synthetic-cpu-model"},
                          "config_sha256": "synthetic-config"}
                self.gpu.append({**common, "executable_sha256": "synthetic-gpu",
                    "process_wall_seconds": 10.0,
                    "postcheck": {"known_optimum_conflicts": 0, "active_edges": item["paper_edges"] - 1},
                    "report": {"lp_backend": "off", "distance_cache": True, "main_pair_cache": True,
                               "gpu_identity": {"name": "synthetic-gpu-model", "compute_major": 8, "compute_minor": 9},
                               "full_degree_metric": True, "termination": "fixed-point",
                               "gpu_replayed": True, "proof_rejected": 0, "profile": "hybrid-e2e",
                               "initial_edges": item["n"] * (item["n"] - 1) // 2,
                               "tour_sha256": None, "input_edges_sha256": None,
                               "input_optimum_labels": False, "instance_sha256": item["tsp_sha256"],
                               "final_state_hash": "synthetic-state",
                               "final_edges": item["paper_edges"] - 1}})
                self.cpu.append({**common, "executable_sha256": "synthetic-cpu",
                    "process_wall_seconds": 20.0, "differential_validated": True,
                    "options_sha256": "synthetic-options", "cpu_core": 2,
                    "instance_sha256": item["tsp_sha256"],
                    "postcheck": {"known_optimum_conflicts": 0, "active_edges": item["paper_edges"],
                                  "edge_set_sha256": "synthetic-state"}})

    def test_all_four_gates(self):
        report = compare(self.gpu, self.cpu, self.items)
        self.assertTrue(report["complete_acceptance"])
        self.assertEqual(report["instances"]["pr1002"]["same_machine_speedup"], 2.0)

    def test_contamination_and_missing_reference(self):
        self.assertFalse(compare(self.gpu, [], self.items)["complete_acceptance"])
        self.gpu[1]["clean"] = False
        self.assertFalse(compare(self.gpu, self.cpu, self.items)["complete_acceptance"])

    def test_input_labels_or_incomplete_graph(self):
        for field, value in (("tour_sha256", "injected"), ("initial_edges", 100),
                             ("input_optimum_labels", True), ("instance_sha256", "wrong")):
            bad = copy.deepcopy(self.gpu)
            bad[1]["report"][field] = value
            self.assertFalse(compare(bad, self.cpu, self.items)["complete_acceptance"])

    def test_no_equal_strength_or_speed_pass(self):
        for index in range(4):
            self.gpu[index]["report"]["final_edges"] = self.items[0]["paper_edges"]
            self.gpu[index]["postcheck"]["active_edges"] = self.items[0]["paper_edges"]
        self.assertFalse(compare(self.gpu, self.cpu, self.items)["complete_acceptance"])
        self.setUp()
        for run in self.gpu:
            run["process_wall_seconds"] = 20.0
        self.assertFalse(compare(self.gpu, self.cpu, self.items)["complete_acceptance"])

    def test_warmup_hash_and_configuration(self):
        for key, value in (("warmup", False), ("executable_sha256", "other"),
                           ("config_sha256", "other")):
            bad = copy.deepcopy(self.gpu)
            bad[0][key] = value
            self.assertFalse(compare(bad, self.cpu, self.items)["complete_acceptance"])
        self.gpu[1]["report"]["final_state_hash"] = "different"
        self.assertFalse(compare(self.gpu, self.cpu, self.items)["complete_acceptance"])

    def test_point_configuration_must_be_shared(self):
        for field, value in (("leaf_permutation_cache", True), ("point_near_first", True),
                             ("point_adaptive_start", True),
                             ("point_prime_near", True),
                             ("quick_reply_cache", True),
                             ("point_leaf_kernel", "subset-dp"), ("point_cta_blocks", 8)):
            bad = copy.deepcopy(self.gpu)
            bad[1]["report"][field] = value
            self.assertFalse(compare(bad, self.cpu, self.items)["complete_acceptance"])

    def test_cross_machine_and_gpu_rejected(self):
        for host in ({}, {"hostname": "remote", "cpu_model": "synthetic-cpu-model"},
                     {"hostname": "synthetic-node", "cpu_model": "other-cpu"}):
            bad = copy.deepcopy(self.gpu)
            bad[1]["host_identity"] = host
            self.assertFalse(compare(bad, self.cpu, self.items)["complete_acceptance"])
        for field, value in (("name", "different GPU"), ("compute_major", 9), ("compute_minor", 0)):
            bad = copy.deepcopy(self.gpu)
            bad[1]["report"]["gpu_identity"][field] = value
            self.assertFalse(compare(bad, self.cpu, self.items)["complete_acceptance"])


if __name__ == "__main__":
    unittest.main()
