#!/usr/bin/env python3
"""检查 FGPU manifest V2 的结构和指定配置值。"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expect", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    lines = arguments.manifest.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "FGPU_ELIM_MANIFEST_V2" or lines[-1] != "END":
        raise RuntimeError("FGPU manifest 缺少 V2 头或 END 尾标")

    fields: dict[str, str] = {}
    for line in lines[1:-1]:
        key, separator, value = line.partition(" ")
        if not separator or not key or not value or key in fields:
            raise RuntimeError(f"FGPU manifest 字段非法或重复: {line!r}")
        fields[key] = value

    required = {
        "instance",
        "input_edges",
        "tour",
        "tour_role",
        "expected_tour_cost",
        "device",
        "numeric",
        "verification",
        "enable_geometry",
        "enable_jv",
        "enable_ht",
        "potential_candidates",
        "geometry_witnesses_per_edge",
        "max_jv_rounds",
        "max_ht_epochs",
        "ht_targets_per_epoch",
        "ht_target_workers",
        "max_paths",
        "max_local_nodes",
        "pdlp",
        "pdlp_iterations_budget",
        "max_pdlp_epochs",
        "termination",
        "certificate_bytes",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise RuntimeError(f"FGPU manifest 缺少字段: {', '.join(missing)}")

    for expected in arguments.expect:
        key, separator, value = expected.partition("=")
        if not separator or fields.get(key) != value:
            raise RuntimeError(
                f"FGPU manifest 值不一致: {key!r}, "
                f"expected={value!r}, actual={fields.get(key)!r}"
            )

    print("FGPU manifest V2 verified")


if __name__ == "__main__":
    main()
