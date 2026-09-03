#!/usr/bin/env python3
"""确认篡改后的 LP-box multiplier 不能继续支撑删边结论。"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.certificate).read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(source):
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "lp_box_dual":
            # 保持证书语法与维度都合法，只把一个 unrestricted multiplier
            # 改成极端值；重新计算后的 box bound 应失去删边证明能力。
            fields[2] = str((1 << 63) - 1)
            source[index] = " ".join(fields)
            changed = True
            break
    if not changed:
        raise RuntimeError("测试证书中没有 lp_box_dual")
    pathlib.Path(args.output).write_text("\n".join(source) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            args.exe,
            "verify",
            "--instance",
            args.instance,
            "--output-edges",
            args.edges,
            "--certificate",
            args.output,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode == 0 or "LP_BOX" not in completed.stdout:
        raise RuntimeError("篡改证书未被预期的 LP 门禁拒绝:\n" + completed.stdout)
    print("tampered FGPU LP certificate rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
