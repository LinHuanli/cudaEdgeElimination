#!/usr/bin/env python3
"""确认篡改 Quick-HS 的 c,d 紧凑见证后，独立重放必须失败。"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--input-edges", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.certificate).read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(source):
        fields = line.split()
        if len(fields) == 10 and fields[0] == "record" and fields[6] == "GPU_QUICK_HS":
            # c 改成目标边端点；语法仍合法，但见证与目标边相交，数学复核必失败。
            fields[7] = fields[4]
            source[index] = " ".join(fields)
            changed = True
            break
    if not changed:
        raise RuntimeError("测试证书中没有 GPU_QUICK_HS record")
    pathlib.Path(args.output).write_text("\n".join(source) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            args.exe,
            "verify",
            "--instance",
            args.instance,
            "--input-edges",
            args.input_edges,
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
    if completed.returncode == 0 or "GPU_QUICK_HS 证明复核失败" not in completed.stdout:
        raise RuntimeError("篡改证书未被预期的 Quick-HS 门禁拒绝:\n" + completed.stdout)
    print("tampered FGPU Quick-HS certificate rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
