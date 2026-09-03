#!/usr/bin/env python3
"""确认任意被篡改的几何见证都会被独立重放器拒绝。"""

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
        if len(fields) == 10 and fields[0] == "record" and fields[6] == "GEOM_MAIN":
            # 把第一个 potential 点改成目标边端点，结构仍合法但数学证明必定失败。
            fields[7] = fields[4]
            source[index] = " ".join(fields)
            changed = True
            break
    if not changed:
        raise RuntimeError("测试证书中没有 GEOM_MAIN record")
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
    if completed.returncode == 0 or "GEOM_MAIN 证明复核失败" not in completed.stdout:
        raise RuntimeError("篡改证书未被预期的几何门禁拒绝:\n" + completed.stdout)
    print("tampered FGPU geometry certificate rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
