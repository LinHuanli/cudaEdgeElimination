#!/usr/bin/env python3
"""确认 verifier 会拒绝与最终证明图不一致的 fixed 输出。"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--nonpairs", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = pathlib.Path(args.fixed).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise RuntimeError("测试 fixed 文件没有可删除的边")
    header = lines[0].split()
    declared = int(header[1])
    if declared < 1 or len(lines) != declared + 1:
        raise RuntimeError("测试 fixed 文件头非法")
    header[1] = str(declared - 1)
    tampered = [" ".join(header), *lines[1:-1]]
    pathlib.Path(args.output).write_text("\n".join(tampered) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            args.exe,
            "verify",
            "--instance",
            args.instance,
            "--output-edges",
            args.edges,
            "--fixed",
            args.output,
            "--nonpairs",
            args.nonpairs,
            "--certificate",
            args.certificate,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode == 0 or "fixed" not in completed.stdout:
        raise RuntimeError("篡改 fixed 输出未被拒绝:\n" + completed.stdout)
    print("tampered FGPU fixed output rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
