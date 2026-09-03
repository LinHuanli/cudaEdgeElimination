#!/usr/bin/env python3
"""确认 FGPU 拒绝文件级符号链接逃逸和互相覆盖的输出路径。"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def invoke(executable: str, instance: str, paths: list[pathlib.Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            executable,
            "run",
            "--instance",
            instance,
            "--enable-geometry",
            "0",
            "--enable-jv",
            "0",
            "--enable-ht",
            "0",
            "--pdlp",
            "off",
            "--output-edges",
            str(paths[0]),
            "--fixed",
            str(paths[1]),
            "--nonpairs",
            str(paths[2]),
            "--certificate",
            str(paths[3]),
            "--manifest",
            str(paths[4]),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--tmp", required=True)
    args = parser.parse_args()

    temporary = pathlib.Path(args.tmp)
    temporary.mkdir(parents=True, exist_ok=True)
    escaped = temporary / "escaped.edg"
    escaped.unlink(missing_ok=True)
    # 指向只读的 procfs 文件：即使门禁回归也不会在仓库外创建产物。
    escaped.symlink_to("/proc/self/status")
    ordinary = [temporary / f"escape-{index}.out" for index in range(4)]
    result = invoke(args.exe, args.instance, [escaped, *ordinary])
    if result.returncode == 0 or "输出路径必须位于仓库中" not in result.stdout:
        raise RuntimeError("文件级符号链接逃逸未被拒绝:\n" + result.stdout)

    duplicate = temporary / "duplicate.out"
    paths = [
        duplicate,
        duplicate,
        temporary / "dup.np",
        temporary / "dup.cert",
        temporary / "dup.manifest",
    ]
    result = invoke(args.exe, args.instance, paths)
    if result.returncode == 0 or "五个输出路径必须互不相同" not in result.stdout:
        raise RuntimeError("重名输出未被拒绝:\n" + result.stdout)

    input_overwrite = [
        pathlib.Path(args.instance),
        temporary / "input.fix",
        temporary / "input.np",
        temporary / "input.cert",
        temporary / "input.manifest",
    ]
    result = invoke(args.exe, args.instance, input_overwrite)
    if result.returncode == 0 or "不得覆盖 instance" not in result.stdout:
        raise RuntimeError("输出覆盖输入的门禁未生效:\n" + result.stdout)

    print("FGPU output boundary violations rejected")


if __name__ == "__main__":
    main()
