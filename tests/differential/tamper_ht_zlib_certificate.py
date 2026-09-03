#!/usr/bin/env python3
"""确认 V5 证书中的压缩 HT sidecar 被翻转一位后必须拒绝。"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--tsp", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    content = bytearray(pathlib.Path(args.proof).read_bytes())
    header_start = content.find(b"ht_proof_zlib ")
    if header_start < 0:
        raise RuntimeError("测试证书中没有 V5 压缩 HT sidecar")
    header_end = content.find(b"\n", header_start)
    if header_end < 0:
        raise RuntimeError("压缩 HT sidecar 头被截断")
    fields = bytes(content[header_start:header_end]).split()
    if len(fields) != 5:
        raise RuntimeError("压缩 HT sidecar 头字段数非法")
    compressed_size = int(fields[3])
    payload_start = header_end + 1
    if compressed_size < 2 or payload_start + compressed_size > len(content):
        raise RuntimeError("压缩 HT sidecar payload 规模非法")
    content[payload_start + compressed_size // 2] ^= 0x01
    pathlib.Path(args.output).write_bytes(content)

    completed = subprocess.run(
        [
            args.exe,
            "verify",
            "--tsp",
            args.tsp,
            "--edges",
            args.edges,
            "--proof",
            args.output,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode == 0 or not any(
        marker in completed.stdout for marker in ("zlib", "CRC32")
    ):
        raise RuntimeError("被篡改的压缩 HT sidecar 未被拒绝:\n" + completed.stdout)
    print("tampered compressed HT certificate rejected")


if __name__ == "__main__":
    main()
