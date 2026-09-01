#!/usr/bin/env python3
"""检查可提交文件、子模块与大文件，避免把依赖或产物误加入 Git。"""

from __future__ import annotations

import pathlib
import subprocess


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    forbidden = {".deps", ".venv", ".tmp", "build", "artifacts"}
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.split(b"\0")
    errors: list[str] = []
    for raw in tracked:
        if not raw:
            continue
        relative = pathlib.Path(raw.decode())
        if relative.parts and relative.parts[0] in forbidden:
            if relative.as_posix() != "artifacts/README.md":
                errors.append(f"禁止提交的路径：{relative}")
        path = root / relative
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"超过 5 MiB 的跟踪文件：{relative}")
    if errors:
        print("\n".join(errors))
        return 1
    print("workspace policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
