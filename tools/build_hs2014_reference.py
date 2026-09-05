#!/usr/bin/env python3
"""仅在项目内复制/适配受限作者源码；不把源文件提交到 Git。"""

import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "references/EdgeElimination/code"


def main():
    files = sorted(path for path in SOURCE.iterdir() if path.suffix in (".h", ".cpp"))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    adapter = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = ROOT / ".deps" / ("hs2014-author-" + identity[:12] + "-" + adapter[:8])
    build = ROOT / "build/hs2014-reference"
    for path in (destination, build):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("作者参考输出越界")
        path.mkdir(parents=True, exist_ok=True)
    patches = []
    for path in files:
        before = path.read_text()
        after = before
        if path.name == "mpi_functions.h":
            after = after.replace('"/lfs/user/schroeder/bin/mpi/include/mpi.h"', '<mpi.h>')
        if path.name == "subwindow.cpp":
            after = after.replace('#include "/lfs/user/schroeder/TSP/TSP_Master/Concorde/130726/concorde.h"',
                                  '#define class cc_class\n#include "concorde.h"\n#undef class')
            # 原版强制 use_heldkarp=true，此不可达调用仍需与所链接旧 Concorde ABI 对齐。
            old = '                        NULL,\n                        NULL,\n                        NULL,\n                        0,\n                        _silent,'
            if after.count(old) != 1:
                raise ValueError("Concorde ABI 适配位置不匹配")
            after = after.replace(old, '                        NULL,\n                        NULL,\n                        _silent,')
        if path.name == "instance_general.cpp":
            # 仅两个 ostringstream 日志错误；不能替换同文件中的 char* filename。
            after = after.replace('<< "Failed to open file " << filename << endl;',
                                  '<< "Failed to open file " << filename.str() << endl;')
            old = 'void Instance::plot_edges(ostringstream& extension, NodeId p_, NodeId q_)\n{'
            if after.count(old) != 1:
                raise ValueError("作者绘图开关适配位置不匹配")
            after = after.replace(old, old + '\n   if (!options().plot_instance_edges()) return;')
        if path.name == "main.cpp":
            old = '      mpi_ask_for_job(inst);'
            if after.count(old) != 1:
                raise ValueError("作者 worker 日志位置不匹配")
            after = after.replace(old, '      inst.log().setstate(std::ios::failbit);\n' + old)
        if path.name == "instance_s3.cpp":
            old = '      int num_edges_iteration = num_edges_beg_iteration - _num_edges;'
            if after.count(old) != 1:
                raise ValueError("作者 S3 dispatch 源码版本不匹配")
            # 完整迭代的停止判断必须等待最后一个 worker 结果；对 MPI/serial 两者同样适配。
            after = after.replace(old, '      mpi_collect_all_data(true);\n' + old)
        target = destination / path.name
        if target.exists() and target.read_text() != after:
            raise ValueError(f"保留已修改作者 overlay，不覆盖：{target}")
        if not target.exists():
            target.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=path.name, tofile=path.name))
    (build / "source-adaptation.patch").write_text("".join(patches))
    sources = [str(destination / path.name) for path in files if path.suffix == ".cpp" and path.name != "main.cpp"]
    concorde = ROOT / "build/concorde"
    if not (concorde / "concorde.a").is_file():
        raise ValueError("需先运行项目内 Concorde bootstrap 构建参考库")
    qsopt = ROOT / ".deps/qsopt/qsopt.a"
    if not qsopt.is_file():
        raise ValueError("未找到项目内 QSopt 静态库")
    environment = os.environ.copy()
    environment["TMPDIR"] = str(ROOT / ".tmp")
    # 保留作者 assert；不设置 NDEBUG，以免改变带副作用的原版断言行为。
    common = ["-O3", "-std=c++20", "-frounding-math", "-include", "iostream", "-I/usr/pkg/include", f"-I{destination}", f"-I{concorde}", *sources,
              str(concorde / "concorde.a"), str(qsopt), "-lm", "-lpthread"]
    serial = ["c++", f'-DCUDAEE_SOURCE_DIR="{ROOT}"', f'-I{ROOT / "benchmarks/hs2014/serial"}',
              str(ROOT / "benchmarks/hs2014/serial_main.cpp"), *common, "-o", str(build / "hs2014-serial")]
    mpi = ["mpicxx", str(destination / "main.cpp"), *common, "-o", str(build / "hs2014-mpi")]
    for name, command in (("serial", serial), ("mpi", mpi)):
        with (build / f"{name}-build.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {"source_date": "2015-02-27", "source_sha256": identity, "source_files": source_hashes, "adapter_sha256": adapter,
              "patch_sha256": hashlib.sha256("".join(patches).encode()).hexdigest(), "commands": [serial, mpi],
              "executables": {name: hashlib.sha256((build / f"hs2014-{name}").read_bytes()).hexdigest() for name in ("serial", "mpi")},
              "differential_validated": False,
              "note": "构建可用不等于单核对照验证通过；必须先与原始单 worker 路径差分。"}
    (build / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(build)


if __name__ == "__main__":
    main()
