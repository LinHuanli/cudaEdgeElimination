#!/usr/bin/env python3
"""锁定原始 TSPLIB 坐标和事后验证标签；不生成求解器输入边集。"""

import gzip
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hs2014_hybrid.json"
DATA = ROOT / ".deps/hs2014-data"


def records():
    return json.loads(CONFIG.read_text())["instances"]


def load_instance(path):
    text = path.read_text()
    header, section = text.split("NODE_COORD_SECTION", 1)
    fields = {a.strip(): b.strip() for a, b in
              (line.split(":", 1) for line in header.splitlines() if ":" in line)}
    n = int(fields["DIMENSION"])
    if fields["EDGE_WEIGHT_TYPE"] not in ("EUC_2D", "CEIL_2D"):
        raise ValueError("不支持的原始距离类型")
    rows = [line.split() for line in section.splitlines() if line.strip() and line.strip() != "EOF"]
    if len(rows) != n or any(len(row) != 3 for row in rows):
        raise ValueError("原始 TSPLIB 坐标不完整")
    points = [None] * n
    for row in rows:
        index = int(row[0])
        # 原版 vm1084 用科学计数法书写整数；不经过二进制浮点归一化。
        raw_x, raw_y = map(Decimal, row[1:])
        if not raw_x.is_finite() or not raw_y.is_finite() or 2 * raw_x != int(2 * raw_x) or 2 * raw_y != int(2 * raw_y):
            raise ValueError("坐标不在整数/半整数精确 GPU 度量范围")
        # 保留原始 Decimal，独立 checker 使用统一半整数分子运算。
        x, y = raw_x, raw_y
        if not 1 <= index <= n or points[index - 1] is not None:
            raise ValueError("原始 TSPLIB 顶点编号重复或越界")
        points[index - 1] = (x, y)
    return points, fields["EDGE_WEIGHT_TYPE"]


def load_tour(path, n):
    text = path.read_text().split("TOUR_SECTION", 1)[1]
    vertices = []
    for token in text.split():
        if token in ("-1", "EOF"):
            break
        vertices.append(int(token) - 1)
    if sorted(vertices) != list(range(n)):
        raise ValueError("标签 tour 不是完整排列")
    return vertices


def distance(a, b, metric):
    square = int(2 * (a[0] - b[0])) ** 2 + int(2 * (a[1] - b[1])) ** 2
    root = math.isqrt(square)
    return root // 2 + (square != (root // 2 * 2) ** 2 if metric == "CEIL_2D" else root % 2)


def validate(item):
    paths = {"tsp": DATA / f'{item["name"]}.tsp', "tour": DATA / f'{item["name"]}.opt.tour'}
    for kind, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != item[f"{kind}_sha256"]:
            raise ValueError(f"原始数据身份不符：{path}")
    points, metric = load_instance(paths["tsp"])
    if len(points) != item["n"]:
        raise ValueError("原始坐标维度不符")
    tour = load_tour(paths["tour"], len(points))
    cost = sum(distance(points[tour[i]], points[tour[(i + 1) % len(tour)]], metric) for i in range(len(tour)))
    if cost != item["optimum"]:
        raise ValueError(f'标签成本不符：{item["name"]}，实算 {cost}')
    return paths, points, metric, tour


def main():
    if not DATA.resolve().is_relative_to(ROOT):
        raise ValueError("下载目录逃出项目")
    DATA.mkdir(parents=True, exist_ok=True)
    for item in records():
        for kind, suffix in (("tsp", "tsp"), ("tour", "opt.tour")):
            path = DATA / f'{item["name"]}.{suffix}'
            expected = item[f"{kind}_sha256"]
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"保留现有文件，不覆写不匹配数据：{path}")
                continue
            url = item[f"{kind}_url"]
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            if url.endswith(".gz"):
                payload = gzip.decompress(payload)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"下载身份不匹配：{url}")
            with path.open("xb") as stream:
                stream.write(payload)
        validate(item)
        print(f'{item["name"]}: n={item["n"]} optimum={item["optimum"]} raw SHA256/cost verified', flush=True)


if __name__ == "__main__":
    main()
