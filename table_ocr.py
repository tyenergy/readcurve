# -*- coding: utf-8 -*-
"""
表格图片识别：OCR 所有文字块 → 按位置聚成行/列 → 输出 CSV + 核对图。

    python table_ocr.py table.png -o table.csv [--preview check.png] [--crop x0,y0,x1,y1]
                                  [--parsed parsed.csv] [--min-score 0.5]

- 行：按文字块的纵向中心聚类；表格横线（三线表/网格）作为额外分隔。
- 列：统计每个横坐标被多少行的文字覆盖，长期无覆盖的竖向空带是列间隙；
      跨列的表头不会把两列粘在一起；有竖线时直接按竖线分列。
- 单元格内多个文字块按 x 顺序用空格拼接。
- --parsed 额外输出数值解析版（±、%、×10^n、科学计数法），解析不了的保留原文。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from auto_detect import (NumLabel, _close_gaps, _runs_ge, ocr_tokens, parse_crop)


# ----------------------------------------------------------------------------
# 文本块去重（多倍率 OCR 会重复）
# ----------------------------------------------------------------------------
def _iou(a: NumLabel, b: NumLabel) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe_tokens(tokens: List[NumLabel], min_score: float = 0.5) -> List[NumLabel]:
    toks = [t for t in tokens if t.score >= min_score and t.text.strip()]
    toks.sort(key=lambda t: -t.score)
    out: List[NumLabel] = []
    for t in toks:
        if any(_iou(t, o) > 0.5 for o in out):
            continue
        # 被已有更长文本块包含（同一行里拆/合的两种结果）：保留覆盖更大的那个
        if any(o.x0 - 2 <= t.x0 and t.x1 <= o.x1 + 2 and o.y0 - 2 <= t.y0 and t.y1 <= o.y1 + 2
               and abs((t.y1 - t.y0) - (o.y1 - o.y0)) < 0.5 * (o.y1 - o.y0) for o in out):
            continue
        out.append(t)
    return out


# ----------------------------------------------------------------------------
# 表格线
# ----------------------------------------------------------------------------
def table_lines(rgb: np.ndarray, min_frac: float = 0.3):
    """返回 (横线行号列表, 竖线列号列表)。"""
    h, w = rgb.shape[:2]
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    dark = (mx < 180) & ((mx - mn) < 60)
    dh = _close_gaps(dark, 1, max(6, int(0.01 * w)))
    dv = _close_gaps(dark, 0, max(6, int(0.01 * h)))
    hrows, vcols = [], []
    for r in range(h):
        runs = _runs_ge(dh[r], int(min_frac * w))
        if runs and dark[r].sum() >= 0.7 * max(b - a + 1 for a, b in runs):
            hrows.append(r)
    for c in range(w):
        runs = _runs_ge(dv[:, c], int(min_frac * h))
        if runs and dark[:, c].sum() >= 0.7 * max(b - a + 1 for a, b in runs):
            vcols.append(c)

    def centers(idx):
        out, start, prev = [], None, None
        for v in idx:
            if start is None:
                start = prev = v
            elif v == prev + 1:
                prev = v
            else:
                out.append((start + prev) / 2.0)
                start = prev = v
        if start is not None:
            out.append((start + prev) / 2.0)
        return out
    return centers(hrows), centers(vcols)


# ----------------------------------------------------------------------------
# 行列聚类
# ----------------------------------------------------------------------------
@dataclass
class Table:
    cells: List[List[str]]
    rows_y: List[Tuple[float, float]]          # 每行的 (y0, y1)
    cols_x: List[Tuple[float, float]]          # 每列的 (x0, x1)
    tokens: List[NumLabel] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


def cluster_rows(tokens: List[NumLabel], hlines: List[float], grid: bool = False) -> List[List[NumLabel]]:
    if not tokens:
        return []
    med_h = float(np.median([t.y1 - t.y0 for t in tokens]))
    toks = sorted(tokens, key=lambda t: t.cy)
    if grid and len(hlines) >= 3:
        # 完整网格：相邻两条横线之间就是一行（单元格内换行的文字属于同一行）
        bounds = [-1.0] + sorted(hlines) + [float("inf")]
        rows = [[] for _ in range(len(bounds) - 1)]
        for t in toks:
            for i in range(len(bounds) - 1):
                if bounds[i] <= t.cy < bounds[i + 1]:
                    rows[i].append(t)
                    break
        return [r for r in rows if r]
    rows: List[List[NumLabel]] = [[toks[0]]]
    for t in toks[1:]:
        cur = rows[-1]
        cy = float(np.mean([u.cy for u in cur]))
        crossed = any(cy < L < t.cy for L in hlines)          # 中间隔着横线
        if abs(t.cy - cy) <= 0.6 * med_h and not crossed:
            cur.append(t)
        else:
            rows.append([t])
    return rows


def cluster_columns(rows: List[List[NumLabel]], width: int, vlines: List[float],
                    med_h: float) -> List[Tuple[float, float]]:
    """返回列的 x 区间列表。"""
    if vlines and len(vlines) >= 2:
        bounds = [0.0] + sorted(vlines) + [float(width)]
        cols = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 0.5 * med_h:
                cols.append((a, b))
        # 只保留里面有文字的列
        return cols
    n_rows = len(rows)
    cover = np.zeros(width + 1, dtype=np.int32)
    for row in rows:
        line = np.zeros(width + 1, dtype=bool)
        for t in row:
            line[int(max(0, t.x0)):int(min(width, t.x1)) + 1] = True
        # 同一行里相邻文字块之间的小空隙（词间距）视为连通
        gap = int(0.8 * med_h)
        idx = np.flatnonzero(line)
        if idx.size:
            d = np.diff(idx)
            for k in np.flatnonzero((d > 1) & (d <= gap)):
                line[idx[k]:idx[k + 1]] = True
        cover += line
    # 覆盖行数很少的位置是列间隙（跨列表头只占 1 行，不会把列粘起来）
    thresh = max(1, int(math.ceil(0.15 * n_rows)))
    covered = cover > thresh if n_rows >= 3 else cover > 0
    cols = []
    idx = np.flatnonzero(covered)
    if idx.size == 0:
        return [(0.0, float(width))]
    breaks = np.flatnonzero(np.diff(idx) > max(3, int(0.5 * med_h)))
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    for s_, e_ in zip(starts, ends):
        cols.append((float(idx[s_]), float(idx[e_])))
    # 把列边界外推到相邻列的中点，便于分配跨界的文字块
    bounds = []
    for i, (a, b) in enumerate(cols):
        left = 0.0 if i == 0 else (cols[i - 1][1] + a) / 2
        right = float(width) if i == len(cols) - 1 else (b + cols[i + 1][0]) / 2
        bounds.append((left, right))
    return bounds


def build_table(tokens: List[NumLabel], width: int, hlines: List[float], vlines: List[float]) -> Table:
    log: List[str] = []
    if not tokens:
        return Table([], [], [], [], ["没有识别到任何文字"])
    med_h = float(np.median([t.y1 - t.y0 for t in tokens]))
    grid = len(hlines) >= 3 and len(vlines) >= 2
    rows = cluster_rows(tokens, hlines, grid)
    cols = cluster_columns(rows, width, vlines, med_h)
    cell_toks: List[List[List[NumLabel]]] = [[[] for _ in cols] for _ in rows]
    spanning = 0
    for ri, row in enumerate(rows):
        for t in row:
            best, best_ov = 0, -1.0                 # 分配到重叠最多的列
            for ci, (a, b) in enumerate(cols):
                ov = min(t.x1, b) - max(t.x0, a)
                if ov > best_ov:
                    best, best_ov = ci, ov
            if best_ov < 0.5 * (t.x1 - t.x0):
                spanning += 1
            cell_toks[ri][best].append(t)

    def join(toks: List[NumLabel]) -> str:
        """单元格内按 行(y) → x 顺序拼接；数字换行续写不加空格。"""
        out = ""
        for t in sorted(toks, key=lambda t: (round(t.cy / max(1.0, 0.6 * med_h)), t.x0)):
            txt = t.text.strip()
            if not out:
                out = txt
            elif out[-1].isdigit() and txt[:1].isdigit() and not any(ch.isalpha() for ch in txt)                     and (t.y0 > max(u.y1 for u in toks if u is not t and u.cy < t.cy) - 2
                         if any(u.cy < t.cy for u in toks if u is not t) else False):
                out += txt                             # 上一行末尾是数字，本行是纯数字 → 续写
            else:
                out += " " + txt
        return out

    cells = [[join(ct) for ct in row] for row in cell_toks]
    rows_y = [(min(t.y0 for t in r), max(t.y1 for t in r)) for r in rows]
    # 去掉没有任何文字的列/行（外侧表格线会造成空列）
    keep_c = [ci for ci in range(len(cols)) if any(cells[ri][ci] for ri in range(len(rows)))]
    cells = [[row[ci] for ci in keep_c] for row in cells]
    cols = [cols[ci] for ci in keep_c]
    keep_r = [ri for ri in range(len(rows)) if any(cells[ri])]
    cells = [cells[ri] for ri in keep_r]
    rows_y = [rows_y[ri] for ri in keep_r]
    log.append(f"文字块 {len(tokens)} 个 → {len(cells)} 行 × {len(cols)} 列"
               + (f"，{len(hlines)} 条横线" if hlines else "") + (f"，{len(vlines)} 条竖线" if vlines else "")
               + ("，按网格线分行" if grid else ""))
    if spanning:
        log.append(f"{spanning} 个文字块跨越多列（多为合并表头），已放入其起始列")
    return Table(cells, rows_y, cols, tokens, log)


# ----------------------------------------------------------------------------
# 数值解析
# ----------------------------------------------------------------------------
_NUM = r"[-−–+]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+)?"
_RE_PM = re.compile(rf"^\s*({_NUM})\s*(?:±|\+/-|\+-)\s*({_NUM})\s*(%?)\s*$")
_RE_NUM = re.compile(rf"^\s*({_NUM})\s*(%?)\s*$")
_RE_POW = re.compile(rf"^\s*({_NUM})\s*[×x*]\s*10\s*\^?\s*([-−–+]?\d+)\s*$")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _f(s: str) -> float:
    return float(s.replace("−", "-").replace("–", "-").replace(",", "."))


def parse_cell(text: str):
    """返回 (值, 误差或None, 原文)；解析不了返回 (None, None, 原文)。"""
    t = text.strip().translate(_SUP).replace("，", ",")
    t = re.sub(r"(\d)\s*,\s*(\d{3})(?!\d)", r"\1\2", t)       # 千分位 1,234
    m = _RE_PM.match(t)
    if m:
        v, e = _f(m.group(1)), _f(m.group(2))
        if m.group(3):
            v, e = v / 100, e / 100
        return v, e, text
    m = _RE_POW.match(t)
    if m:
        return _f(m.group(1)) * 10 ** int(m.group(2).replace("−", "-").replace("–", "-")), None, text
    m = _RE_NUM.match(t)
    if m:
        v = _f(m.group(1))
        return (v / 100 if m.group(2) else v), None, text
    return None, None, text


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def recognize_table(rgb: np.ndarray, min_score: float = 0.5, use_lines: bool = True) -> Table:
    toks = dedupe_tokens(ocr_tokens(rgb), min_score)
    hl, vl = table_lines(rgb) if use_lines else ([], [])
    tb = build_table(toks, rgb.shape[1], hl, vl)
    return tb


def save_table_preview(rgb: np.ndarray, tb: Table, path: str):
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, (y0, y1) in enumerate(tb.rows_y):
        d.rectangle([0, y0 - 1, img.width - 1, y1 + 1], outline=(0, 160, 255) if i % 2 else (0, 200, 120))
    for a, b in tb.cols_x:
        d.line([(a, 0), (a, img.height)], fill=(255, 140, 0))
    for t in tb.tokens:
        d.rectangle([t.x0, t.y0, t.x1, t.y1], outline=(255, 0, 255))
    img.save(path)


def write_csv(path: str, cells: List[List[str]], delimiter: str = ","):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=delimiter)
        for row in cells:
            w.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description="表格图片 → CSV")
    ap.add_argument("image")
    ap.add_argument("-o", "--output", help="CSV 输出路径（默认 <图片名>_table.csv）")
    ap.add_argument("--preview", help="核对图：行带、列线、文字框")
    ap.add_argument("--parsed", help="额外输出数值解析版 CSV（± 拆成 值 和 误差 两列）")
    ap.add_argument("--crop", help="只处理 x0,y0,x1,y1 区域")
    ap.add_argument("--min-score", type=float, default=0.5, help="OCR 置信度门槛")
    ap.add_argument("--no-lines", action="store_true", help="不用表格线辅助分行分列")
    args = ap.parse_args(argv)

    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    if args.crop:
        img = img.crop(parse_crop(args.crop, img.size))
    rgb = np.asarray(img)
    tb = recognize_table(rgb, args.min_score, not args.no_lines)
    for line in tb.log:
        print(line)
    if not tb.cells:
        return 1
    out = args.output or os.path.splitext(args.image)[0] + "_table.csv"
    delim = "\t" if out.lower().endswith((".txt", ".tsv", ".dat")) else ","
    write_csv(out, tb.cells, delim)
    print("已导出:", out)
    for row in tb.cells[:12]:
        print("  | " + " | ".join(row))
    if len(tb.cells) > 12:
        print(f"  ... 共 {len(tb.cells)} 行")
    if args.parsed:
        rows = []
        for row in tb.cells:
            prow = []
            for cell in row:
                v, e, raw = parse_cell(cell)
                if v is None:
                    prow += [raw, ""]
                else:
                    prow += ["%.10g" % v, "" if e is None else "%.10g" % e]
            rows.append(prow)
        write_csv(args.parsed, rows, delim)
        print("数值解析版已导出:", args.parsed, "（每个单元格两列：值, 误差）")
    if args.preview:
        save_table_preview(rgb, tb, args.preview)
        print("核对图已保存:", args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
