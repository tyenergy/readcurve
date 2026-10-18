# -*- coding: utf-8 -*-
"""
全自动识别：坐标轴框 → OCR 读刻度数字做标定 → 按颜色自动找出所有曲线。

命令行用法（批处理）：
    python auto_detect.py chart.png -o out.csv [--mode separate|common] [--n 200]
                                     [--tol 50] [--preview check.png]
在 GUI 中由“一键自动”按钮调用。

OCR 使用 rapidocr_onnxruntime（pip install rapidocr_onnxruntime）；
未安装时仍可自动找坐标轴框和曲线，只是刻度数值需手动输入。
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from curve_core import AxisCalibration, Curve, color_mask, trace_curve

# ----------------------------------------------------------------------------
# 1. 坐标轴框检测
# ----------------------------------------------------------------------------
@dataclass
class PlotFrame:
    x_axis_row: Optional[int]       # X 轴所在行
    y_axis_col: Optional[int]       # Y 轴所在列
    box: Tuple[int, int, int, int]  # 绘图区 (x0, y0, x1, y1)
    has_top: bool = False           # 顶部有边框线
    has_right: bool = False         # 右侧有边框线

    def inner_box(self, shrink: int = 4, grow: int = 3) -> Tuple[int, int, int, int]:
        """有轴线/边框的一侧向内缩 shrink（去掉线本身），没有的一侧外扩 grow（曲线可能顶到边）。"""
        x0, y0, x1, y1 = self.box
        return (x0 + shrink if self.y_axis_col is not None else x0 - grow,
                y0 + shrink if self.has_top else y0 - grow,
                x1 - shrink if self.has_right else x1 + grow,
                y1 - shrink if self.x_axis_row is not None else y1 + grow)


def _close_gaps(mask: np.ndarray, axis: int, max_gap: int) -> np.ndarray:
    """沿 axis 方向把长度 <= max_gap 的 False 断口填上（曲线穿过坐标轴会切断轴线）。"""
    m = mask if axis == 1 else mask.T
    n = m.shape[1]
    big = n + 1
    fwd = np.empty(m.shape, dtype=np.int32)
    bwd = np.empty(m.shape, dtype=np.int32)
    d = np.full(m.shape[0], big, dtype=np.int32)
    for j in range(n):
        d = np.where(m[:, j], 0, d + 1)
        fwd[:, j] = d
    d = np.full(m.shape[0], big, dtype=np.int32)
    for j in range(n - 1, -1, -1):
        d = np.where(m[:, j], 0, d + 1)
        bwd[:, j] = d
    out = m | ((fwd + bwd) <= max_gap + 1)
    return out if axis == 1 else out.T


def _longest_runs(mask: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 方向，每条线上最长连续 True 段长度。axis=1 → 每行；axis=0 → 每列。"""
    m = mask if axis == 1 else mask.T
    run = np.zeros(m.shape[0], dtype=np.int32)
    best = np.zeros(m.shape[0], dtype=np.int32)
    for j in range(m.shape[1]):
        run = (run + 1) * m[:, j]
        best = np.maximum(best, run)
    return best


def _group_lines(idx: np.ndarray) -> List[Tuple[int, int]]:
    """把相邻的行/列号分组（粗线占多行）。"""
    if idx.size == 0:
        return []
    groups, start, prev = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v != prev + 1:
            groups.append((start, prev))
            start = v
        prev = v
    groups.append((start, prev))
    return groups


def _run_extent(line: np.ndarray) -> Tuple[int, int]:
    """一行/一列里最长 True 段的起止下标。"""
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return 0, line.size - 1
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    k = int(np.argmax(ends - starts))
    return int(idx[starts[k]]), int(idx[ends[k]])


def find_plot_frame(rgb: np.ndarray) -> PlotFrame:
    h, w = rgb.shape[:2]
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    dark = (mx < 170) & ((mx - mn) < 60)        # 深色且低饱和：坐标轴/边框
    gap_h, gap_v = max(10, int(0.04 * w)), max(10, int(0.04 * h))
    dark_h = _close_gaps(dark, axis=1, max_gap=gap_h)  # 允许曲线穿越造成的断口
    dark_v = _close_gaps(dark, axis=0, max_gap=gap_v)

    row_best = _longest_runs(dark_h, axis=1)
    col_best = _longest_runs(dark_v, axis=0)
    # 真实（未填补）暗像素数必须占填补后长度的大部分：一排刻度数字填补后也很长，但密度低
    row_cnt = dark.sum(axis=1)
    col_cnt = dark.sum(axis=0)
    rows = np.flatnonzero((row_best >= 0.4 * w) & (row_cnt >= 0.7 * row_best))
    cols = np.flatnonzero((col_best >= 0.4 * h) & (col_cnt >= 0.7 * col_best))
    gray = rgb.mean(axis=2)

    def line_intensity(idx, axis):
        line = gray[idx] if axis == 1 else gray[:, idx]
        d = dark[idx] if axis == 1 else dark[:, idx]
        return float(np.median(line[d])) if d.any() else 255.0

    def prefer_dark(idx, axis):
        """浅灰色的面板边框/网格线不应盖过真正的黑色坐标轴：
        只保留与最深者（中位灰度）相差 <= 80 的候选，再要求长度不短于其中最长者的 70%（排除图例框等短线）。"""
        if idx.size == 0:
            return idx
        inten = np.array([line_intensity(int(i), axis) for i in idx])
        idx = idx[inten <= inten.min() + 80]
        best = row_best if axis == 1 else col_best
        longest = max(int(best[i]) for i in idx)
        idx = np.array([i for i in idx if best[i] >= 0.7 * longest], dtype=idx.dtype)
        # 紧贴图像边缘的长线多半是裁剪后残留的面板边框（真正的坐标轴外侧要留刻度数字的位置）
        n = h if axis == 1 else w
        edge = max(8, int(0.025 * n))
        inner = idx[(idx > edge) & (idx < n - 1 - edge)]
        return inner if inner.size else idx

    rows = prefer_dark(rows, 1)
    cols = prefer_dark(cols, 0)
    rgroups = _group_lines(rows)
    cgroups = _group_lines(cols)

    x_axis_row = y_axis_col = None
    has_top = has_right = False
    x0 = y0 = 0
    x1, y1 = w - 1, h - 1

    if rgroups:
        g = rgroups[-1]                              # 最靠下的长横线 = X 轴
        x_axis_row = (g[0] + g[1]) // 2
        y1 = g[0] - 1
        a, b = _run_extent(dark_h[x_axis_row])
        x0, x1 = a, b
        if len(rgroups) >= 2 and (g[0] - rgroups[0][1]) > 0.2 * h:
            ta, tb = _run_extent(dark_h[(rgroups[0][0] + rgroups[0][1]) // 2])
            if abs(ta - a) <= 0.1 * w and abs(tb - b) <= 0.1 * w:   # 与 X 轴范围一致才是边框（排除图例框）
                y0 = rgroups[0][1] + 1               # 顶部边框
                has_top = True
    if cgroups:
        g = cgroups[0]                               # 最靠左的长竖线 = Y 轴
        y_axis_col = (g[0] + g[1]) // 2
        x0 = g[1] + 1
        a, b = _run_extent(dark_v[:, y_axis_col])
        if not (rgroups and len(rgroups) >= 2):
            y0 = a
        if x_axis_row is None:
            y1 = b
        if len(cgroups) >= 2 and (cgroups[-1][0] - g[1]) > 0.2 * w:
            ra, rb = _run_extent(dark_v[:, (cgroups[-1][0] + cgroups[-1][1]) // 2])
            if abs(ra - a) <= 0.1 * h and abs(rb - b) <= 0.1 * h:
                x1 = cgroups[-1][0] - 1              # 右侧边框
                has_right = True
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
        x_axis_row = y_axis_col = None
        has_top = has_right = False
    return PlotFrame(x_axis_row, y_axis_col, (x0, y0, x1, y1), has_top, has_right)


# ----------------------------------------------------------------------------
# 1b. 刻度线检测
# ----------------------------------------------------------------------------
def _tick_positions(dark: np.ndarray, line_idx: int, lo: int, hi: int, axis: int,
                    min_len: int = 3, max_len: int = 14) -> List[Tuple[float, int]]:
    """沿轴线找垂直于轴的小短线。返回 [(位置, 长度)]，位置为亚像素中心。
    axis=1: 轴线是行 line_idx，沿列扫描；axis=0: 轴线是列，沿行扫描。"""
    n = dark.shape[1] if axis == 1 else dark.shape[0]
    lo, hi = max(0, lo), min(n - 1, hi)
    best = []
    for side in (+1, -1):                       # 外侧 / 内侧
        lengths = np.zeros(hi - lo + 1, dtype=int)
        for k in range(2, max_len + 1):
            j = line_idx + side * k
            if not (0 <= j < (dark.shape[0] if axis == 1 else dark.shape[1])):
                break
            line = dark[j, lo:hi + 1] if axis == 1 else dark[lo:hi + 1, j]
            alive = (lengths == k - 2)              # 到目前为止连续
            lengths[alive & line] += 1
        cand = np.flatnonzero(lengths >= min_len)
        ticks = []
        for a, b in _group_lines(cand):
            if b - a + 1 <= 6:                       # 刻度线宽不会很粗（粗的是文字/曲线）
                ticks.append(((a + b) / 2.0 + lo, int(lengths[a:b + 1].max())))
        if len(ticks) >= 2:
            pos = np.array([t[0] for t in ticks])
            sp = np.diff(pos)
            reg = float(np.median(sp)) if sp.size else 0
            regular = (sp > 0.4 * reg).sum() if reg > 0 else 0
            score = (regular, len(ticks))
            if not best or score > best[0]:
                best = [score, ticks]
    if not best:
        return []
    ticks = best[1]
    # 主刻度：长度 >= 最长刻度的 60%
    mx = max(t[1] for t in ticks)
    major = [t for t in ticks if t[1] >= 0.6 * mx]
    return major if len(major) >= 2 else ticks


def _regular_subset(pos: List[float], tol_frac: float = 0.12) -> List[float]:
    """只保留落在等间距网格上的位置：以出现最多的相邻间距为基准，位置相对基准点的偏移
    必须接近基准间距的整数倍。少于 3 个在网格上 → 返回空（2 个无法判断规则性）。"""
    pos = sorted(pos)
    if len(pos) < 2:
        return []
    if len(pos) == 2:
        return pos
    best: List[float] = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            step = pos[j] - pos[i]
            if step <= 2:
                continue
            on = []
            for q in pos:
                k = round((q - pos[i]) / step)
                if abs((q - pos[i]) - k * step) <= tol_frac * step:
                    on.append(q)
            # 要求网格上相邻点大多是 1 步（不是跨好几步）
            if step < 8 or len(on) < 3 or len(on) <= len(best):
                continue
            sp = np.diff(on)
            ones = int((np.abs(sp - step) <= tol_frac * step).sum())
            # 至少一半、且不少于 2 个相邻间距正好等于基准间距（避免用 2 个点凑网格）
            if ones >= max(2, 0.5 * sp.size):
                best = on
    return best


def detect_ticks(rgb: np.ndarray, frame: PlotFrame) -> Tuple[List[float], List[float]]:
    """返回 (x 轴刻度像素列表(从左到右), y 轴刻度像素列表(从下到上))；只保留等间距的刻度。"""
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    dark = (mx < 170) & ((mx - mn) < 60)
    x0, y0, x1, y1 = frame.box
    xt, yt = [], []
    if frame.x_axis_row is not None:
        xt = _regular_subset([t[0] for t in _tick_positions(dark, frame.x_axis_row, x0 - 3, x1 + 3, axis=1)])
    if frame.y_axis_col is not None:
        yt = _regular_subset([t[0] for t in _tick_positions(dark, frame.y_axis_col, y0 - 3, y1 + 3, axis=0)])
        yt = sorted(yt, reverse=True)
    return sorted(xt), yt


def calibration_from_ticks(frame: PlotFrame, xticks_px: List[float], yticks_px: List[float],
                           xvals: Optional[List[float]], yvals: Optional[List[float]],
                           log_x: bool = False, log_y: bool = False,
                           xlabels_px: Optional[List[float]] = None,
                           ylabels_px: Optional[List[float]] = None):
    """
    用调用方给的刻度数值 + 检测到的刻度线位置做标定；刻度线不可用时退回到轴旁文本标签的位置
    （标签中心对准刻度）。值的个数 == 锚点个数：一一对应；只给 2 个值：对应最外侧两个锚点。
    返回 (AxisCalibration, 说明)。
    """
    cal = AxisCalibration(log_x=log_x, log_y=log_y)
    log: List[str] = []

    def match(vals, ticks, labels_px, name):
        if vals is None:
            return None
        sources = [("刻度线", ticks or []), ("标签位置", labels_px or [])]
        # 个数恰好相等的来源优先；其次首尾匹配（要求该来源 >= 2 个锚点）
        for src_name, anchors in sources:
            if len(anchors) >= 2 and len(vals) == len(anchors):
                log.append(f"{name} 轴：{len(vals)} 个给定刻度值与 {len(anchors)} 个{src_name}一一对应")
                return list(zip(anchors, vals))
        if len(vals) == 2:
            for src_name, anchors in sources:
                if len(anchors) >= 2:
                    log.append(f"{name} 轴：首尾两个刻度值对应最外侧的{src_name}"
                               f"（{anchors[0]:.0f} px 与 {anchors[-1]:.0f} px，共 {len(anchors)} 个锚点，请核对）")
                    return [(anchors[0], vals[0]), (anchors[-1], vals[1])]
        desc = "；".join(f"{n} {len(a)} 个" + (f"（{', '.join('%.0f' % t for t in a)}）" if a else "")
                        for n, a in sources)
        log.append(f"{name} 轴：给了 {len(vals)} 个刻度值，但 {desc}，无法匹配；"
                   f"请给同样个数的值、只给首尾两个值，或用 --cal 直接给像素位置")
        return None

    px = match(xvals, xticks_px, xlabels_px, "X")
    py = match(yvals, yticks_px, ylabels_px, "Y")
    if px:
        cal.x1_px, cal.x1_val = px[0]
        cal.x2_px, cal.x2_val = px[-1]
    if py:
        cal.y1_px, cal.y1_val = py[0]
        cal.y2_px, cal.y2_val = py[-1]
    return cal, log


def parse_cal_spec(spec: str) -> AxisCalibration:
    """'x:103=5.2,648=6.0;y:461=-30,24=0'  →  AxisCalibration（像素=数值）。"""
    cal = AxisCalibration()
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        axis, _, body = part.partition(":")
        axis = axis.strip().lower()
        pairs = []
        for item in body.split(","):
            if "=" in item:
                a, b = item.split("=")
                pairs.append((float(a), float(b)))
        if len(pairs) < 2:
            raise ValueError(f"--cal 的 {axis} 轴需要两个 像素=数值 对")
        if axis.startswith("logx") or axis.startswith("logy"):
            setattr(cal, "log_" + axis[3], True)
            axis = axis[3]
        if axis == "x":
            cal.x1_px, cal.x1_val = pairs[0]
            cal.x2_px, cal.x2_val = pairs[1]
        elif axis == "y":
            cal.y1_px, cal.y1_val = pairs[0]
            cal.y2_px, cal.y2_val = pairs[1]
    return cal


# ----------------------------------------------------------------------------
# 2. OCR 读数字
# ----------------------------------------------------------------------------
@dataclass
class NumLabel:
    value: float
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float


_NUM_RE = re.compile(r"[-−–]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+)?")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_POW_RE = re.compile(r"^\s*10\s*[\^]\s*([-+]?\d+)\s*$")
_ocr_engine = None


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _parse_number(text: str) -> Optional[float]:
    t = text.replace("−", "-").replace("–", "-").replace("，", ",").replace("。", ".")
    t = t.replace("O", "0").replace("o", "0")
    # 10^3 / 10³ / 10^-2 / 10⁻²  →  1000 / 0.01
    if any(ch in text for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"):
        sup = "".join(ch for ch in text if ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺").translate(_SUP)
        base = text.split(next(ch for ch in text if ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"))[0].strip()
        if base in ("10", "1O", "l0") and sup.lstrip("+-").isdigit():
            return 10.0 ** int(sup)
        return None
    m = _POW_RE.match(t)
    if m:
        return 10.0 ** int(m.group(1))
    m = _NUM_RE.search(t)
    if not m:
        return None
    s = m.group(0).replace(",", ".")
    # 只有当整段文本基本就是这个数字时才接受（避免 "Fig.2" 之类）
    rest = re.sub(r"[\s\-−–:;,.%°]", "", t[:m.start()] + t[m.end():])
    if len(rest) > 1:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ocr_tokens(rgb: np.ndarray, scales=None) -> List[NumLabel]:
    """在多个放大倍率下 OCR 所有文本；value 为解析出的数值，解析不了则为 nan。
    小图（短边 < 500）用 2x/3x 放大，否则 1x/2x。"""
    from PIL import Image
    ocr = _get_ocr()
    img = Image.fromarray(rgb)
    if scales is None:
        scales = (2, 3) if min(img.size) < 500 else (1, 2)
    labels: List[NumLabel] = []
    for s in scales:
        big = img if s == 1 else img.resize((img.width * s, img.height * s), Image.BICUBIC)
        res, _ = ocr(np.asarray(big))
        for box, text, score in (res or []):
            v = _parse_number(text)
            xs = [p[0] / s for p in box]
            ys = [p[1] / s for p in box]
            labels.append(NumLabel(float("nan") if v is None else v, sum(xs) / 4, sum(ys) / 4,
                                   min(xs), min(ys), max(xs), max(ys), str(text), float(score)))
    return labels


def ocr_numbers(rgb: np.ndarray, scales=None) -> List[NumLabel]:
    """只保留能解析成数值的 OCR 结果。"""
    return [l for l in ocr_tokens(rgb, scales) if not math.isnan(l.value)]


def _dedupe_positions(labels: List[NumLabel], key, tol: float) -> List[NumLabel]:
    """多倍率 OCR 会把同一标签识别多次：按位置去重，保留得分最高者。"""
    out: List[NumLabel] = []
    for l in sorted(labels, key=lambda l: -l.score):
        if all(abs(key(l) - key(o)) > tol for o in out):
            out.append(l)
    return sorted(out, key=key)


def axis_band_labels(rgb: np.ndarray, frame: PlotFrame, labels: List[NumLabel]):
    """分别取 X 轴下方、Y 轴左侧最靠近轴的那一排文本（可含无法解析的），按位置去重。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = frame.box
    med_h = float(np.median([l.y1 - l.y0 for l in labels])) if labels else 14.0
    if frame.x_axis_row is not None:
        ax = frame.x_axis_row
        xc = [l for l in labels if ax < l.cy <= ax + max(60, 0.15 * h)
              and x0 - 0.05 * w <= l.cx <= x1 + 0.05 * w]
    else:
        xc = [l for l in labels if l.cy > 0.75 * h]
    xc = _select_band(xc, lambda l: l.cy, (frame.x_axis_row or h), med_h * 0.8)
    if frame.y_axis_col is not None:
        ay = frame.y_axis_col
        yc = [l for l in labels if ay - max(100, 0.2 * w) <= l.x1 < ay + 2
              and y0 - 0.05 * h <= l.cy <= y1 + 0.05 * h
              and not (frame.x_axis_row is not None and l.cy > frame.x_axis_row + med_h * 0.4)]
    else:
        yc = [l for l in labels if l.cx < 0.25 * w]
    yc = _select_band(yc, lambda l: l.x1, (frame.y_axis_col or 0), med_h * 1.2)
    xc = _dedupe_positions(xc, lambda l: l.cx, med_h * 0.5)
    yc = _dedupe_positions(yc, lambda l: l.cy, med_h * 0.5)
    return xc, yc


# ----------------------------------------------------------------------------
# 3. 稳健拟合：数值 -> 像素位置
# ----------------------------------------------------------------------------
@dataclass
class AxisFit:
    log: bool
    slope: float          # px = slope * f(value) + intercept，f = log10 或恒等
    intercept: float
    inliers: List[int] = field(default_factory=list)
    n_total: int = 0

    def px(self, v: float) -> float:
        return self.slope * (math.log10(v) if self.log else v) + self.intercept


def fit_axis(values: List[float], positions: List[float], tol_px: float = 4.0) -> Optional[AxisFit]:
    """RANSAC 式配对投票：每对样本给出一条直线，取内点最多者再最小二乘。"""
    best: Optional[AxisFit] = None
    n_total = len(values)
    for use_log in (False, True):
        idx = [i for i, v in enumerate(values) if (v > 0 or not use_log)]
        if len(idx) < 2:
            continue
        f = np.array([math.log10(values[i]) if use_log else values[i] for i in idx])
        p = np.array([positions[i] for i in idx])
        cand = None
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if f[a] == f[b] or p[a] == p[b]:
                    continue
                slope = (p[b] - p[a]) / (f[b] - f[a])
                inter = p[a] - slope * f[a]
                resid = np.abs(slope * f + inter - p)
                inl = np.flatnonzero(resid <= tol_px)
                # 同一数值出现两次（误读/重复）只算一次
                uniq = len(set(round(f[i], 9) for i in inl))
                key = (uniq, -float(resid[inl].sum()))
                if cand is None or key > cand[0]:
                    cand = (key, inl, slope, inter)
        if cand is None:
            continue
        _, inl, slope, inter = cand
        if len(inl) >= 2:
            A = np.vstack([f[inl], np.ones(len(inl))]).T
            slope, inter = np.linalg.lstsq(A, p[inl], rcond=None)[0]
        fit = AxisFit(use_log, float(slope), float(inter), [idx[i] for i in inl], n_total)
        uniq = len(set(round(f[i], 9) for i in inl))
        if use_log and best is not None:
            # 对数模型只有在明显更好时才采用（线性能解释的数据优先线性）
            if uniq <= len(set(round(values[i], 9) for i in best.inliers)):
                continue
        if best is None or uniq > len(set(round(values[i], 9) for i in best.inliers)):
            best = fit
    return best


def _select_band(labels: List[NumLabel], key, anchor: float, tol: float) -> List[NumLabel]:
    """按 key(label) 聚成若干条带，返回离 anchor 最近的那条带。"""
    if not labels:
        return []
    labs = sorted(labels, key=key)
    bands, cur = [], [labs[0]]
    for l in labs[1:]:
        if key(l) - key(cur[-1]) <= tol:
            cur.append(l)
        else:
            bands.append(cur)
            cur = [l]
    bands.append(cur)
    # 优先选文本数 >= 2 的那一排（避免另一根轴的端点标签单独成排抢位），再按距离
    return min(bands, key=lambda b: (len(b) < 2, abs(np.median([key(l) for l in b]) - anchor)))


def auto_calibrate(rgb: np.ndarray, frame: PlotFrame, labels: List[NumLabel],
                   xticks_px: Optional[List[float]] = None, yticks_px: Optional[List[float]] = None):
    """返回 (AxisCalibration | None, 说明文字列表)。给出刻度线位置时，标签位置吸附到最近的刻度线。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = frame.box
    log: List[str] = []
    med_h = float(np.median([l.y1 - l.y0 for l in labels])) if labels else 14.0

    # ---- X 轴标签：轴线下方、绘图区水平范围内 ----
    if frame.x_axis_row is not None:
        ax = frame.x_axis_row
        xc = [l for l in labels if ax < l.cy <= ax + max(60, 0.15 * h)
              and x0 - 0.05 * w <= l.cx <= x1 + 0.05 * w]
    else:
        xc = [l for l in labels if l.cy > 0.75 * h]
    xc = _select_band(xc, lambda l: l.cy, (frame.x_axis_row or h), med_h * 0.8)

    # ---- Y 轴标签：轴线左侧、绘图区垂直范围内（按右边缘聚带） ----
    if frame.y_axis_col is not None:
        ay = frame.y_axis_col
        yc = [l for l in labels if ay - max(100, 0.2 * w) <= l.x1 < ay + 2
              and y0 - 0.05 * h <= l.cy <= y1 + 0.05 * h
              and not (frame.x_axis_row is not None and l.cy > frame.x_axis_row + med_h * 0.4)]
    else:
        yc = [l for l in labels if l.cx < 0.25 * w]
    yc = _select_band(yc, lambda l: l.x1, (frame.y_axis_col or 0), med_h * 1.2)

    def snap(pos, ticks, span):
        """标签中心若在某刻度线附近（半个标签尺寸内），用刻度线位置代替。"""
        if not ticks:
            return pos
        t = min(ticks, key=lambda t: abs(t - pos))
        return t if abs(t - pos) <= 0.6 * span else pos

    cal = AxisCalibration()
    ok = True
    for name, cands, pos_key, ticks, span_key in (
            ("X", xc, lambda l: l.cx, xticks_px, lambda l: l.x1 - l.x0),
            ("Y", yc, lambda l: l.cy, yticks_px, lambda l: l.y1 - l.y0)):
        if len(cands) < 2:
            log.append(f"{name} 轴：只识别到 {len(cands)} 个刻度数字，无法自动标定")
            ok = False
            continue
        positions = [snap(pos_key(l), ticks, span_key(l)) for l in cands]
        fit = fit_axis([l.value for l in cands], positions,
                       tol_px=max(3.0, med_h * 0.3))
        if fit is None or len(fit.inliers) < 2 or (len(cands) >= 3 and len(fit.inliers) < 3):
            log.append(f"{name} 轴：{len(cands)} 个刻度数字无法拟合成一致的刻度（可能误读），请手动标定")
            ok = False
            continue
        vals = sorted({cands[i].value for i in fit.inliers})
        v1, v2 = vals[0], vals[-1]
        p1, p2 = fit.px(v1), fit.px(v2)
        if name == "X":
            cal.x1_px, cal.x1_val, cal.x2_px, cal.x2_val, cal.log_x = p1, v1, p2, v2, fit.log
        else:
            cal.y1_px, cal.y1_val, cal.y2_px, cal.y2_val, cal.log_y = p1, v1, p2, v2, fit.log
        used = ", ".join("%g" % v for v in vals)
        log.append(f"{name} 轴：{'对数' if fit.log else '线性'}，用 {len(fit.inliers)}/{len(cands)} 个刻度 [{used}]")
    if not ok or cal.validate():
        return None, log
    return cal, log


# ----------------------------------------------------------------------------
# 3b. 插图/照片区域检测（实心大块 → 排除）
# ----------------------------------------------------------------------------
def _min_filter_1d(m: np.ndarray, k: int, axis: int) -> np.ndarray:
    out = m.copy()
    for s_ in range(1, k + 1):
        a = np.roll(m, s_, axis=axis)
        b = np.roll(m, -s_, axis=axis)
        if axis == 0:
            a[:s_] = True; b[-s_:] = True
        else:
            a[:, :s_] = True; b[:, -s_:] = True
        out &= a & b
    return out


def _max_filter_1d(m: np.ndarray, k: int, axis: int) -> np.ndarray:
    out = m.copy()
    for s_ in range(1, k + 1):
        a = np.roll(m, s_, axis=axis)
        b = np.roll(m, -s_, axis=axis)
        if axis == 0:
            a[:s_] = False; b[-s_:] = False
        else:
            a[:, :s_] = False; b[:, -s_:] = False
        out |= a | b
    return out


def erode(m: np.ndarray, k: int) -> np.ndarray:
    """方形结构元 (2k+1)^2 腐蚀。"""
    return _min_filter_1d(_min_filter_1d(m, k, 0), k, 1)


def dilate(m: np.ndarray, k: int) -> np.ndarray:
    return _max_filter_1d(_max_filter_1d(m, k, 0), k, 1)


def background_color(rgb: np.ndarray) -> np.ndarray:
    """出现最多的颜色（量化到 16 级）作为背景色。"""
    q = (rgb >> 4).reshape(-1, 3).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    sel = key == vals[np.argmax(cnt)]
    return rgb.reshape(-1, 3)[sel].mean(axis=0)


def dense_mask(rgb: np.ndarray, thin: int = 8, margin: int = 5, bg_tol: int = 40) -> np.ndarray:
    """
    找出“实心大块”区域（照片、模型图、填充色块）：
    非背景像素 → 腐蚀掉细于 2*thin+1 像素的线条/文字/标记 → 再膨胀回去并外扩 margin。
    曲线（线宽几像素）和数据标记（<= 2*thin 像素）不会被标记。
    """
    bg = background_color(rgb)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    fg = diff > bg_tol
    core = erode(fg, thin)
    if not core.any():
        return np.zeros(fg.shape, dtype=bool)
    return dilate(core, thin + margin)


# ----------------------------------------------------------------------------
# 4. 自动找曲线颜色并提取
# ----------------------------------------------------------------------------
def detect_curve_colors(rgb: np.ndarray, box: Tuple[int, int, int, int],
                        min_count: Optional[int] = None, max_colors: int = 10,
                        merge_dist: float = 45.0, include_dark: bool = True,
                        exclude: Optional[np.ndarray] = None, drop_halo: bool = True) -> List[Tuple[int, int, int]]:
    x0, y0, x1, y1 = box
    sub = rgb[y0:y1 + 1, x0:x1 + 1].astype(np.int16)
    mx, mn = sub.max(axis=2), sub.min(axis=2)
    sat = mx - mn
    sel = sat >= 50                       # 彩色
    if include_dark:
        sel |= (mx < 90)                  # 黑色/深色曲线
    if exclude is not None:
        sel &= ~exclude[y0:y1 + 1, x0:x1 + 1]
    pix = sub[sel]
    if pix.size == 0:
        return []
    q = (pix // 24).astype(np.int32)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.zeros((uk.size, 3))
    np.add.at(means, inv, pix)
    means /= cnt[:, None]
    order = np.argsort(-cnt)
    clusters: List[List] = []            # [mean(3), count]
    for i in order:
        c, n = means[i], int(cnt[i])
        for cl in clusters:
            if np.linalg.norm(cl[0] - c) <= merge_dist:
                cl[0] = (cl[0] * cl[1] + c * n) / (cl[1] + n)
                cl[1] += n
                break
        else:
            clusters.append([c.copy(), n])
    if min_count is None:
        min_count = max(30, int(0.3 * (x1 - x0)))
    clusters = [cl for cl in clusters if cl[1] >= min_count]
    clusters.sort(key=lambda cl: -cl[1])
    colors = [tuple(int(round(v)) for v in cl[0]) for cl in clusters[:max_colors]]
    if drop_halo and len(colors) > 1:
        # 抗锯齿/JPEG 使曲线边缘出现一圈浅色，会形成独立的小簇：若某簇像素大多紧贴着
        # 另一个更深/更饱和簇的像素，则它只是“影子”，丢弃
        sub_u8 = rgb[y0:y1 + 1, x0:x1 + 1]
        masks = [color_mask(sub_u8, c, 40) for c in colors]
        if exclude is not None:
            ex_sub = exclude[y0:y1 + 1, x0:x1 + 1]
            masks = [m & ~ex_sub for m in masks]
        def strength(c):
            return (max(c) - min(c)) + (255 - max(c))      # 饱和度 + 深度
        keep = []
        for i, ci in enumerate(colors):
            halo = False
            for j, cj in enumerate(colors):
                if i == j or strength(cj) <= strength(ci):
                    continue
                near = dilate(masks[j], 2)
                if masks[i].sum() > 0 and (masks[i] & near).sum() >= 0.6 * masks[i].sum():
                    halo = True
                    break
            if not halo:
                keep.append(ci)
        colors = keep
    return colors


def _pick_seed(remaining: np.ndarray, box) -> Optional[Tuple[int, int]]:
    """在剩余掩码里挑一个种子：优先只有一段像素、且靠近中间的列。"""
    x0, y0, x1, y1 = box
    sub = remaining[y0:y1 + 1, x0:x1 + 1]
    colcount = sub.sum(axis=0)
    if colcount.max() == 0:
        return None
    starts = sub & ~np.vstack([np.zeros((1, sub.shape[1]), bool), sub[:-1]])
    nruns = starts.sum(axis=0)
    single = np.flatnonzero((nruns == 1) & (colcount >= 1))
    if single.size:
        mid = sub.shape[1] / 2
        c = int(single[np.argmin(np.abs(single - mid))])
    else:
        c = int(np.argmax(colcount))
    rows = np.flatnonzero(sub[:, c])
    # 取最长的一段的中心
    breaks = np.flatnonzero(np.diff(rows) > 1)
    s = np.concatenate(([0], breaks + 1)); e = np.concatenate((breaks, [rows.size - 1]))
    k = int(np.argmax(e - s))
    r = int((rows[s[k]] + rows[e[k]]) // 2)
    return x0 + c, y0 + r


def auto_extract_curves(rgb: np.ndarray, box: Tuple[int, int, int, int],
                        colors: Optional[List[Tuple[int, int, int]]] = None,
                        tol: float = 50, max_jump: float = 20, max_gap: int = 30, max_run: int = 80,
                        min_span_frac: float = 0.2, attempts_per_color: int = 4,
                        exclude: Optional[np.ndarray] = None, min_density: float = 0.25) -> List[Curve]:
    """对每种颜色，反复挑种子追踪，保留跨度足够长的轨迹。box 应已排除坐标轴线本身。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    inner = (max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1))
    if colors is None:
        colors = detect_curve_colors(rgb, inner, exclude=exclude)
    curves: List[Curve] = []
    width = inner[2] - inner[0] + 1

    def is_dup(cols, rows, others):
        """新轨迹的大部分点（>=60%）都紧贴（<6 px）某条已接受轨迹 → 视为重复
        （抗锯齿边缘色会复制出“影子线”，甚至在两条线之间来回切换）。"""
        if not others:
            return False
        near = np.zeros(cols.size, dtype=bool)
        for pc, pr in others:
            common, ia, ib = np.intersect1d(cols, pc, return_indices=True)
            if common.size:
                near[ia] |= np.abs(rows[ia] - pr[ib]) < 6
        return near.mean() >= 0.6

    all_traces = []                      # 跨颜色去重用（抗锯齿边缘色会复制同一条线）
    for color in colors:
        mask = color_mask(rgb, color, tol)
        if exclude is not None:
            mask &= ~exclude
        remaining = mask.copy()
        found = []
        for _ in range(attempts_per_color):
            seed = _pick_seed(remaining, inner)
            if seed is None:
                break
            cols, rows = trace_curve(mask, seed[0], seed[1], region=inner,
                                     max_jump=max_jump, max_gap=max_gap, max_run=max_run,
                                     rgb=rgb, color=color)
            if cols.size == 0:
                remaining[seed[1], seed[0]] = False
                continue
            # 从剩余掩码里抹掉这条轨迹附近的像素
            for c, r in zip(cols.astype(int), rows):
                r = int(round(r))
                remaining[max(0, r - 4):r + 5, c] = False
            remaining[:, int(seed[0])] = False
            span_cols = cols.max() - cols.min() + 1
            span = span_cols / width
            if span < min_span_frac:
                continue
            if cols.size < min_density * span_cols:      # 零零碎碎的点（文字、噪点）不是曲线
                continue
            # 几乎全是同一水平线（边框/图例框/网格线，拐角处会拖出一小段竖线）：不是数据曲线
            if cols.size > 20 and np.mean(np.abs(rows - np.median(rows)) <= 1.0) >= 0.85:
                continue
            if not is_dup(cols, rows, found) and not is_dup(cols, rows, all_traces):
                found.append((cols, rows))
        # 同色且列范围互不重叠的轨迹 → 很可能是同一条被长间断切开的曲线，合并
        found.sort(key=lambda t: t[0][0])
        merged = []
        for cols, rows in found:
            if merged and cols[0] > merged[-1][0][-1]:
                merged[-1] = (np.concatenate([merged[-1][0], cols]), np.concatenate([merged[-1][1], rows]))
            else:
                merged.append((cols, rows))
        all_traces += merged
        for cols, rows in merged:
            name = f"曲线{len(curves) + 1}"
            curves.append(Curve(name=name, color=color, px=cols, py=rows,
                                seed=(int(cols[cols.size // 2]), int(round(rows[cols.size // 2])))))
    return curves


# ----------------------------------------------------------------------------
# 5. 一键流程
# ----------------------------------------------------------------------------
def curve_quality(curve: Curve, plot_width: int) -> dict:
    """覆盖率（跨度/绘图区宽）、点密度（点数/跨度）、断口数（>=5 列的缺失段）。"""
    if len(curve) == 0:
        return {"coverage": 0.0, "density": 0.0, "gaps": 0}
    cols = np.sort(curve.px)
    span = float(cols[-1] - cols[0] + 1)
    d = np.diff(cols)
    return {"coverage": round(span / max(1, plot_width), 3),
            "density": round(len(curve) / span, 3),
            "gaps": int((d >= 5).sum())}


_LEGEND_SKIP = re.compile(r"^[\s\-−–=.:;,()\[\]|_]*$")


def name_curves_by_legend(rgb: np.ndarray, tokens: List[NumLabel], curves: List[Curve],
                          tol: float = 50) -> List[str]:
    """
    用图例给曲线命名：对每段 OCR 文本，看它左侧一小段（图例里的线条/标记样本）里
    哪条曲线的颜色最多，就把该文本作为那条曲线的名字。返回说明日志。
    """
    log: List[str] = []
    if not tokens or not curves:
        return log
    h, w = rgb.shape[:2]
    cand = []
    for t in tokens:
        text = t.text.strip().lstrip("=-−–_|:;.*·•~ ")
        if not math.isnan(t.value) or len(text) < 2 or t.score < 0.5 or _LEGEND_SKIP.match(text):
            continue
        if len(set(text)) < 3 or len(set(text)) < 0.5 * len(text) or sum(ch.isalnum() for ch in text) < 2:
            continue                                  # 标记串被误读成 "OTOTOTO" 之类的垃圾文本
        if re.match(r"^\(?[a-zA-Z]\)$", text) or re.match(r"^\([a-zA-Z0-9]\)$", text):
            continue                                  # (a) (b) 这类子图编号
        th = t.y1 - t.y0
        if th > 40 or th < 5:
            continue
        x1 = int(t.x0) - 2
        x0 = max(0, int(t.x0 - max(50, 5 * th)))
        y0, y1 = max(0, int(t.y0)), min(h, int(t.y1) + 1)
        if x1 - x0 < 8 or y1 - y0 < 3:
            continue
        strip = rgb[y0:y1, x0:x1]
        for ci, c in enumerate(curves):
            n = int(color_mask(strip, c.color, tol).sum())
            if n >= 4:
                cand.append((n, ci, text))
    cand.sort(reverse=True)
    used_curve, used_text = set(), set()
    for n, ci, text in cand:
        if ci in used_curve or text in used_text:
            continue
        used_curve.add(ci); used_text.add(text)
        name = text.replace(",", " ").replace(";", " ").strip()
        log.append(f"{curves[ci].name} → '{name}'（图例，{n} 像素匹配）")
        curves[ci].meta["legend"] = name
        curves[ci].name = name
    # 去重名
    seen = {}
    for c in curves:
        if c.name in seen:
            seen[c.name] += 1
            c.name = f"{c.name}_{seen[c.name]}"
        else:
            seen[c.name] = 1
    return log


def text_exclude_mask(rgb: np.ndarray, tokens: List[NumLabel], frame: PlotFrame) -> np.ndarray:
    """
    绘图区内的文字（图例条目、标注）不是曲线数据：把这些文本框、以及文本左侧紧邻的
    图例样条（短横线/标记，位于文字中线附近）加入排除掩码。
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = frame.box
    bg = background_color(rgb)
    for t in tokens:
        text = t.text.strip()
        th = t.y1 - t.y0
        if not math.isnan(t.value) or len(text) < 2 or th > 40 or th < 5:
            continue
        # 带标记的曲线会被 OCR 误读成 "OTOTOTO" 之类：要求置信度高、字符种类 >= 3、字母数字 >= 2
        if t.score < 0.8 or len(set(text)) < 3 or sum(ch.isalnum() for ch in text) < 2                 or len(set(text)) < 0.5 * len(text):
            continue
        if not (x0 - 5 <= t.cx <= x1 + 5 and y0 - 5 <= t.cy <= y1 + 5):
            continue                                   # 只处理绘图区内的文字
        ty0, ty1 = max(0, int(t.y0) - 2), min(h, int(t.y1) + 3)
        tx0, tx1 = max(0, int(t.x0) - 2), min(w, int(t.x1) + 3)
        mask[ty0:ty1, tx0:tx1] = True
        # 左侧样条：内容须集中在文字中线附近（图例样条），否则可能是穿过的真曲线，不排除
        sx0 = max(0, int(t.x0 - max(50, 5 * th)))
        sx1 = max(sx0, int(t.x0) - 2)
        if sx1 - sx0 < 8:
            continue
        strip = rgb[ty0:ty1, sx0:sx1]
        fg = np.abs(strip.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 40
        if fg.sum() < 4:
            continue
        rows = np.flatnonzero(fg.any(axis=1))
        cy = (ty0 + ty1) / 2 - ty0
        if rows.min() >= cy - 0.6 * th and rows.max() <= cy + 0.6 * th:
            mask[ty0:ty1, sx0:sx1] = True
    return mask


@dataclass
class AutoResult:
    frame: PlotFrame
    calibration: Optional[AxisCalibration]
    curves: List[Curve]
    labels: List[NumLabel]
    log: List[str]
    exclude: Optional[np.ndarray] = None


def auto_digitize(rgb: np.ndarray, use_ocr: bool = True, tol: float = 50,
                  max_jump: float = 20, max_gap: int = 30, max_run: int = 80,
                  progress=None, exclude_dense: bool = True,
                  calibration: Optional[AxisCalibration] = None,
                  xticks: Optional[List[float]] = None, yticks: Optional[List[float]] = None,
                  log_x: bool = False, log_y: bool = False,
                  frame: Optional[PlotFrame] = None) -> AutoResult:
    """
    calibration : 直接给定的标定（最高优先级）
    xticks/yticks : 调用方读出的刻度数值（全部、或首尾两个），与检测到的刻度线匹配
    否则用 OCR 自动读刻度。
    frame : 已知的坐标轴框（多子图拆分时沿用拆分结果，不再重新检测）
    """
    def report(msg):
        if progress:
            progress(msg)

    if frame is None:
        report("正在检测坐标轴框…")
        frame = find_plot_frame(rgb)
    log = [f"绘图区: x {frame.box[0]}~{frame.box[2]}, y {frame.box[1]}~{frame.box[3]}"
           + (f"，X 轴行 {frame.x_axis_row}" if frame.x_axis_row is not None else "，未找到 X 轴线")
           + (f"，Y 轴列 {frame.y_axis_col}" if frame.y_axis_col is not None else "，未找到 Y 轴线")]
    xt, yt = detect_ticks(rgb, frame)
    if xt:
        log.append(f"X 轴刻度线 {len(xt)} 个: " + ", ".join("%.0f" % t for t in xt))
    if yt:
        log.append(f"Y 轴刻度线 {len(yt)} 个(从下到上): " + ", ".join("%.0f" % t for t in yt))
    cal, labels = None, []
    if calibration is not None:
        err = calibration.validate()
        if err:
            log.append("给定标定无效：" + err)
        else:
            cal = calibration
            log.append("使用调用方给定的标定")
    elif xticks or yticks:
        xl_px = yl_px = None
        if use_ocr and ocr_available():
            try:
                report("正在 OCR 定位轴旁文本…")
                labels = ocr_tokens(rgb)
                xl, yl = axis_band_labels(rgb, frame, labels)
                xl_px = [l.cx for l in xl]
                yl_px = sorted([l.cy for l in yl], reverse=True)
            except Exception as ex:
                log.append(f"OCR 出错：{ex}")
        cal, clog = calibration_from_ticks(frame, xt, yt, xticks, yticks, log_x, log_y, xl_px, yl_px)
        log += clog
        if cal.validate():
            # 只给了一个轴：另一个轴尝试 OCR
            partial = cal
            cal = None
            if use_ocr and ocr_available():
                report("正在 OCR 识别刻度数字…")
                try:
                    labels = labels or ocr_tokens(rgb)
                    ocal, olog = auto_calibrate(rgb, frame, [l for l in labels if not math.isnan(l.value)], xt, yt)
                    log += olog
                    merged = AxisCalibration(log_x=log_x, log_y=log_y)
                    for axis in ("x", "y"):
                        src = partial if getattr(partial, axis + "_ready")() else ocal
                        if src is None:
                            continue
                        for k in ("1_px", "1_val", "2_px", "2_val"):
                            setattr(merged, axis + k, getattr(src, axis + k))
                        if src is ocal:
                            setattr(merged, "log_" + axis, getattr(ocal, "log_" + axis))
                    if merged.validate() is None:
                        cal = merged
                except Exception as ex:
                    log.append(f"OCR 出错：{ex}")
            if cal is None:
                log.append("标定不完整：请同时给出 X 和 Y 的刻度值，或用 --cal")
    elif use_ocr:
        if ocr_available():
            report("正在 OCR 识别刻度数字…")
            try:
                labels = ocr_tokens(rgb)
                cal, clog = auto_calibrate(rgb, frame, [l for l in labels if not math.isnan(l.value)], xt, yt)
                log += clog
            except Exception as ex:          # OCR 失败不影响曲线提取
                log.append(f"OCR 出错：{ex}")
        else:
            log.append("未安装 rapidocr_onnxruntime，跳过刻度识别（pip install rapidocr_onnxruntime）")
    if cal is None and labels:
        try:
            xl, yl = axis_band_labels(rgb, frame, labels)
            if xl:
                log.append("X 轴旁文本(像素x@文本): " + ", ".join(f"{l.cx:.0f}@'{l.text}'" for l in xl))
            if yl:
                log.append("Y 轴旁文本(像素y@文本, 从下到上): "
                           + ", ".join(f"{l.cy:.0f}@'{l.text}'" for l in sorted(yl, key=lambda l: -l.cy)))
            log.append("可据此用 --cal 'x:px=值,px=值;y:px=值,px=值' 直接标定（对数轴用 logy:）")
        except Exception:
            pass
    exclude = None
    if exclude_dense:
        report("正在检测插图/照片区域…")
        exclude = dense_mask(rgb)
        frac = float(exclude[frame.box[1]:frame.box[3] + 1, frame.box[0]:frame.box[2] + 1].mean()) if exclude.any() else 0.0
        if frac > 0:
            log.append(f"检测到插图/照片等实心区域，占绘图区 {frac:.0%}，已排除")
        else:
            exclude = None
    if labels:
        tmask = text_exclude_mask(rgb, labels, frame)
        if tmask.any():
            exclude = tmask if exclude is None else (exclude | tmask)
            log.append("绘图区内的文字（图例/标注）已加入排除区")
    report("正在识别曲线…")
    curves = auto_extract_curves(rgb, frame.inner_box(), tol=tol, max_jump=max_jump, max_gap=max_gap,
                                 max_run=max_run, exclude=exclude)
    if curves and labels:
        log += name_curves_by_legend(rgb, labels, curves)
    pw = frame.box[2] - frame.box[0] + 1
    for c in curves:
        c.meta.update(curve_quality(c, pw))
    log.append(f"识别到 {len(curves)} 条曲线：" + "、".join(
        f"{c.name}{c.color} {len(c)}点[覆盖{c.meta['coverage']:.0%} 密度{c.meta['density']:.2f} 断口{c.meta['gaps']}]"
        for c in curves))
    if not curves and exclude is not None:
        log.append("提示：若曲线非常密集或很粗而被当成实心区域排除，可用 --no-exclude 重试")
    return AutoResult(frame, cal, curves, labels, log, exclude)


# ----------------------------------------------------------------------------
# 6. 多子图面板检测
# ----------------------------------------------------------------------------
@dataclass
class Panel:
    label: str                          # a, b, c ...
    box: Tuple[int, int, int, int]      # 坐标轴框 (x0, y0, x1, y1)
    crop: Tuple[int, int, int, int]     # 含刻度数字的裁剪框

    def frame(self) -> "PlotFrame":
        """相对裁剪图的坐标轴框。"""
        bx0, by0, bx1, by1 = self.box
        cx0, cy0 = self.crop[0], self.crop[1]
        return PlotFrame(x_axis_row=by1 - cy0, y_axis_col=bx0 - cx0,
                         box=(bx0 - cx0 + 1, by0 - cy0, bx1 - cx0, by1 - cy0 - 1))


def _runs_ge(line: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends) if idx[b] - idx[a] + 1 >= min_len]


def find_panels(rgb: np.ndarray, min_frac: float = 0.15, min_px: int = 50) -> List[Panel]:
    """
    找出图中所有坐标系：每个坐标系至少有一条 X 轴（长横线）和一条 Y 轴（长竖线）在左下角相接。
    返回按阅读顺序（先上后下、先左后右）编号 a, b, c… 的面板。
    """
    h, w = rgb.shape[:2]
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    dark = (mx < 170) & ((mx - mn) < 60)
    gap_h, gap_v = max(10, int(0.02 * w)), max(10, int(0.02 * h))
    dark_h = _close_gaps(dark, 1, gap_h)
    dark_v = _close_gaps(dark, 0, gap_v)
    min_w = max(min_px, int(min_frac * w))
    min_h = max(min_px, int(min_frac * h))

    def lines(closed, raw, axis, min_len):
        """[(idx, a, b)]：第 idx 行/列上长度 >= min_len 且真实密度 >= 0.7 的段，粗线合并。"""
        out = []
        n = closed.shape[0] if axis == 1 else closed.shape[1]
        cand = []
        for i in range(n):
            line = closed[i] if axis == 1 else closed[:, i]
            rl = raw[i] if axis == 1 else raw[:, i]
            for a, b in _runs_ge(line, min_len):
                if rl[a:b + 1].sum() >= 0.7 * (b - a + 1):
                    cand.append((i, a, b))
        # 合并相邻 idx 且范围重叠的段（线宽 > 1）
        cand.sort()
        for i, a, b in cand:
            if out and i - out[-1][0] <= 2 and min(b, out[-1][2]) - max(a, out[-1][1]) > 0.8 * min(b - a, out[-1][2] - out[-1][1]):
                pi, pa, pb = out[-1]
                out[-1] = (i, min(a, pa), max(b, pb))
            else:
                out.append((i, a, b))
        return out

    hl = lines(dark_h, dark, 1, min_w)      # (row, c0, c1)
    vl = lines(dark_v, dark, 0, min_h)      # (col, r0, r1)
    boxes = []
    tol = max(10, int(0.04 * min(w, h)))     # 刻度线会让轴线端点多出一截
    for col, r0, r1 in vl:
        for row, c0, c1 in hl:
            if abs(col - c0) <= tol and abs(row - r1) <= tol and c1 - col >= min_w * 0.8 and row - r0 >= min_h * 0.8:
                boxes.append((col, r0, c1, row))
    # 去重 / 去掉嵌套的小框（图例框）
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    dense = dense_mask(rgb)
    keep: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        # 框内大部分是实心区域（照片/插图的边缘被当成了轴线）→ 不是坐标系
        if dense[b[1]:b[3] + 1, b[0]:b[2] + 1].mean() >= 0.5:
            continue
        dup = False
        for j, k in enumerate(keep):
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            if inter >= 0.8 * area_b:              # b 基本被 k 包含
                if area_b >= 0.5 * area_k:         # 尺寸相近：k 是面板外边框，b 才是坐标轴
                    keep[j] = b
                dup = True                         # 否则 b 是图例框等，丢弃
                break
        if not dup:
            keep.append(b)
    if not keep:
        return []
    # 阅读顺序
    med_h = np.median([b[3] - b[1] for b in keep])
    keep.sort(key=lambda b: (round(b[1] / max(1.0, med_h * 0.5)), b[0]))
    panels = []
    for i, b in enumerate(keep):
        x0, y0, x1, y1 = b
        bw, bh = x1 - x0, y1 - y0
        L, B, T, R = max(90, int(0.3 * bw)), max(70, int(0.25 * bh)), max(25, int(0.1 * bh)), max(30, int(0.08 * bw))
        cx0, cy0, cx1, cy1 = x0 - L, y0 - T, x1 + R, y1 + B
        for o in keep:
            if o is b:
                continue
            ov_y = min(y1, o[3]) - max(y0, o[1]) > 0
            ov_x = min(x1, o[2]) - max(x0, o[0]) > 0
            if ov_y and o[2] <= x0:                    # 左邻
                cx0 = max(cx0, (o[2] + x0) // 2 + 1)
            if ov_y and o[0] >= x1:                    # 右邻
                cx1 = min(cx1, (x1 + o[0]) // 2 - 1)
            if ov_x and o[3] <= y0:                    # 上邻
                cy0 = max(cy0, (o[3] + y0) // 2 + 1)
            if ov_x and o[1] >= y1:                    # 下邻
                cy1 = min(cy1, (y1 + o[1]) // 2 - 1)
        crop = (max(0, cx0), max(0, cy0), min(w, cx1), min(h, cy1))
        panels.append(Panel(chr(ord("a") + i) if i < 26 else str(i + 1), b, crop))
    return panels


def digitize_panels(rgb: np.ndarray, panels: Optional[List[Panel]] = None, **kw):
    """逐个面板运行 auto_digitize。返回 [(Panel, 裁剪后 rgb, AutoResult)]。"""
    if panels is None:
        panels = find_panels(rgb)
    out = []
    for p in panels:
        x0, y0, x1, y1 = p.crop
        sub = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        res = auto_digitize(sub, frame=p.frame(), **kw)
        out.append((p, sub, res))
    return out


def save_preview(rgb: np.ndarray, result: AutoResult, path: str):
    """把识别结果画在图上，便于核对。"""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    if result.exclude is not None and result.exclude.any():
        arr = np.asarray(img).copy()
        m = result.exclude
        arr[m] = (arr[m] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = result.frame.box
    d.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=1)
    for l in result.labels:
        d.rectangle([l.x0, l.y0, l.x1, l.y1], outline=(255, 150, 0), width=1)
    try:
        xt, yt = detect_ticks(rgb, result.frame)
        for t in xt:
            d.line([(t, y1 + 1), (t, y1 + 12)], fill=(0, 160, 255), width=2)
        for t in yt:
            d.line([(x0 - 12, t), (x0 - 1, t)], fill=(0, 160, 255), width=2)
    except Exception:
        pass
    cal = result.calibration
    if cal:
        for px in (cal.x1_px, cal.x2_px):
            d.line([(px, y0), (px, y1)], fill=(255, 150, 0), width=1)
        for py in (cal.y1_px, cal.y2_px):
            d.line([(x0, py), (x1, py)], fill=(255, 150, 0), width=1)
    for c in result.curves:
        pts = list(zip(c.px.tolist(), c.py.tolist()))
        if len(pts) > 1:
            d.line(pts, fill=(255, 255, 255), width=3)
            d.line(pts, fill=(255, 0, 255), width=1)
    img.save(path)


def parse_crop(text: str, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """'x0,y0,x1,y1' -> 裁剪框；允许用 0~1 的比例值。"""
    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != 4:
        raise ValueError("--crop 需要 4 个数：x0,y0,x1,y1")
    w, h = size
    if all(0 <= v <= 1 for v in vals):
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    x0, y0, x1, y1 = [int(round(v)) for v in vals]
    return max(0, min(x0, x1)), max(0, min(y0, y1)), min(w, max(x0, x1)), min(h, max(y0, y1))


def main(argv=None):
    ap = argparse.ArgumentParser(description="全自动从图表图片提取曲线数据")
    ap.add_argument("image")
    ap.add_argument("-o", "--output", help="输出 CSV/TXT 路径（默认 <图片名>_data.csv）")
    ap.add_argument("--mode", choices=["separate", "common"], default="separate")
    ap.add_argument("--n", type=int, default=0, help="导出点数（0=全部）")
    ap.add_argument("--tol", type=float, default=50, help="颜色容差")
    ap.add_argument("--no-ocr", action="store_true", help="不做 OCR 标定，只输出像素坐标")
    ap.add_argument("--preview", help="把识别结果叠加画到此 PNG 便于核对")
    ap.add_argument("--crop", help="只处理图中某个子图区域，像素坐标 x0,y0,x1,y1（多子图时先裁出一个面板）")
    ap.add_argument("--xticks", help="X 轴刻度数值（逗号分隔，全部或首尾两个），与检测到的刻度线匹配，代替 OCR")
    ap.add_argument("--yticks", help="Y 轴刻度数值（从下到上，逗号分隔，全部或首尾两个）")
    ap.add_argument("--logx", action="store_true", help="X 轴为对数坐标（配合 --xticks）")
    ap.add_argument("--logy", action="store_true", help="Y 轴为对数坐标（配合 --yticks）")
    ap.add_argument("--cal", help="直接给像素标定：'x:103=5.2,648=6.0;y:461=-30,24=0'（对数轴写 logx:/logy:）")
    ap.add_argument("--no-exclude", action="store_true", help="不排除实心区域（曲线非常密集/很粗时使用）")
    ap.add_argument("--panels", action="store_true", help="多子图：自动找出所有坐标系并逐个处理（输出文件加 _a/_b 后缀）")
    ap.add_argument("--panel", help="多子图：只处理指定面板（a/b/c…，按先上后下、先左后右编号）")
    args = ap.parse_args(argv)

    import os
    from PIL import Image
    from curve_core import export_csv
    img = Image.open(args.image).convert("RGB")
    if args.crop:
        img = img.crop(parse_crop(args.crop, img.size))
    rgb = np.asarray(img)
    ticks = lambda t: [float(v) for v in t.split(",")] if t else None
    kw = dict(use_ocr=not args.no_ocr, tol=args.tol, progress=print,
              exclude_dense=not args.no_exclude,
              calibration=parse_cal_spec(args.cal) if args.cal else None,
              xticks=ticks(args.xticks), yticks=ticks(args.yticks),
              log_x=args.logx, log_y=args.logy)
    out = args.output or os.path.splitext(args.image)[0] + "_data.csv"
    delim = "\t" if out.lower().endswith((".txt", ".dat")) else ","

    def finish(res, rgb_, out_path, preview_path):
        for line in res.log:
            print(line)
        if preview_path:
            save_preview(rgb_, res, preview_path)
            print("预览图已保存:", preview_path)
        if not res.curves:
            print("没有识别到曲线，未导出。")
            return 1
        if res.calibration is None:
            cal = AxisCalibration(0, 0, 1, 1, 0, 0, 1, 1)   # 无标定：导出像素坐标（y 向下为正）
            print("警告：坐标轴未能自动标定，导出的是像素坐标。")
        else:
            cal = res.calibration
        export_csv(out_path, res.curves, cal, mode=args.mode, n_points=args.n, delimiter=delim)
        print("已导出:", out_path)
        return 0

    panels = find_panels(rgb)
    if args.panels or args.panel:
        if not panels:
            print("没有检测到坐标系面板。")
            return 1
        print(f"检测到 {len(panels)} 个坐标系面板: " + "; ".join(f"{p.label}=轴框{p.box} 裁剪{p.crop}" for p in panels))
        if args.panel:
            sel = [p for p in panels if p.label == args.panel.strip().lower()]
            if not sel:
                print(f"没有面板 {args.panel}，可选: " + ", ".join(p.label for p in panels))
                return 1
            p = sel[0]
            x0, y0, x1, y1 = p.crop
            print(f"=== 面板 {p.label}（裁剪 {p.crop}，以下像素坐标均相对裁剪后的图）===")
            return finish(auto_digitize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), frame=p.frame(), **kw),
                          rgb[y0:y1, x0:x1], out, args.preview)
        rc = 0
        stem, ext = os.path.splitext(out)
        pstem, pext = os.path.splitext(args.preview) if args.preview else (None, None)
        for p, sub, res in digitize_panels(rgb, panels, **kw):
            print(f"=== 面板 {p.label}（裁剪 {p.crop}，以下像素坐标均相对裁剪后的图）===")
            rc |= finish(res, sub, f"{stem}_{p.label}{ext}", f"{pstem}_{p.label}{pext}" if pstem else None)
        return rc
    if len(panels) > 1:
        print(f"提示：检测到 {len(panels)} 个坐标系面板（" + ", ".join(f"{p.label}={p.box}" for p in panels)
              + "），整图处理通常不可靠；建议加 --panels 逐个处理，或 --panel a 只处理某一个")
    res = auto_digitize(rgb, **kw)
    return finish(res, rgb, out, args.preview)


if __name__ == "__main__":
    raise SystemExit(main())
