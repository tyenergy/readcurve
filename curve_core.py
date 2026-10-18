# -*- coding: utf-8 -*-
"""
曲线数字化核心算法（不依赖 GUI，可单独导入/测试）。

- AxisCalibration : 像素坐标 <-> 数据坐标 换算（支持线性/对数轴）
- color_mask      : 按颜色容差生成掩码
- trace_curve     : 从种子点出发，沿列追踪同色曲线，返回像素坐标序列
- Curve           : 一条已提取曲线（像素点 + 元数据）
- resample / export_csv : 重采样与导出
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 坐标标定
# ----------------------------------------------------------------------------
@dataclass
class AxisCalibration:
    """用 X 轴上两点、Y 轴上两点做标定（假定坐标轴与图像边平行）。"""
    x1_px: Optional[float] = None
    x1_val: Optional[float] = None
    x2_px: Optional[float] = None
    x2_val: Optional[float] = None
    y1_px: Optional[float] = None
    y1_val: Optional[float] = None
    y2_px: Optional[float] = None
    y2_val: Optional[float] = None
    log_x: bool = False
    log_y: bool = False

    # -- 状态 ---------------------------------------------------------------
    def x_ready(self) -> bool:
        return None not in (self.x1_px, self.x1_val, self.x2_px, self.x2_val) \
            and self.x1_px != self.x2_px

    def y_ready(self) -> bool:
        return None not in (self.y1_px, self.y1_val, self.y2_px, self.y2_val) \
            and self.y1_px != self.y2_px

    def is_ready(self) -> bool:
        return self.x_ready() and self.y_ready()

    def validate(self) -> Optional[str]:
        """返回错误信息；None 表示合法。"""
        if not self.x_ready():
            return "X 轴标定不完整（需要两个像素位置不同的点及其数值）"
        if not self.y_ready():
            return "Y 轴标定不完整（需要两个像素位置不同的点及其数值）"
        if self.log_x and (self.x1_val <= 0 or self.x2_val <= 0):
            return "对数 X 轴的标定值必须为正数"
        if self.log_y and (self.y1_val <= 0 or self.y2_val <= 0):
            return "对数 Y 轴的标定值必须为正数"
        if self.x1_val == self.x2_val:
            return "X 轴两个标定值不能相同"
        if self.y1_val == self.y2_val:
            return "Y 轴两个标定值不能相同"
        return None

    # -- 换算 ---------------------------------------------------------------
    @staticmethod
    def _lin(p, p1, v1, p2, v2, log: bool):
        p = np.asarray(p, dtype=float)
        if log:
            lv1, lv2 = math.log10(v1), math.log10(v2)
            return 10.0 ** (lv1 + (p - p1) * (lv2 - lv1) / (p2 - p1))
        return v1 + (p - p1) * (v2 - v1) / (p2 - p1)

    @staticmethod
    def _inv(v, p1, v1, p2, v2, log: bool):
        v = np.asarray(v, dtype=float)
        if log:
            lv1, lv2 = math.log10(v1), math.log10(v2)
            return p1 + (np.log10(v) - lv1) * (p2 - p1) / (lv2 - lv1)
        return p1 + (v - v1) * (p2 - p1) / (v2 - v1)

    def px_to_x(self, px):
        return self._lin(px, self.x1_px, self.x1_val, self.x2_px, self.x2_val, self.log_x)

    def px_to_y(self, py):
        return self._lin(py, self.y1_px, self.y1_val, self.y2_px, self.y2_val, self.log_y)

    def x_to_px(self, x):
        return self._inv(x, self.x1_px, self.x1_val, self.x2_px, self.x2_val, self.log_x)

    def y_to_px(self, y):
        return self._inv(y, self.y1_px, self.y1_val, self.y2_px, self.y2_val, self.log_y)

    def plot_box(self, margin: int = 0) -> Optional[Tuple[int, int, int, int]]:
        """由四个标定点围成的像素区域 (x0, y0, x1, y1)，可向外扩 margin 像素。"""
        if not self.is_ready():
            return None
        x0, x1 = sorted((self.x1_px, self.x2_px))
        y0, y1 = sorted((self.y1_px, self.y2_px))
        return (int(math.floor(x0)) - margin, int(math.floor(y0)) - margin,
                int(math.ceil(x1)) + margin, int(math.ceil(y1)) + margin)


# ----------------------------------------------------------------------------
# 颜色掩码与曲线追踪
# ----------------------------------------------------------------------------
def pick_color(rgb: np.ndarray, x: int, y: int, radius: int = 1) -> Tuple[int, int, int]:
    """取 (x, y) 周围 (2r+1)^2 像素中与中心像素接近的颜色的平均值，抗抖动。"""
    h, w = rgb.shape[:2]
    x = min(max(int(x), 0), w - 1)
    y = min(max(int(y), 0), h - 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(float)
    center = rgb[y, x].astype(float)
    d = np.sqrt(((patch - center) ** 2).sum(axis=1))
    near = patch[d <= 40] if (d <= 40).any() else patch
    c = near.mean(axis=0)
    return tuple(int(round(v)) for v in c)


def color_mask(rgb: np.ndarray, color: Sequence[int], tol: float) -> np.ndarray:
    """RGB 欧氏距离 <= tol 的像素为 True。"""
    diff = rgb.astype(np.int32) - np.asarray(color, dtype=np.int32)[None, None, :]
    dist2 = (diff * diff).sum(axis=2)
    return dist2 <= tol * tol


def _runs_in_column(col: np.ndarray, y_offset: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[s]) + y_offset, int(idx[e]) + y_offset) for s, e in zip(starts, ends)]


def trace_curve(mask: np.ndarray,
                seed_x: int, seed_y: int,
                region: Optional[Tuple[int, int, int, int]] = None,
                max_jump: float = 20.0,
                max_gap: int = 30,
                max_run: int = 80,
                merge_gap: int = 4,
                exclude: Optional[np.ndarray] = None,
                rgb: Optional[np.ndarray] = None,
                color: Optional[Sequence[int]] = None,
                color_weight: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    从种子点所在列出发，向左右逐列追踪曲线。

    每一列里把掩码像素分成若干连续段（间隙 <= merge_gap 的段合并，使空心标记成为一段），
    用最近几列的斜率预测本列 y，选择离预测值最近的段；
    若距离超过 max_jump*(间断列数+1) 则视为本列无曲线（允许虚线/间断）。
    高度超过 max_run 的段（竖直坐标轴、文字块等）被忽略；
    明显比线条粗得多的段（曲线碰到图例框/文字时）不取中心，而取段内最接近预测值的位置。
    exclude 为 True 的像素（照片/插图区域）视为不存在。
    给出 rgb 和 color 时，候选段按 位置距离 + color_weight*颜色距离 打分，
    同色系的两条曲线（如红与橙）交叉时能按颜色分开。

    返回 (cols, rows)：按列升序的像素坐标（行坐标为段中心，亚像素）。
    """
    if exclude is not None:
        mask = mask & ~exclude
    h, w = mask.shape
    if region is None:
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    else:
        rx0, ry0, rx1, ry1 = region
        x0, x1 = max(0, min(rx0, rx1)), min(w - 1, max(rx0, rx1))
        y0, y1 = max(0, min(ry0, ry1)), min(h - 1, max(ry0, ry1))
    seed_x = int(min(max(seed_x, x0), x1))
    seed_y = int(min(max(seed_y, y0), y1))

    def runs(c: int):
        rs = _runs_in_column(mask[y0:y1 + 1, c], y0)
        if merge_gap > 0 and len(rs) > 1:
            merged = [list(rs[0])]
            for a, b in rs[1:]:
                if a - merged[-1][1] - 1 <= merge_gap:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            rs = [tuple(r) for r in merged]
        if max_run > 0:
            rs = [r for r in rs if (r[1] - r[0] + 1) <= max_run]
        return rs

    ref_color = np.asarray(color, dtype=float) if (rgb is not None and color is not None) else None

    def color_dist(c: int, a: int, b: int) -> float:
        seg = rgb[a:b + 1, c].astype(float)
        return float(np.sqrt(((seg.mean(axis=0) - ref_color) ** 2).sum()))

    def pick(rs, pred: float, allow: float, thick: float) -> Optional[Tuple[float, float]]:
        """返回 (y, 段高度)。"""
        best, best_d, best_score = None, None, None
        for a, b in rs:
            d = 0.0 if a <= pred <= b else min(abs(pred - a), abs(pred - b))
            if d > allow:
                continue
            score = d
            if ref_color is not None:
                score += color_weight * color_dist(cur_col[0], a, b)
            if best_score is None or score < best_score:
                best, best_d, best_score = (a, b), d, score
        if best is None:
            return None
        a, b = best
        height = b - a + 1
        if height > max(10.0, 3.0 * thick + 4):
            # 粗段：曲线很可能贴着别的东西，取段内最接近预测值的位置
            return min(max(pred, a), b), height
        return (a + b) / 2.0, height

    cur_col = [seed_x]
    # 种子列
    seed_pick = pick(runs(seed_x), seed_y, max(max_jump, 3.0), 3.0)
    if seed_pick is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    y_seed, thick0 = seed_pick
    thick0 = max(1.0, min(thick0, 8.0))

    pts = {seed_x: y_seed}

    for direction in (1, -1):
        prev_y = y_seed
        slope = 0.0
        gap = 0
        c = seed_x + direction
        while x0 <= c <= x1:
            cur_col[0] = c
            pred = prev_y + slope * (gap + 1) * 0.7
            res = pick(runs(c), pred, max_jump * (gap + 1), thick0)
            if res is None:
                gap += 1
                if gap > max_gap:
                    break
            else:
                y, _ = res
                slope = 0.5 * slope + 0.5 * (y - prev_y) / (gap + 1) if gap == 0 else 0.0
                pts[c] = y
                prev_y = y
                gap = 0
            c += direction

    cols = np.array(sorted(pts.keys()), dtype=float)
    rows = np.array([pts[int(c)] for c in cols], dtype=float)
    return cols, rows


# ----------------------------------------------------------------------------
# 曲线对象、重采样、导出
# ----------------------------------------------------------------------------
@dataclass
class Curve:
    name: str
    color: Tuple[int, int, int] = (255, 0, 0)
    px: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    py: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    manual: bool = False
    seed: Optional[Tuple[int, int]] = None      # 自动提取的种子点
    visible: bool = True
    meta: dict = field(default_factory=dict)    # 质量指标等附加信息（coverage/density/gaps/legend）

    def __len__(self):
        return int(self.px.size)

    def hex_color(self) -> str:
        return "#%02x%02x%02x" % tuple(int(v) for v in self.color)

    def data(self, cal: AxisCalibration, n_points: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """换算为数据坐标；n_points>0 时在 x 上均匀重采样。"""
        if self.px.size == 0:
            return np.array([]), np.array([])
        order = np.argsort(self.px, kind="stable")
        x = cal.px_to_x(self.px[order])
        y = cal.px_to_y(self.py[order])
        if n_points and n_points > 1 and x.size > 1:
            x, y = resample(x, y, n_points, log_x=cal.log_x)
        return x, y

    def add_point(self, x: float, y: float):
        self.px = np.append(self.px, float(x))
        self.py = np.append(self.py, float(y))

    def remove_nearest(self, x: float, y: float, max_dist: float) -> bool:
        if self.px.size == 0:
            return False
        d = np.hypot(self.px - x, self.py - y)
        i = int(np.argmin(d))
        if d[i] <= max_dist:
            self.px = np.delete(self.px, i)
            self.py = np.delete(self.py, i)
            return True
        return False


def _unique_xy(x: np.ndarray, y: np.ndarray):
    """按 x 排序并合并重复 x（取 y 平均）。"""
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.zeros_like(ux, dtype=float)
    cnt = np.zeros_like(ux, dtype=float)
    np.add.at(uy, inv, y)
    np.add.at(cnt, inv, 1)
    return ux, uy / cnt


def resample(x: np.ndarray, y: np.ndarray, n: int, log_x: bool = False):
    """在 [xmin, xmax] 均匀取 n 点线性插值（对数轴在 log 空间均匀）。"""
    ux, uy = _unique_xy(np.asarray(x, float), np.asarray(y, float))
    if ux.size < 2:
        return ux, uy
    if log_x and ux[0] > 0:
        gx = np.logspace(np.log10(ux[0]), np.log10(ux[-1]), n)
        gy = np.interp(np.log10(gx), np.log10(ux), uy)
    else:
        gx = np.linspace(ux[0], ux[-1], n)
        gy = np.interp(gx, ux, uy)
    return gx, gy


def _fmt(v: float) -> str:
    return "%.6g" % v


def export_csv(path: str, curves: List[Curve], cal: AxisCalibration,
               mode: str = "separate", n_points: int = 0,
               delimiter: str = ",", x_label: str = "x", y_label: str = "y"):
    """
    mode:
      "separate" - 每条曲线占两列 (name_x, name_y)，行数不等时补空
      "common"   - 所有曲线插值到同一 x 网格，输出 x, name1, name2...
                   网格点数取 n_points（<=1 时用点数最多曲线的点数）
    """
    if not curves:
        raise ValueError("没有可导出的曲线")
    err = cal.validate()
    if err:
        raise ValueError(err)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=delimiter)
        if mode == "common":
            pairs = [(c, c.data(cal)) for c in curves]
            pairs = [(c, d) for c, d in pairs if d[0].size > 1]
            if not pairs:
                raise ValueError("曲线点数不足，无法插值")
            lo = max(float(np.min(d[0])) for _, d in pairs)
            hi = min(float(np.max(d[0])) for _, d in pairs)
            if hi <= lo:
                raise ValueError("各曲线的 x 范围没有重叠，无法统一网格；请改用“各曲线独立列”模式")
            n = n_points if n_points and n_points > 1 else max(d[0].size for _, d in pairs)
            use_log = cal.log_x and lo > 0
            grid = np.logspace(np.log10(lo), np.log10(hi), n) if use_log else np.linspace(lo, hi, n)
            cols = []
            for _, (x, y) in pairs:
                ux, uy = _unique_xy(x, y)
                if use_log:
                    cols.append(np.interp(np.log10(grid), np.log10(ux), uy))
                else:
                    cols.append(np.interp(grid, ux, uy))
            w.writerow([x_label] + [c.name for c, _ in pairs])
            for i in range(n):
                w.writerow([_fmt(grid[i])] + [_fmt(col[i]) for col in cols])
        else:
            datas = [c.data(cal, n_points) for c in curves]
            header = []
            for c in curves:
                header += [f"{c.name}_{x_label}", f"{c.name}_{y_label}"]
            w.writerow(header)
            nrow = max(x.size for x, _ in datas)
            for i in range(nrow):
                row = []
                for x, y in datas:
                    if i < x.size:
                        row += [_fmt(x[i]), _fmt(y[i])]
                    else:
                        row += ["", ""]
                w.writerow(row)
