# -*- coding: utf-8 -*-
"""
生成一张合成测试图 (sample_chart.png)，用已知的坐标轴位置做标定，
自动提取两条曲线并与真实函数比较误差。运行：python test_core.py
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curve_core import (AxisCalibration, Curve, color_mask, export_csv,
                        pick_color, trace_curve)

HERE = os.path.dirname(os.path.abspath(__file__))

# 绘图区像素范围与数据范围
PX0, PX1 = 80, 760          # x 像素: 0 .. 10
PY0, PY1 = 540, 60          # y 像素: -1 .. 1  (注意 y 像素向下递增)
XMIN, XMAX = 0.0, 10.0
YMIN, YMAX = -1.0, 1.0


def to_px(x, y):
    px = PX0 + (x - XMIN) / (XMAX - XMIN) * (PX1 - PX0)
    py = PY0 + (y - YMIN) / (YMAX - YMIN) * (PY1 - PY0)
    return px, py


def make_chart(path):
    img = Image.new("RGB", (820, 600), "white")
    d = ImageDraw.Draw(img)
    # 网格
    for i in range(11):
        x, _ = to_px(i, 0)
        d.line([(x, PY1), (x, PY0)], fill=(220, 220, 220), width=1)
    for j in range(-10, 11, 2):
        _, y = to_px(0, j / 10)
        d.line([(PX0, y), (PX1, y)], fill=(220, 220, 220), width=1)
    # 坐标轴 + 刻度 + 刻度数字
    d.line([(PX0, PY1), (PX0, PY0), (PX1, PY0)], fill="black", width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i in range(11):
        x, _ = to_px(i, 0)
        d.line([(x, PY0), (x, PY0 + 6)], fill="black", width=2)
        d.text((x, PY0 + 10), str(i), fill="black", font=font, anchor="mt")
    for j in range(-10, 11, 5):
        _, y = to_px(0, j / 10)
        d.line([(PX0 - 6, y), (PX0, y)], fill="black", width=2)
        d.text((PX0 - 10, y), "%g" % (j / 10), fill="black", font=font, anchor="rm")
    d.text(((PX0 + PX1) / 2, PY0 + 34), "Time (s)", fill="black", font=font, anchor="mt")
    # 曲线 1：红色实线 sin(x)
    pts = [to_px(x, math.sin(x)) for x in np.linspace(XMIN, XMAX, 2000)]
    d.line(pts, fill=(220, 30, 30), width=3, joint="curve")
    # 曲线 2：蓝色虚线 cos(x)*exp(-x/8)
    xs = np.linspace(XMIN, XMAX, 2000)
    seg = []
    for k, x in enumerate(xs):
        if (k // 30) % 2 == 0:
            seg.append(to_px(x, math.cos(x) * math.exp(-x / 8)))
        elif seg:
            d.line(seg, fill=(30, 60, 220), width=3)
            seg = []
    if seg:
        d.line(seg, fill=(30, 60, 220), width=3)
    # 曲线 3：绿色点线 0.6*sin(2x)*exp(-x/10)（每 7 像素一个 2 像素的点）
    xs3 = np.linspace(XMIN, XMAX, 2000)
    last = None
    for x in xs3:
        px, py = to_px(x, 0.6 * math.sin(2 * x) * math.exp(-x / 10))
        if last is None or math.hypot(px - last[0], py - last[1]) >= 7:
            d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(20, 150, 40))
            last = (px, py)
    # 图例（与曲线同色的文字 / 短线，用于检验追踪不被干扰）
    d.line([(600, 90), (640, 90)], fill=(220, 30, 30), width=3)
    d.text((648, 83), "sin(x)", fill=(220, 30, 30), font=font)
    d.line([(600, 110), (640, 110)], fill=(30, 60, 220), width=3)
    d.text((648, 103), "cosine", fill=(30, 60, 220), font=font)
    for xx in range(600, 641, 7):
        d.ellipse([xx - 1, 129, xx + 1, 131], fill=(20, 150, 40))
    d.text((648, 123), "dotted", fill=(20, 150, 40), font=font)
    img.save(path)
    return img


def main():
    path = os.path.join(HERE, "sample_chart.png")
    img = make_chart(path)
    rgb = np.asarray(img.convert("RGB"))

    cal = AxisCalibration(x1_px=PX0, x1_val=XMIN, x2_px=PX1, x2_val=XMAX,
                          y1_px=PY0, y1_val=YMIN, y2_px=PY1, y2_val=YMAX)
    assert cal.validate() is None
    region = cal.plot_box(margin=3)

    ok = True
    curves = []
    cases = [
        ("sin", lambda x: np.sin(x), to_px(2.0, math.sin(2.0)), 0.03),
        ("cos_decay", lambda x: np.cos(x) * np.exp(-x / 8), to_px(1.0, math.cos(1.0) * math.exp(-1 / 8)), 0.03),
    ]
    for name, fn, seed, tol_err in cases:
        sx, sy = int(round(seed[0])), int(round(seed[1]))
        color = pick_color(rgb, sx, sy)
        mask = color_mask(rgb, color, 60)
        cols, rows = trace_curve(mask, sx, sy, region=region, max_jump=20, max_gap=40, max_run=80)
        cv = Curve(name=name, color=color, px=cols, py=rows, seed=(sx, sy))
        curves.append(cv)
        x, y = cv.data(cal)
        err = np.abs(y - fn(x))
        cover = (x.max() - x.min()) / (XMAX - XMIN)
        print(f"[{name}] color={color} points={len(cv)} x:[{x.min():.3f},{x.max():.3f}] "
              f"coverage={cover:.1%} max_err={err.max():.4f} mean_err={err.mean():.4f}")
        if err.max() > tol_err or cover < 0.95:
            ok = False
            print("  -> FAIL")

    out1 = os.path.join(HERE, "sample_out_separate.csv")
    out2 = os.path.join(HERE, "sample_out_common.csv")
    export_csv(out1, curves, cal, mode="separate")
    export_csv(out2, curves, cal, mode="common", n_points=101)
    with open(out2, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    print("common grid csv rows:", len(lines) - 1, "header:", lines[0])
    assert len(lines) == 102

    # 对数轴换算自检
    lc = AxisCalibration(x1_px=0, x1_val=1, x2_px=100, x2_val=1000, log_x=True,
                         y1_px=0, y1_val=10, y2_px=100, y2_val=0.1, log_y=True)
    assert abs(lc.px_to_x(50) - math.sqrt(1000)) < 1e-9
    assert abs(lc.px_to_y(50) - 1.0) < 1e-9
    assert abs(lc.x_to_px(100) - 66.6666667) < 1e-5

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
