# -*- coding: utf-8 -*-
"""
全自动识别的回归测试：对 sample_chart.png 做一键识别，
检查坐标轴框、OCR 标定和曲线提取的精度。运行：python test_auto.py
"""
import math
import os
import sys

import numpy as np
from PIL import Image

import test_core
from auto_detect import auto_digitize, ocr_available

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = os.path.join(HERE, "sample_chart.png")
    if not os.path.exists(path):
        test_core.make_chart(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    ok = True

    res = auto_digitize(rgb, use_ocr=ocr_available())
    for line in res.log:
        print(line)

    # 1. 绘图区
    x0, y0, x1, y1 = res.frame.box
    print(f"frame box={res.frame.box} x_axis_row={res.frame.x_axis_row} y_axis_col={res.frame.y_axis_col}")
    if not (abs(x0 - test_core.PX0) <= 4 and abs(x1 - test_core.PX1) <= 4
            and abs(y0 - test_core.PY1) <= 4 and abs(y1 - test_core.PY0) <= 4):
        ok = False
        print("  -> FAIL: 绘图区位置不对")

    # 2. OCR 标定
    if ocr_available():
        cal = res.calibration
        if cal is None:
            ok = False
            print("  -> FAIL: 自动标定失败")
        else:
            # 用标定把真实数据点换算成像素，与已知像素位置比较
            for v, true_px in ((0, test_core.PX0), (10, test_core.PX1)):
                e = abs(float(cal.x_to_px(v)) - true_px)
                print(f"x={v}: px err {e:.2f}")
                ok &= e <= 1.5
            for v, true_px in ((-1, test_core.PY0), (1, test_core.PY1)):
                e = abs(float(cal.y_to_px(v)) - true_px)
                print(f"y={v}: px err {e:.2f}")
                ok &= e <= 1.5
    else:
        print("（未安装 OCR，跳过标定检查）")
        cal = None

    # 3. 曲线：应恰好两条，各自覆盖 >95% 的 x 范围，误差小
    fns = {(220, 30, 30): np.sin, (30, 60, 220): lambda x: np.cos(x) * np.exp(-x / 8),
           (20, 150, 40): lambda x: 0.6 * np.sin(2 * x) * np.exp(-x / 10)}
    if len(res.curves) != 3:
        ok = False
        print(f"  -> FAIL: 期望 3 条曲线，得到 {len(res.curves)}")
    names = sorted(c.name for c in res.curves)
    print("curve names:", names)
    if not (any("sin" in n for n in names) and any("cos" in n for n in names) and any("dot" in n for n in names)):
        ok = False
        print("  -> FAIL: 图例命名不对")
    ref = cal if cal is not None else test_core.AxisCalibration(
        x1_px=test_core.PX0, x1_val=0, x2_px=test_core.PX1, x2_val=10,
        y1_px=test_core.PY0, y1_val=-1, y2_px=test_core.PY1, y2_val=1)
    for cv in res.curves:
        x, y = cv.data(ref)
        key = min(fns, key=lambda k: np.linalg.norm(np.array(k) - cv.color))
        err = np.abs(y - fns[key](x))
        cover = (x.max() - x.min()) / 10
        print(f"{cv.name} color={cv.color} n={len(cv)} coverage={cover:.1%} max_err={err.max():.4f} mean={err.mean():.4f}")
        dotted = cv.color[1] > 100 and cv.color[0] < 60
        if cover < (0.9 if dotted else 0.95) or err.max() > (0.06 if dotted else 0.04) or err.mean() > 0.012:
            ok = False
            print("  -> FAIL")

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
