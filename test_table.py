# -*- coding: utf-8 -*-
"""
表格识别回归测试：合成一张三线表（含 ±、×10^n、科学计数法、负数）和一张中文网格表，
检查行列结构、单元格文本与数值解析。运行：python test_table.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from table_ocr import parse_cell, recognize_table

HERE = os.path.dirname(os.path.abspath(__file__))


def make_threeline(path):
    font = ImageFont.truetype("arial.ttf", 18)
    bold = ImageFont.truetype("arialbd.ttf", 18)
    rows = [["Sample", "Thickness (nm)", "Resistivity (ohm cm)", "Mobility", "Yield (%)"],
            ["A-1", "120 ± 5", "3.2×10^-3", "45.6 ± 1.2", "92.5"],
            ["A-2", "135 ± 4", "2.8×10^-3", "48.1 ± 0.9", "93.1"],
            ["B-1", "98 ± 6", "5.1×10^-3", "-12.4 ± 2.0", "87.0"],
            ["B-2", "101 ± 3", "4.9e-3", "31.0 ± 1.5", "88.4"]]
    W, H = 760, 230
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    colx = [30, 150, 330, 540, 680]
    y = 25
    d.line([(20, y - 8), (W - 20, y - 8)], fill="black", width=2)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            d.text((colx[c], y), cell, fill="black", font=bold if r == 0 else font)
        y += 38
        if r == 0:
            d.line([(20, y - 8), (W - 20, y - 8)], fill="black", width=1)
    d.line([(20, y - 8), (W - 20, y - 8)], fill="black", width=2)
    img.save(path)
    return rows


def make_grid_cn(path):
    font = None
    for f in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            font = ImageFont.truetype(f, 20)
            break
        except Exception:
            pass
    if font is None:
        return None
    rows = [["参数", "数值", "备注"], ["中心频率", "5.52", "实测"], ["带宽", "120", "仿真"],
            ["增益", "6.8", "仿真"], ["效率", "85.3", "实测"]]
    W, H = 520, 230
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xs = [20, 200, 340, 500]
    ys = [20, 60, 100, 140, 180, 220]
    for x in xs:
        d.line([(x, ys[0]), (x, ys[-1])], fill=(80, 80, 80), width=1)
    for yy in ys:
        d.line([(xs[0], yy), (xs[-1], yy)], fill=(80, 80, 80), width=1)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            d.text((xs[c] + 12, ys[r] + 9), cell, fill="black", font=font)
    img.save(path)
    return rows


def norm(s):
    return s.replace(" ", "").replace("±", "+-").lower()


def main():
    ok = True
    # ---- 三线表 ----
    p1 = os.path.join(HERE, "sample_table.png")
    truth = make_threeline(p1)
    tb = recognize_table(np.asarray(Image.open(p1).convert("RGB")))
    print("\n".join(tb.log))
    if len(tb.cells) != len(truth) or any(len(r) != 5 for r in tb.cells):
        ok = False
        print(f"  -> FAIL: 结构 {len(tb.cells)}x{[len(r) for r in tb.cells]}，期望 5x5")
    else:
        bad = 0
        for r in range(1, 5):
            for c in range(5):
                tv = parse_cell(truth[r][c])
                gv = parse_cell(tb.cells[r][c])
                if c == 0:
                    good = norm(tb.cells[r][c]) == norm(truth[r][c])
                else:
                    good = gv[0] is not None and abs(gv[0] - tv[0]) <= 1e-9 * max(1, abs(tv[0])) \
                        and ((gv[1] is None) == (tv[1] is None)) and (gv[1] is None or abs(gv[1] - tv[1]) < 1e-9)
                if not good:
                    bad += 1
                    print(f"  cell[{r}][{c}] got {tb.cells[r][c]!r} want {truth[r][c]!r}")
        print(f"三线表：数据单元格 {20 - bad}/20 正确")
        ok &= bad == 0
    # ---- 中文网格表 ----
    p2 = os.path.join(HERE, "sample_table_cn.png")
    truth = make_grid_cn(p2)
    if truth is None:
        print("（没有中文字体，跳过网格表测试）")
    else:
        tb = recognize_table(np.asarray(Image.open(p2).convert("RGB")))
        print("\n".join(tb.log))
        if len(tb.cells) != 5 or any(len(r) != 3 for r in tb.cells):
            ok = False
            print(f"  -> FAIL: 结构 {len(tb.cells)}x{[len(r) for r in tb.cells]}，期望 5x3")
        else:
            bad = sum(1 for r in range(5) for c in range(3) if norm(tb.cells[r][c]) != norm(truth[r][c]))
            for r in range(5):
                for c in range(3):
                    if norm(tb.cells[r][c]) != norm(truth[r][c]):
                        print(f"  cell[{r}][{c}] got {tb.cells[r][c]!r} want {truth[r][c]!r}")
            print(f"中文网格表：单元格 {15 - bad}/15 正确")
            ok &= bad <= 1          # 允许一个 OCR 误读
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
