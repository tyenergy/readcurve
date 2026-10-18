# -*- coding: utf-8 -*-
"""
ReadCurve —— 图表曲线数字化工具（图形界面）

用法：python readcurve.py [图片文件]

工作流程：
  1. 打开图片
  2. 标定坐标轴：依次点击 X1、X2、Y1、Y2 四个按钮，再在图上点对应刻度位置并输入其数值
  3. 提取曲线：
     - 自动：点“点击曲线自动提取”，再在图中点一下曲线本身（按颜色追踪）
     - 手动：进入手动取点模式后逐点左击，右击删除最近的点
  4. 导出 CSV / TXT
"""
from __future__ import annotations

import json
import math
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

from curve_core import (AxisCalibration, Curve, color_mask, export_csv,
                        pick_color, trace_curve)

APP_TITLE = "ReadCurve - 图表曲线数字化工具"
IMG_TYPES = [("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"), ("所有文件", "*.*")]

HELP_TEXT = """使用说明

一、打开图片
    文件 → 打开图片，或把图片路径作为命令行参数。
    鼠标滚轮缩放，按住右键（或中键）拖动平移，“适应窗口”复位。

○ 一键自动：点顶部“★ 一键自动”，程序自动找坐标轴框、OCR 读刻度数字完成标定、
    并按颜色找出全部曲线。之后只需核对（移动鼠标看状态栏坐标是否正确）并导出。
    识别不准时，按下面的步骤手动修正即可。

二、标定坐标轴（“标定”页）
    依次点击 “X1” “X2” “Y1” “Y2” 按钮，然后在图上点击对应的刻度位置，
    在弹窗中输入该刻度的真实数值。X1/X2 取 X 轴上相距尽量远的两个刻度，
    Y1/Y2 取 Y 轴上两个刻度。对数坐标请勾选“对数 X/Y 轴”。
    标定完成后，状态栏会实时显示鼠标所在位置的数据坐标，可用来核对。
    （本工具假定坐标轴与图像边平行；若图片歪斜请先用其他工具旋转校正。）

三、提取曲线（“提取”页）
    自动提取：点击“点击曲线自动提取”，再在图中点一下目标曲线。程序以该点
        颜色为基准，在容差范围内逐列追踪曲线，支持虚线/间断。
        若提取结果不理想，调节“颜色容差 / 最大跳变 / 最大间断”后点“重新提取所选曲线”。
        曲线相互交叉时颜色不同即可区分；同色曲线建议用手动模式。
    提取区域：默认限制在标定框内（排除图例/标题）。也可“框选区域”手动限定。
    插图/照片：程序自动找出图中实心的大块区域（照片、模型图、填充色块），以绿色淡显，
        自动提取和点击提取都不会进入这些区域；曲线穿过插图的部分按间断处理。
        若误判，可在“提取”页取消勾选。
    手动取点：进入手动模式后，左键逐点点击；右键单击删除最近的点；Ctrl+Z 撤销。

四、导出（“曲线/导出”页）
    各曲线独立列：每条曲线输出 名称_x, 名称_y 两列。
    统一 X 网格：所有曲线插值到同一组 x 上，便于对比。
    文件后缀 .csv 用逗号分隔，.txt/.dat 用制表符分隔。
    “导出点数”为 0 时输出全部采样点（每像素列一个点）。

五、工程文件
    文件 → 保存工程 可把标定、区域、所有曲线保存为 .json，下次继续。
"""


class ReadCurveApp(tk.Tk):
    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(900, 600)

        self.image: Image.Image | None = None
        self.image_path: str | None = None
        self.rgb: np.ndarray | None = None
        self.cal = AxisCalibration()
        self.curves: list[Curve] = []
        self.cur_index: int | None = None
        self.region: tuple | None = None          # 用户框选区域（像素，含端点）
        self.exclude: np.ndarray | None = None     # 插图/照片等实心区域掩码（True=排除）
        self.mode: str | None = None              # cal_x1/cal_x2/cal_y1/cal_y2/region/auto/manual
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._photo = None
        self._drag = None
        self._rubber = None
        self._undo_stack: list = []

        self._build_menu()
        self._build_ui()
        self._bind_events()
        self.set_mode(None)

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="打开图片...  (Ctrl+O)", command=self.open_image)
        fm.add_separator()
        fm.add_command(label="打开工程...", command=self.load_project)
        fm.add_command(label="保存工程...  (Ctrl+S)", command=self.save_project)
        fm.add_separator()
        fm.add_command(label="导出所选曲线...", command=lambda: self.export(selected_only=True))
        fm.add_command(label="导出全部曲线...  (Ctrl+E)", command=lambda: self.export(selected_only=False))
        fm.add_separator()
        fm.add_command(label="退出", command=self.destroy)
        m.add_cascade(label="文件", menu=fm)

        vm = tk.Menu(m, tearoff=0)
        vm.add_command(label="适应窗口  (F)", command=self.fit_to_window)
        vm.add_command(label="实际大小 100%  (1)", command=lambda: self.set_zoom(1.0))
        m.add_cascade(label="视图", menu=vm)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="使用说明", command=self.show_help)
        m.add_cascade(label="帮助", menu=hm)
        self.config(menu=m)

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # 左侧控制面板
        left = ttk.Frame(root, width=330)
        left.pack(side="left", fill="y", padx=(6, 0), pady=6)
        left.pack_propagate(False)

        top = ttk.Frame(left)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="打开图片", command=self.open_image).pack(side="left", fill="x", expand=True)
        ttk.Button(top, text="★ 一键自动", command=lambda: self.run_auto(True, True)).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(top, text="适应窗口", command=self.fit_to_window, width=8).pack(side="left")
        ttk.Button(top, text="说明", command=self.show_help, width=5).pack(side="left", padx=(4, 0))

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        self._build_tab_calib()
        self._build_tab_extract()
        self._build_tab_curves()

        # 右侧画布
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(right, bg="#404040", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # 状态栏
        self.status = ttk.Label(self, anchor="w", relief="sunken", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")

    # -- 标定页 ---------------------------------------------------------------
    def _build_tab_calib(self):
        f = ttk.Frame(self.nb, padding=8)
        self.nb.add(f, text=" ① 标定 ")
        ttk.Button(f, text="自动识别坐标轴（OCR 读刻度）", command=lambda: self.run_auto(True, False)
                   ).grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 6))
        ttk.Label(f, text="或手动标定：点击按钮后，在图上点对应刻度\n位置，再输入该刻度的真实数值。",
                  foreground="#555").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.cal_vars = {}
        self.cal_px_labels = {}
        rows = [("x1", "X1"), ("x2", "X2"), ("y1", "Y1"), ("y2", "Y2")]
        for i, (key, label) in enumerate(rows, start=2):
            b = ttk.Button(f, text=f"点 {label}", width=8, command=lambda k=key: self.set_mode("cal_" + k))
            b.grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            e = ttk.Entry(f, textvariable=var, width=12)
            e.grid(row=i, column=1, sticky="w", padx=4)
            e.bind("<Return>", lambda _e: self.sync_cal(show_error=True))
            e.bind("<FocusOut>", lambda _e: self.sync_cal(show_error=False))
            self.cal_vars[key] = var
            lab = ttk.Label(f, text="px: —", foreground="#777")
            lab.grid(row=i, column=2, sticky="w")
            self.cal_px_labels[key] = lab

        self.log_x_var = tk.BooleanVar(value=False)
        self.log_y_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="对数 X 轴", variable=self.log_x_var,
                        command=lambda: self.sync_cal(False)).grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(f, text="对数 Y 轴", variable=self.log_y_var,
                        command=lambda: self.sync_cal(False)).grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Button(f, text="清除标定", command=self.clear_calibration).grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.cal_status = ttk.Label(f, text="状态：未标定", foreground="#b00")
        self.cal_status.grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Separator(f).grid(row=10, column=0, columnspan=3, sticky="we", pady=10)
        ttk.Label(f, text="提示：标定完成后移动鼠标，状态栏会显示\n对应的数据坐标，可用于核对标定是否正确。",
                  foreground="#555").grid(row=11, column=0, columnspan=3, sticky="w")

    # -- 提取页 ---------------------------------------------------------------
    def _build_tab_extract(self):
        f = ttk.Frame(self.nb, padding=8)
        self.nb.add(f, text=" ② 提取 ")

        lf = ttk.LabelFrame(f, text="提取区域", padding=6)
        lf.pack(fill="x")
        self.limit_box_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="限制在标定框内（排除图例/标题）", variable=self.limit_box_var,
                        command=self.redraw).pack(anchor="w")
        self.exclude_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="自动排除插图/照片等实心区域（绿色淡显）", variable=self.exclude_var,
                        command=self._on_toggle_exclude).pack(anchor="w")
        rf = ttk.Frame(lf)
        rf.pack(fill="x", pady=(4, 0))
        ttk.Button(rf, text="框选区域", command=lambda: self.set_mode("region")).pack(side="left", fill="x", expand=True)
        ttk.Button(rf, text="清除框选", command=self.clear_region).pack(side="left", fill="x", expand=True, padx=(4, 0))

        af = ttk.LabelFrame(f, text="自动提取（按颜色追踪）", padding=6)
        af.pack(fill="x", pady=(8, 0))
        ttk.Button(af, text="自动识别图中所有曲线", command=lambda: self.run_auto(False, True)).pack(fill="x")
        ttk.Button(af, text="点击某条曲线提取", command=lambda: self.set_mode("auto")).pack(fill="x", pady=(4, 0))

        g = ttk.Frame(af)
        g.pack(fill="x", pady=(6, 0))
        ttk.Label(g, text="颜色容差").grid(row=0, column=0, sticky="w")
        self.tol_var = tk.IntVar(value=50)
        ttk.Scale(g, from_=5, to=200, variable=self.tol_var, orient="horizontal",
                  command=lambda v: self.tol_var.set(int(float(v)))).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(g, textvariable=self.tol_var, width=4).grid(row=0, column=2)

        self.jump_var = tk.IntVar(value=20)
        self.gap_var = tk.IntVar(value=30)
        self.run_var = tk.IntVar(value=80)
        for r, (text, var, tip) in enumerate([
            ("最大跳变(px)", self.jump_var, "相邻列允许的最大 y 变化"),
            ("最大间断(px)", self.gap_var, "虚线/缺口允许的最大列数"),
            ("忽略色块高于(px)", self.run_var, "过滤竖轴/文字等大块"),
        ], start=1):
            ttk.Label(g, text=text).grid(row=r, column=0, sticky="w", pady=1)
            ttk.Spinbox(g, from_=1, to=2000, textvariable=var, width=7).grid(row=r, column=1, sticky="w", padx=4)
            ttk.Label(g, text=tip, foreground="#777").grid(row=r, column=2, columnspan=2, sticky="w")
        g.columnconfigure(1, weight=1)

        cf = ttk.Frame(af)
        cf.pack(fill="x", pady=(6, 0))
        ttk.Label(cf, text="所选曲线颜色：").pack(side="left")
        self.swatch = tk.Label(cf, width=4, relief="solid", bg="#ffffff")
        self.swatch.pack(side="left")
        ttk.Button(af, text="用当前参数重新提取所选曲线", command=self.re_extract).pack(fill="x", pady=(6, 0))

        mf = ttk.LabelFrame(f, text="手动取点", padding=6)
        mf.pack(fill="x", pady=(8, 0))
        ttk.Button(mf, text="进入手动取点模式（新建曲线）", command=self.start_manual).pack(fill="x")
        ttk.Label(mf, text="左键添加点，右键单击删除最近点，\nCtrl+Z 撤销，Esc 退出模式。",
                  foreground="#555").pack(anchor="w", pady=(4, 0))

        ttk.Button(f, text="退出当前模式 (Esc)", command=lambda: self.set_mode(None)).pack(fill="x", pady=(10, 0))

    # -- 曲线/导出页 ----------------------------------------------------------
    def _build_tab_curves(self):
        f = ttk.Frame(self.nb, padding=8)
        self.nb.add(f, text=" ③ 曲线/导出 ")

        ttk.Label(f, text="曲线列表").pack(anchor="w")
        lb_frame = ttk.Frame(f)
        lb_frame.pack(fill="x")
        self.listbox = tk.Listbox(lb_frame, height=6, exportselection=False, activestyle="none")
        self.listbox.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(lb_frame, command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_curve)
        self.listbox.bind("<Double-Button-1>", lambda e: self.rename_curve())

        bf = ttk.Frame(f)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="重命名", command=self.rename_curve).pack(side="left", fill="x", expand=True)
        ttk.Button(bf, text="显示/隐藏", command=self.toggle_visible).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(bf, text="删除", command=self.delete_curve).pack(side="left", fill="x", expand=True)

        ttk.Label(f, text="数据预览（所选曲线，前 2000 行）").pack(anchor="w", pady=(6, 0))
        tv_frame = ttk.Frame(f)
        tv_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tv_frame, columns=("i", "x", "y"), show="headings", height=10)
        for col, text, w in (("i", "#", 50), ("x", "x", 110), ("y", "y", 110)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="e", stretch=(col != "i"))
        self.tree.pack(side="left", fill="both", expand=True)
        tsb = ttk.Scrollbar(tv_frame, command=self.tree.yview)
        tsb.pack(side="left", fill="y")
        self.tree.config(yscrollcommand=tsb.set)

        ef = ttk.LabelFrame(f, text="导出", padding=6)
        ef.pack(fill="x", pady=(8, 0))
        self.export_mode = tk.StringVar(value="separate")
        ttk.Radiobutton(ef, text="各曲线独立列 (name_x, name_y)", variable=self.export_mode,
                        value="separate").pack(anchor="w")
        ttk.Radiobutton(ef, text="统一 X 网格（插值到同一组 x）", variable=self.export_mode,
                        value="common").pack(anchor="w")
        nf = ttk.Frame(ef)
        nf.pack(fill="x", pady=(4, 0))
        ttk.Label(nf, text="导出点数 (0=全部采样点)").pack(side="left")
        self.npts_var = tk.IntVar(value=0)
        ttk.Spinbox(nf, from_=0, to=100000, textvariable=self.npts_var, width=8).pack(side="left", padx=4)
        bf2 = ttk.Frame(ef)
        bf2.pack(fill="x", pady=(6, 0))
        ttk.Button(bf2, text="导出所选曲线", command=lambda: self.export(True)).pack(side="left", fill="x", expand=True)
        ttk.Button(bf2, text="导出全部曲线", command=lambda: self.export(False)).pack(side="left", fill="x", expand=True, padx=(4, 0))

    # ------------------------------------------------------------------ events
    def _bind_events(self):
        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<Motion>", self._on_motion)
        c.bind("<ButtonPress-1>", self._on_left_down)
        c.bind("<B1-Motion>", self._on_left_drag)
        c.bind("<ButtonRelease-1>", self._on_left_up)
        for btn in (2, 3):
            c.bind(f"<ButtonPress-{btn}>", self._on_pan_start)
            c.bind(f"<B{btn}-Motion>", self._on_pan_move)
            c.bind(f"<ButtonRelease-{btn}>", self._on_pan_end)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        c.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))
        self.bind("<Escape>", lambda e: self.set_mode(None))
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_project())
        self.bind("<Control-e>", lambda e: self.export(False))
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Key-f>", lambda e: self._key_shortcut(e, self.fit_to_window))
        self.bind("<Key-1>", lambda e: self._key_shortcut(e, lambda: self.set_zoom(1.0)))
        self.bind("<Delete>", lambda e: self._key_shortcut(e, self.delete_curve))

    def _key_shortcut(self, event, fn):
        # 输入框获得焦点时不触发单键快捷键
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text)):
            return
        fn()

    # 坐标换算：画布 <-> 图像像素（像素中心坐标，像素 i 的中心为 i）
    def canvas_to_img(self, cx, cy):
        return (cx - self.ox) / self.zoom - 0.5, (cy - self.oy) / self.zoom - 0.5

    def img_to_canvas(self, ix, iy):
        return (ix + 0.5) * self.zoom + self.ox, (iy + 0.5) * self.zoom + self.oy

    def _in_image(self, ix, iy):
        if self.image is None:
            return False
        w, h = self.image.size
        return -0.5 <= ix < w - 0.5 and -0.5 <= iy < h - 0.5

    def _on_motion(self, e):
        if self.image is None:
            return
        ix, iy = self.canvas_to_img(e.x, e.y)
        if not self._in_image(ix, iy):
            self._set_status(None)
            return
        px, py = int(round(ix)), int(round(iy))
        txt = f"像素: ({px}, {py})"
        if self.cal.is_ready():
            try:
                x = float(self.cal.px_to_x(ix))
                y = float(self.cal.px_to_y(iy))
                txt += f"    数据: X = {x:.6g},  Y = {y:.6g}"
            except Exception:
                pass
        r, g, b = self.rgb[py, px]
        txt += f"    颜色: ({r},{g},{b})"
        self._set_status(txt)

    def _on_left_down(self, e):
        if self.image is None:
            return
        ix, iy = self.canvas_to_img(e.x, e.y)
        if self.mode == "region":
            self._drag = (ix, iy)
            return
        if not self._in_image(ix, iy):
            return
        if self.mode and self.mode.startswith("cal_"):
            self._set_cal_point(self.mode[4:], ix, iy)
        elif self.mode == "auto":
            self.auto_extract(int(round(ix)), int(round(iy)))
        elif self.mode == "manual":
            self.add_manual_point(ix, iy)

    def _on_left_drag(self, e):
        if self.mode == "region" and self._drag:
            ix, iy = self.canvas_to_img(e.x, e.y)
            self._draw_rubber(self._drag, (ix, iy))

    def _on_left_up(self, e):
        if self.mode == "region" and self._drag:
            ix0, iy0 = self._drag
            ix1, iy1 = self.canvas_to_img(e.x, e.y)
            self._drag = None
            w, h = self.image.size
            x0, x1 = sorted((int(round(ix0)), int(round(ix1))))
            y0, y1 = sorted((int(round(iy0)), int(round(iy1))))
            x0, x1 = max(0, x0), min(w - 1, x1)
            y0, y1 = max(0, y0), min(h - 1, y1)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                self.region = (x0, y0, x1, y1)
            self.set_mode(None)
            self.redraw()

    def _on_pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy, False)

    def _on_pan_move(self, e):
        if not getattr(self, "_pan", None):
            return
        sx, sy, ox, oy, _ = self._pan
        if abs(e.x - sx) + abs(e.y - sy) > 2:
            self._pan = (sx, sy, ox, oy, True)
            self.ox = ox + (e.x - sx)
            self.oy = oy + (e.y - sy)
            self.redraw()

    def _on_pan_end(self, e):
        pan = getattr(self, "_pan", None)
        self._pan = None
        if pan and not pan[4] and e.num == 3 and self.mode == "manual":
            # 右键单击：删除最近的手动点
            ix, iy = self.canvas_to_img(e.x, e.y)
            cv = self.current_curve()
            if cv is not None and cv.manual:
                self._undo_stack.append((cv, cv.px.copy(), cv.py.copy()))
                if cv.remove_nearest(ix, iy, 12 / self.zoom):
                    self.refresh_curve_list()
                    self.redraw()
                else:
                    self._undo_stack.pop()

    def _on_wheel(self, e):
        self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def _zoom_at(self, cx, cy, factor):
        if self.image is None:
            return
        new = min(60.0, max(0.02, self.zoom * factor))
        factor = new / self.zoom
        self.ox = cx - (cx - self.ox) * factor
        self.oy = cy - (cy - self.oy) * factor
        self.zoom = new
        self.redraw()

    def set_zoom(self, z):
        if self.image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._zoom_at(cw / 2, ch / 2, z / self.zoom)

    def fit_to_window(self):
        if self.image is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        w, h = self.image.size
        self.zoom = min(cw / w, ch / h) * 0.97
        self.ox = (cw - w * self.zoom) / 2
        self.oy = (ch - h * self.zoom) / 2
        self.redraw()

    # ------------------------------------------------------------------ mode / status
    MODE_HINTS = {
        None: "就绪。滚轮缩放，右键拖动平移。",
        "cal_x1": "请在图上点击 X 轴第 1 个刻度的位置（X1）",
        "cal_x2": "请在图上点击 X 轴第 2 个刻度的位置（X2）",
        "cal_y1": "请在图上点击 Y 轴第 1 个刻度的位置（Y1）",
        "cal_y2": "请在图上点击 Y 轴第 2 个刻度的位置（Y2）",
        "region": "请按住左键拖动，框选曲线提取区域",
        "auto": "自动提取模式：在图中点击曲线本身（可连续点击多条曲线），Esc 退出",
        "manual": "手动取点模式：左键添加点，右键删除最近点，Ctrl+Z 撤销，Esc 退出",
    }

    def set_mode(self, mode):
        self.mode = mode
        self.canvas.config(cursor="crosshair" if mode else "arrow")
        self._set_status(None)

    def _set_status(self, coord_text):
        hint = self.MODE_HINTS.get(self.mode, "")
        z = f"缩放 {self.zoom * 100:.0f}%"
        parts = [f"[{hint}]", z]
        if coord_text:
            parts.append(coord_text)
        self.status.config(text="    ".join(parts))

    # ------------------------------------------------------------------ image
    def open_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="打开图片", filetypes=IMG_TYPES)
            if not path:
                return
        try:
            img = Image.open(path)
            img.load()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # 透明背景铺白，避免透明像素被当成黑色
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", img.size, "white")
                bg.paste(rgba, mask=rgba.split()[3])
                img = bg
            img = img.convert("RGB")
        except Exception as ex:
            messagebox.showerror("打开失败", f"无法打开图片：\n{ex}")
            return
        self.image = img
        self.image_path = path
        self.rgb = np.asarray(img)
        self.exclude = None
        self._disp_cache = None
        self._compute_exclude()
        self.curves.clear()
        self.cur_index = None
        self.region = None
        self._undo_stack.clear()
        self.clear_calibration(redraw=False)
        self.title(f"{APP_TITLE} - {os.path.basename(path)}")
        self.refresh_curve_list()
        self.update_idletasks()
        self.fit_to_window()
        self.nb.select(0)

    # ------------------------------------------------------------------ calibration
    def _set_cal_point(self, key, ix, iy):
        px = ix if key.startswith("x") else iy
        label = key.upper()
        cur = self._parse_float(self.cal_vars[key].get())
        val = simpledialog.askfloat("输入刻度值", f"{label} 处的{'X' if key[0]=='x' else 'Y'}坐标真实数值：",
                                    initialvalue=cur if cur is not None else 0.0, parent=self)
        if val is None:
            self.set_mode(None)
            return
        setattr(self.cal, f"{key}_px", float(px))
        self.cal_vars[key].set("%g" % val)
        self.cal_px_labels[key].config(text=f"px: {px:.1f}")
        self.set_mode(None)
        self.sync_cal(show_error=False)

    @staticmethod
    def _parse_float(s):
        try:
            return float(str(s).strip())
        except Exception:
            return None

    def sync_cal(self, show_error=False):
        """把输入框里的数值同步到 self.cal，并刷新状态。"""
        for key in ("x1", "x2", "y1", "y2"):
            setattr(self.cal, f"{key}_val", self._parse_float(self.cal_vars[key].get()))
        self.cal.log_x = bool(self.log_x_var.get())
        self.cal.log_y = bool(self.log_y_var.get())
        err = self.cal.validate()
        if err is None:
            self.cal_status.config(text="状态：标定完成 ✓", foreground="#080")
        else:
            self.cal_status.config(text="状态：" + err, foreground="#b00", wraplength=300)
            if show_error:
                messagebox.showwarning("标定", err)
        self.redraw()
        self.refresh_preview()
        return err

    def apply_calibration(self, cal: AxisCalibration):
        """把一个 AxisCalibration 写入界面（输入框、像素标签、对数勾选）。"""
        self.cal = cal
        self.log_x_var.set(bool(cal.log_x))
        self.log_y_var.set(bool(cal.log_y))
        for key in ("x1", "x2", "y1", "y2"):
            v = getattr(cal, f"{key}_val")
            px = getattr(cal, f"{key}_px")
            self.cal_vars[key].set("" if v is None else "%g" % v)
            self.cal_px_labels[key].config(text="px: —" if px is None else f"px: {px:.1f}")
        self.sync_cal(False)

    def run_auto(self, do_axes=True, do_curves=True):
        """一键自动：检测坐标轴框 → OCR 标定 → 自动找所有曲线。"""
        if self.rgb is None:
            messagebox.showinfo("自动识别", "请先打开图片。")
            return
        import auto_detect as ad
        self.config(cursor="watch")
        self._set_status("正在自动识别，请稍候…")
        self.update_idletasks()
        log = []
        labels = []
        try:
            frame = ad.find_plot_frame(self.rgb)
            b = frame.box
            log.append(f"绘图区：x {b[0]}~{b[2]}，y {b[1]}~{b[3]}"
                       + ("" if frame.x_axis_row is not None else "（未找到 X 轴线）")
                       + ("" if frame.y_axis_col is not None else "（未找到 Y 轴线）"))
            if do_axes:
                if not ad.ocr_available():
                    log.append("未安装 OCR 库，无法自动读刻度。请运行：pip install rapidocr_onnxruntime，"
                               "或手动标定。")
                else:
                    self._set_status("正在 OCR 识别刻度数字…")
                    self.update_idletasks()
                    labels = ad.ocr_tokens(self.rgb)          # 全部文本：数字用于标定，其余用于图例
                    numeric = [l for l in labels if not math.isnan(l.value)]
                    xt, yt = ad.detect_ticks(self.rgb, frame)
                    cal, clog = ad.auto_calibrate(self.rgb, frame, numeric, xt, yt)
                    log += clog
                    if cal is not None:
                        self.apply_calibration(cal)
                        log.append("标定已自动完成，请移动鼠标核对状态栏坐标；不对可手动重标。")
                    else:
                        log.append("自动标定失败，请在“标定”页手动点四个刻度。")
            if do_curves:
                self._set_status("正在识别曲线…")
                self.update_idletasks()
                p = self._params()
                region = frame.inner_box()
                if self.region is not None:
                    r = self.region
                    region = (max(region[0], r[0]), max(region[1], r[1]), min(region[2], r[2]), min(region[3], r[3]))
                ex = self.active_exclude()
                if ex is not None:
                    frac = float(ex[region[1]:region[3] + 1, region[0]:region[2] + 1].mean())
                    if frac > 0:
                        log.append(f"检测到插图/照片等实心区域（占绘图区 {frac:.0%}），已排除（绿色淡显）")
                if not labels and ad.ocr_available():
                    try:
                        labels = ad.ocr_tokens(self.rgb)      # 用于排除图例文字、按图例命名
                    except Exception:
                        labels = []
                if labels:
                    tmask = ad.text_exclude_mask(self.rgb, labels, frame)
                    if tmask.any():
                        ex = tmask if ex is None else (ex | tmask)
                curves = ad.auto_extract_curves(self.rgb, region, tol=p["tol"], max_jump=p["max_jump"],
                                                max_gap=p["max_gap"], max_run=p["max_run"], exclude=ex)
                if labels and curves:
                    log += ad.name_curves_by_legend(self.rgb, labels, curves)
                existing = {c.name for c in self.curves}
                for c in curves:
                    if c.name.startswith("曲线") or c.name in existing:
                        c.name = self._unique_name()
                    existing.add(c.name)
                    self.curves.append(c)
                if curves:
                    self.cur_index = len(self.curves) - len(curves)
                    log.append(f"识别到 {len(curves)} 条曲线：" + "，".join(f"{c.name} {len(c)} 点" for c in curves))
                    log.append("如有漏掉的曲线（例如黑色/灰色曲线），可用“点击某条曲线提取”补充。")
                else:
                    log.append("没有识别到曲线。可试着增大颜色容差，或用“点击某条曲线提取”。")
        except Exception as ex:
            log.append(f"出错：{ex}")
        finally:
            self.config(cursor="")
        self.refresh_curve_list()
        self.redraw()
        self.nb.select(2 if do_curves else 0)
        messagebox.showinfo("自动识别结果", "\n".join(log))

    def clear_calibration(self, redraw=True):
        self.cal = AxisCalibration(log_x=self.log_x_var.get(), log_y=self.log_y_var.get())
        for key in ("x1", "x2", "y1", "y2"):
            self.cal_vars[key].set("")
            self.cal_px_labels[key].config(text="px: —")
        self.cal_status.config(text="状态：未标定", foreground="#b00")
        if redraw:
            self.redraw()

    # ------------------------------------------------------------------ region
    def _compute_exclude(self):
        """检测照片/插图等实心区域；失败不影响其他功能。"""
        self.exclude = None
        if self.rgb is None or not self.exclude_var.get():
            return
        try:
            import auto_detect as ad
            m = ad.dense_mask(self.rgb)
            self.exclude = m if m.any() else None
        except Exception:
            self.exclude = None

    def _on_toggle_exclude(self):
        self._compute_exclude()
        self._disp_cache = None
        self.redraw()

    def active_exclude(self):
        return self.exclude if (self.exclude_var.get() and self.exclude is not None) else None

    def effective_region(self):
        if self.region is not None:
            return self.region
        if self.limit_box_var.get() and self.cal.is_ready():
            return self.cal.plot_box(margin=3)
        return None

    def clear_region(self):
        self.region = None
        self.redraw()

    def _draw_rubber(self, p0, p1):
        self.canvas.delete("rubber")
        x0, y0 = self.img_to_canvas(*p0)
        x1, y1 = self.img_to_canvas(*p1)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00e5ff", dash=(4, 3), width=2, tags="rubber")

    # ------------------------------------------------------------------ extraction
    def _params(self):
        def geti(var, default):
            try:
                return int(var.get())
            except Exception:
                return default
        return dict(tol=geti(self.tol_var, 50), max_jump=geti(self.jump_var, 20),
                    max_gap=geti(self.gap_var, 30), max_run=geti(self.run_var, 80))

    def _unique_name(self, base="曲线"):
        names = {c.name for c in self.curves}
        i = len(self.curves) + 1
        while f"{base}{i}" in names:
            i += 1
        return f"{base}{i}"

    def _snap_to_ink(self, px, py, radius=5):
        """点击处若是背景色，吸附到周围最近的非背景像素（鼠标差一两个像素很常见）。"""
        h, w = self.rgb.shape[:2]
        try:
            import auto_detect as ad
            bg = ad.background_color(self.rgb)
        except Exception:
            return px, py
        def is_ink(x, y):
            return np.abs(self.rgb[y, x].astype(int) - bg).max() > 40
        if is_ink(px, py):
            return px, py
        best, best_d = None, None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < w and 0 <= y < h and is_ink(x, y):
                    d = dx * dx + dy * dy
                    if best_d is None or d < best_d:
                        best, best_d = (x, y), d
        return best if best else (px, py)

    def auto_extract(self, px, py, curve: Curve | None = None):
        if self.rgb is None:
            return
        p = self._params()
        px, py = self._snap_to_ink(px, py)
        color = pick_color(self.rgb, px, py)
        mask = color_mask(self.rgb, color, p["tol"])
        cols, rows = trace_curve(mask, px, py, region=self.effective_region(),
                                 max_jump=p["max_jump"], max_gap=p["max_gap"], max_run=p["max_run"],
                                 exclude=self.active_exclude())
        if cols.size == 0:
            messagebox.showinfo("自动提取", "在点击位置附近没有找到可追踪的曲线像素。\n"
                                "请点在曲线线条上，或增大颜色容差。")
            return
        if curve is None:
            curve = Curve(name=self._unique_name(), color=color, px=cols, py=rows, seed=(px, py))
            self.curves.append(curve)
            self.cur_index = len(self.curves) - 1
        else:
            curve.color, curve.px, curve.py, curve.seed = color, cols, rows, (px, py)
        self.refresh_curve_list()
        self.redraw()
        self._set_status(f"提取到 {cols.size} 个点，颜色 {color}")

    def re_extract(self):
        cv = self.current_curve()
        if cv is None:
            messagebox.showinfo("重新提取", "请先在曲线列表中选择一条自动提取的曲线。")
            return
        if cv.manual or cv.seed is None:
            messagebox.showinfo("重新提取", "所选曲线是手动取点曲线，无法重新自动提取。")
            return
        self.auto_extract(cv.seed[0], cv.seed[1], curve=cv)

    def start_manual(self):
        if self.image is None:
            messagebox.showinfo("手动取点", "请先打开图片。")
            return
        cv = Curve(name=self._unique_name("手动"), color=(255, 0, 255), manual=True)
        self.curves.append(cv)
        self.cur_index = len(self.curves) - 1
        self.refresh_curve_list()
        self.set_mode("manual")
        self.nb.select(1)

    def add_manual_point(self, ix, iy):
        cv = self.current_curve()
        if cv is None or not cv.manual:
            cv = Curve(name=self._unique_name("手动"), color=(255, 0, 255), manual=True)
            self.curves.append(cv)
            self.cur_index = len(self.curves) - 1
        self._undo_stack.append((cv, cv.px.copy(), cv.py.copy()))
        cv.add_point(ix, iy)
        self.refresh_curve_list()
        self.redraw()

    def undo(self):
        if not self._undo_stack:
            return
        cv, px, py = self._undo_stack.pop()
        cv.px, cv.py = px, py
        self.refresh_curve_list()
        self.redraw()

    # ------------------------------------------------------------------ curve list
    def current_curve(self) -> Curve | None:
        if self.cur_index is None or not (0 <= self.cur_index < len(self.curves)):
            return None
        return self.curves[self.cur_index]

    def refresh_curve_list(self):
        self.listbox.delete(0, "end")
        for i, c in enumerate(self.curves):
            kind = "手动" if c.manual else "自动"
            vis = "" if c.visible else "（隐藏）"
            self.listbox.insert("end", f"{c.name}  [{kind}] {len(c)} 点 {vis}")
            self.listbox.itemconfig(i, foreground=c.hex_color())
        if self.cur_index is not None and 0 <= self.cur_index < len(self.curves):
            self.listbox.selection_set(self.cur_index)
            self.listbox.see(self.cur_index)
        cv = self.current_curve()
        self.swatch.config(bg=cv.hex_color() if cv else "#ffffff")
        self.refresh_preview()

    def _on_select_curve(self, _e=None):
        sel = self.listbox.curselection()
        self.cur_index = int(sel[0]) if sel else None
        cv = self.current_curve()
        self.swatch.config(bg=cv.hex_color() if cv else "#ffffff")
        self.refresh_preview()
        self.redraw()

    def refresh_preview(self):
        self.tree.delete(*self.tree.get_children())
        cv = self.current_curve()
        if cv is None or len(cv) == 0:
            return
        if self.cal.validate() is None:
            x, y = cv.data(self.cal)
            fx = fy = lambda v: "%.6g" % v
            self.tree.heading("x", text="x")
            self.tree.heading("y", text="y")
        else:
            order = np.argsort(cv.px, kind="stable")
            x, y = cv.px[order], cv.py[order]
            fx = fy = lambda v: "%.1f" % v
            self.tree.heading("x", text="x (像素,未标定)")
            self.tree.heading("y", text="y (像素,未标定)")
        n = min(len(x), 2000)
        for i in range(n):
            self.tree.insert("", "end", values=(i + 1, fx(x[i]), fy(y[i])))

    def rename_curve(self):
        cv = self.current_curve()
        if cv is None:
            return
        name = simpledialog.askstring("重命名", "曲线名称：", initialvalue=cv.name, parent=self)
        if name:
            cv.name = name.strip()
            self.refresh_curve_list()
            self.redraw()

    def toggle_visible(self):
        cv = self.current_curve()
        if cv is None:
            return
        cv.visible = not cv.visible
        self.refresh_curve_list()
        self.redraw()

    def delete_curve(self):
        cv = self.current_curve()
        if cv is None:
            return
        if not messagebox.askyesno("删除曲线", f"确定删除“{cv.name}”（{len(cv)} 点）？"):
            return
        del self.curves[self.cur_index]
        self.cur_index = min(self.cur_index, len(self.curves) - 1) if self.curves else None
        self.refresh_curve_list()
        self.redraw()

    # ------------------------------------------------------------------ export
    def export(self, selected_only=False):
        if not self.curves:
            messagebox.showinfo("导出", "还没有提取任何曲线。")
            return
        err = self.sync_cal(show_error=False)
        if err:
            messagebox.showwarning("导出", "请先完成坐标轴标定：\n" + err)
            self.nb.select(0)
            return
        if selected_only:
            cv = self.current_curve()
            if cv is None:
                messagebox.showinfo("导出", "请先在列表中选择一条曲线。")
                return
            curves = [cv]
        else:
            curves = [c for c in self.curves if len(c) > 0]
        if not curves or all(len(c) == 0 for c in curves):
            messagebox.showinfo("导出", "所选曲线没有数据点。")
            return
        base = os.path.splitext(os.path.basename(self.image_path or "curve"))[0]
        path = filedialog.asksaveasfilename(
            title="导出数据", defaultextension=".csv", initialfile=base + "_data.csv",
            filetypes=[("CSV 文件(逗号分隔)", "*.csv"), ("文本文件(制表符分隔)", "*.txt *.dat"), ("所有文件", "*.*")])
        if not path:
            return
        delim = "\t" if path.lower().endswith((".txt", ".dat")) else ","
        try:
            n = int(self.npts_var.get())
        except Exception:
            n = 0
        try:
            export_csv(path, curves, self.cal, mode=self.export_mode.get(), n_points=n, delimiter=delim)
        except Exception as ex:
            messagebox.showerror("导出失败", str(ex))
            return
        messagebox.showinfo("导出完成", f"已导出 {len(curves)} 条曲线到：\n{path}")

    # ------------------------------------------------------------------ project
    def save_project(self):
        if self.image is None:
            messagebox.showinfo("保存工程", "请先打开图片。")
            return
        self.sync_cal(False)
        path = filedialog.asksaveasfilename(title="保存工程", defaultextension=".json",
                                            initialfile=os.path.splitext(os.path.basename(self.image_path))[0] + ".readcurve.json",
                                            filetypes=[("ReadCurve 工程", "*.json")])
        if not path:
            return
        data = {
            "image": os.path.relpath(self.image_path, os.path.dirname(path)) if self.image_path else None,
            "image_abs": self.image_path,
            "calibration": {k: getattr(self.cal, k) for k in
                            ("x1_px", "x1_val", "x2_px", "x2_val", "y1_px", "y1_val", "y2_px", "y2_val", "log_x", "log_y")},
            "region": list(self.region) if self.region else None,
            "limit_box": bool(self.limit_box_var.get()),
            "params": self._params(),
            "curves": [{
                "name": c.name, "color": list(c.color), "manual": c.manual, "seed": c.seed,
                "visible": c.visible, "px": c.px.tolist(), "py": c.py.tolist(),
            } for c in self.curves],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        messagebox.showinfo("保存工程", f"已保存：\n{path}")

    def load_project(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="打开工程", filetypes=[("ReadCurve 工程", "*.json")])
            if not path:
                return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            messagebox.showerror("打开工程失败", str(ex))
            return
        img = None
        for cand in (data.get("image"), data.get("image_abs")):
            if not cand:
                continue
            p = cand if os.path.isabs(cand) else os.path.join(os.path.dirname(path), cand)
            if os.path.exists(p):
                img = p
                break
        if img is None:
            messagebox.showwarning("打开工程", "找不到工程对应的图片文件，请手动选择。")
            img = filedialog.askopenfilename(title="选择工程图片", filetypes=IMG_TYPES)
            if not img:
                return
        self.open_image(img)
        c = data.get("calibration", {})
        self.apply_calibration(AxisCalibration(
            **{k: c.get(k) for k in ("x1_px", "x1_val", "x2_px", "x2_val", "y1_px", "y1_val", "y2_px", "y2_val")},
            log_x=bool(c.get("log_x")), log_y=bool(c.get("log_y"))))
        self.region = tuple(data["region"]) if data.get("region") else None
        self.limit_box_var.set(data.get("limit_box", True))
        prm = data.get("params", {})
        for var, key in ((self.tol_var, "tol"), (self.jump_var, "max_jump"),
                         (self.gap_var, "max_gap"), (self.run_var, "max_run")):
            if key in prm:
                var.set(prm[key])
        self.curves = []
        for cd in data.get("curves", []):
            self.curves.append(Curve(name=cd["name"], color=tuple(cd["color"]), manual=cd.get("manual", False),
                                     seed=tuple(cd["seed"]) if cd.get("seed") else None,
                                     visible=cd.get("visible", True),
                                     px=np.array(cd["px"], float), py=np.array(cd["py"], float)))
        self.cur_index = 0 if self.curves else None
        self.sync_cal(False)
        self.refresh_curve_list()
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def _display_image(self):
        """带排除区淡显的显示用图像（缓存）。"""
        ex = self.active_exclude()
        if ex is None:
            return self.image
        if getattr(self, "_disp_cache", None) is None:
            arr = self.rgb.copy()
            arr[ex] = (arr[ex] * 0.55 + np.array([60, 220, 60]) * 0.45).astype(np.uint8)
            self._disp_cache = Image.fromarray(arr)
        return self._disp_cache

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.image is None:
            cw, ch = c.winfo_width(), c.winfo_height()
            c.create_text(cw / 2, ch / 2, text="文件 → 打开图片（Ctrl+O）\n或把图片拖到命令行作为参数启动",
                          fill="#bbb", font=("Microsoft YaHei UI", 14), justify="center")
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        W, H = self.image.size
        z = self.zoom
        ix0 = max(0, int(math.floor((0 - self.ox) / z)))
        iy0 = max(0, int(math.floor((0 - self.oy) / z)))
        ix1 = min(W, int(math.ceil((cw - self.ox) / z)) + 1)
        iy1 = min(H, int(math.ceil((ch - self.oy) / z)) + 1)
        if ix1 > ix0 and iy1 > iy0:
            crop = self._display_image().crop((ix0, iy0, ix1, iy1))
            sw = max(1, int(round((ix1 - ix0) * z)))
            sh = max(1, int(round((iy1 - iy0) * z)))
            resample = Image.NEAREST if z >= 1 else Image.BILINEAR
            disp = crop.resize((sw, sh), resample)
            self._photo = ImageTk.PhotoImage(disp)
            c.create_image(ix0 * z + self.ox, iy0 * z + self.oy, anchor="nw", image=self._photo)

        # 提取区域
        reg = self.effective_region()
        if reg is not None:
            x0, y0 = self.img_to_canvas(reg[0] - 0.5, reg[1] - 0.5)
            x1, y1 = self.img_to_canvas(reg[2] + 0.5, reg[3] + 0.5)
            color = "#00e5ff" if self.region is not None else "#80d8ff"
            c.create_rectangle(x0, y0, x1, y1, outline=color, dash=(6, 3), width=1)

        # 曲线
        for i, cv in enumerate(self.curves):
            if not cv.visible or len(cv) == 0:
                continue
            sel = (i == self.cur_index)
            col = cv.hex_color()
            order = np.argsort(cv.px, kind="stable")
            xs = (cv.px[order] + 0.5) * z + self.ox
            ys = (cv.py[order] + 0.5) * z + self.oy
            width = 3 if sel else 2
            if len(cv) >= 2 and (not cv.manual or len(cv) > 1):
                pts = []
                for x, y in zip(xs, ys):
                    pts += [x, y]
                c.create_line(*pts, fill="white", width=width + 2)
                c.create_line(*pts, fill=col, width=width)
            if cv.manual or len(cv) < 400 or z >= 6:
                r = 4 if sel else 3
                for x, y in zip(xs, ys):
                    c.create_oval(x - r, y - r, x + r, y + r, fill=col, outline="white")
            # 名称标签
            c.create_text(xs[0], ys[0] - 12, text=cv.name, fill=col, anchor="s",
                          font=("Microsoft YaHei UI", 9, "bold"))

        # 标定点
        for key in ("x1", "x2", "y1", "y2"):
            px = getattr(self.cal, f"{key}_px")
            if px is None:
                continue
            val = self.cal_vars[key].get()
            if key.startswith("x"):
                cx = (px + 0.5) * z + self.ox
                c.create_line(cx, 0, cx, ch, fill="#ff9800", dash=(3, 4))
                yy = ch - 20
                if self.cal.y1_px is not None:
                    yy = (self.cal.y1_px + 0.5) * z + self.oy
                c.create_text(cx, yy + 14, text=f"{key.upper()}={val}", fill="#ff9800", anchor="n",
                              font=("Microsoft YaHei UI", 9, "bold"))
            else:
                cy = (px + 0.5) * z + self.oy
                c.create_line(0, cy, cw, cy, fill="#ff9800", dash=(3, 4))
                xx = 20
                if self.cal.x1_px is not None:
                    xx = (self.cal.x1_px + 0.5) * z + self.ox
                c.create_text(xx - 6, cy, text=f"{key.upper()}={val}", fill="#ff9800", anchor="e",
                              font=("Microsoft YaHei UI", 9, "bold"))
        self._set_status(None)

    # ------------------------------------------------------------------ help
    def show_help(self):
        win = tk.Toplevel(self)
        win.title("使用说明")
        win.geometry("640x560")
        txt = tk.Text(win, wrap="word", font=("Microsoft YaHei UI", 10), padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.config(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=6)


def main():
    app = ReadCurveApp()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        if sys.argv[1].lower().endswith(".json"):
            app.after(100, lambda: app.load_project(sys.argv[1]))
        else:
            app.after(100, lambda: app.open_image(sys.argv[1]))
    app.mainloop()


if __name__ == "__main__":
    main()
