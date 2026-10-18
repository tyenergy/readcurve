# ReadCurve —— 图表曲线数字化工具

从图表图片（PNG/JPG 等）中读取曲线数值，导出为 CSV / TXT 数据文件。
适用于论文插图、扫描图、截图等只有图片没有原始数据的场景。

## 安装与运行

需要 Python 3.8+（OCR 库 `rapidocr_onnxruntime` 为可选，用于自动读刻度）：

```bash
pip install -r requirements.txt
python readcurve.py                # 启动后用“打开图片”
python readcurve.py sample_chart.png   # 直接打开某张图
```

Windows 下也可双击 `run.bat`，或把图片拖到 `run.bat` 上。

## 一键自动（推荐先试）

打开图片后点顶部 **★ 一键自动**，程序会：

1. 检测坐标轴线 / 边框，确定绘图区；
2. 用 OCR 读出 X、Y 轴的刻度数字，稳健拟合（自动剔除误读）完成标定，自动判断线性/对数轴；
3. 按颜色找出绘图区内所有曲线并逐列追踪（自动排除图例、网格线、坐标轴）。

然后移动鼠标核对状态栏显示的数据坐标，确认无误后直接导出。
也可以只做一半：标定页的“自动识别坐标轴”、提取页的“自动识别图中所有曲线”。

命令行（不打开界面，适合脚本 / Agent 调用，详见 [docs/SKILL_USAGE.md](docs/SKILL_USAGE.md)）：

```bash
python auto_detect.py chart.png -o chart.csv --preview check.png
python auto_detect.py fig.png --crop 8,15,360,320 --xticks=0,0.5 --yticks=0,140 -o a.csv   # 多子图先裁剪；OCR 失败时直接给刻度值
python auto_detect.py fig.png --cal "x:59=-100,240=100;logy:474=10,710=0.001" -o a.csv     # 或直接给像素=数值
python batch_digitize.py figs_dir/ -o out_dir/        # 批处理：每图一个 CSV + 核对图 + summary.json
```

`--preview` 输出一张叠加了识别结果的核对图（曲线洋红、标定线橙色、刻度线蓝色、排除区绿色）；
`--mode common --n 200` 可统一 X 网格；`--no-exclude` 不排除实心区域；`--no-ocr` 跳过 OCR。
标定失败时日志会列出轴旁每个文本的像素位置，可据此用 `--cal`。

多子图与图例：

```bash
python auto_detect.py fig.png --panels -o out.csv --preview check.png   # 自动找出 (a)(b)(c)… 各坐标系，逐个输出 out_a.csv, out_b.csv …
python auto_detect.py fig.png --panel b -o out.csv                      # 只处理面板 b
```

曲线会按图例自动命名（OCR 图例文字 + 匹配文字左侧样条的颜色），CSV 列名如 `measurement_x, measurement_y`；
每条曲线附带质量指标（覆盖率、点密度、断口数），见日志和批处理的 `summary.json`。
绘图区内的文字（图例、标注）会自动排除，不会被当成曲线。

图中有照片、模型图、填充色块等插图时：程序会先找出这些“实心大块”区域（曲线和数据标记都是细的，
插图是实心的），在界面上以绿色淡显，自动提取和点击提取都不会进入；曲线穿过插图的那一段按间断处理。
误判时可在“提取”页取消勾选“自动排除插图/照片”。`examples/` 里有一张带实物照片插图的论文图及识别结果。

在 8 张不同来源的真实论文图（Scientific Reports、MDPI、arXiv、网页截图风格）上的测试结果见
[docs/SKILL_USAGE.md](docs/SKILL_USAGE.md) 末尾。

自动识别的局限：
- 黑色/灰色曲线：能自动提取，但若与黑色图例框、文字贴得很近，个别点可能偏移，请核对；
- 刻度数字太小、带上标（如 10³）或被遮挡时 OCR 可能失败，此时请手动标定四个点；
- 没有坐标轴线的图（纯网格或无框），绘图区会退化为整张图，建议用“框选区域”限定。

## 手动 / 半自动步骤

1. **打开图片**：滚轮缩放，按住右键（或中键）拖动平移，`F` 键适应窗口。
2. **标定坐标轴**（① 标定页）：
   依次点 `点 X1` → 在图上点击 X 轴某个刻度 → 输入该刻度数值；同样完成 X2、Y1、Y2。
   两点相距越远越准。对数坐标请勾选“对数 X/Y 轴”。
   标定完成后移动鼠标，状态栏实时显示数据坐标，可核对标定是否正确。
3. **提取曲线**（② 提取页）：
   - **自动提取**：点 `点击曲线自动提取`，然后在图中点一下曲线本身。程序以该点颜色为基准，
     在容差范围内逐列追踪整条曲线，支持虚线和间断。可连续点击多条曲线。
     效果不理想时调节“颜色容差 / 最大跳变 / 最大间断”，再点 `重新提取所选曲线`。
   - **手动取点**：同色曲线相互交叉、或曲线颜色与网格相近时，用手动模式逐点点击；
     右键删除最近点，`Ctrl+Z` 撤销。
   - 提取区域默认限制在四个标定点围成的框内（自动排除图例、标题），也可 `框选区域` 手动指定。
4. **导出**（③ 曲线/导出页）：
   - *各曲线独立列*：每条曲线输出 `名称_x, 名称_y` 两列。
   - *统一 X 网格*：所有曲线插值到同一组 x，便于直接对比。
   - 文件后缀 `.csv` 用逗号分隔，`.txt`/`.dat` 用制表符分隔。
   - “导出点数”为 0 时输出全部采样点（每像素列一个点）。
5. **保存工程**：文件 → 保存工程，把标定、区域和曲线存成 `.json`，下次可继续。

## 文件说明

| 文件 | 作用 |
|---|---|
| `readcurve.py` | 图形界面（tkinter） |
| `curve_core.py` | 核心算法：坐标标定、颜色追踪、重采样、导出。可单独 `import` 用于批处理 |
| `auto_detect.py` | 全自动识别：坐标轴框检测、OCR 刻度标定、自动找曲线；也是命令行批处理入口 |
| `test_core.py` | 生成合成测试图 `sample_chart.png` 并验证提取精度：`python test_core.py` |
| `batch_digitize.py` | 批处理入口：目录或多张图 → CSV + 核对图 + `summary.json` |
| `table_ocr.py` | 表格图片 → CSV（行列版面分析 + 数值解析，`--parsed` 输出值/误差两列）：`python table_ocr.py t.png -o t.csv` |
| `test_table.py` | 表格识别回归测试（合成三线表与中文网格表）：`python test_table.py` |
| `test_auto.py` | 全自动识别的回归测试：`python test_auto.py` |
| `docs/SKILL_USAGE.md` | 面向 Agent / 论文复现 skill 的调用说明：流程、日志解读、失败处理 |
| `sample_chart.png` | 示例图（红色实线 sin(x)，蓝色虚线 cos(x)e^(-x/8)） |
| `examples/` | 真实论文图示例：`s11.png` 原图、`s11_data.csv` 一键自动导出结果、`s11_check.png` 识别叠加图 |

## 在脚本中调用核心算法

```python
import numpy as np
from PIL import Image
from curve_core import AxisCalibration, Curve, color_mask, pick_color, trace_curve, export_csv

rgb = np.asarray(Image.open("chart.png").convert("RGB"))
cal = AxisCalibration(x1_px=80, x1_val=0, x2_px=760, x2_val=10,
                      y1_px=540, y1_val=-1, y2_px=60, y2_val=1)
sx, sy = 216, 82                      # 曲线上任意一点的像素坐标
color = pick_color(rgb, sx, sy)
cols, rows = trace_curve(color_mask(rgb, color, tol=50), sx, sy, region=cal.plot_box(margin=3))
curve = Curve("sin", color=color, px=cols, py=rows)
x, y = curve.data(cal)                # 数据坐标
export_csv("out.csv", [curve], cal)
```

## 精度与限制

- 自动提取按“每个像素列取同色像素段的中心”采样，精度约 1 像素；曲线很陡处误差稍大。
- 带数据标记（方块、圆点）的曲线：标记中心落在线上，逐列取中心即可，空心标记会自动合并成一段。
- 假定坐标轴与图像边平行。歪斜的扫描图请先旋转校正。
- 同一颜色的多条曲线交叉时，追踪会沿“离上一列最近”的方向走，交叉后可能串线；
  此时可缩小提取区域分段提取，或改用手动取点。
