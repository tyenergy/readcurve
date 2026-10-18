# -*- coding: utf-8 -*-
"""
Batch digitize: run the fully automatic pipeline on many chart images.

    python batch_digitize.py <image-or-dir> [more ...] -o <outdir> [--no-ocr] [--tol 50]

For every image writes  <outdir>/<name>_data.csv  and  <outdir>/<name>_check.png
(recognition overlay for visual verification) and prints a summary table.
A machine-readable summary goes to <outdir>/summary.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from auto_detect import auto_digitize, digitize_panels, find_panels, parse_crop, save_preview
from curve_core import AxisCalibration, export_csv

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(EXT):
                    files.append(os.path.join(p, f))
        else:
            files += sorted(glob.glob(p)) or [p]
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--tol", type=float, default=50)
    ap.add_argument("--mode", choices=["separate", "common"], default="separate")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--crop", help="x0,y0,x1,y1 (pixels or 0-1 fractions); applied to every input")
    ap.add_argument("--panels", action="store_true", help="split multi-panel figures and digitize each panel")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    summary = []
    for path in collect(args.inputs):
        name = os.path.splitext(os.path.basename(path))[0]
        rec = {"image": path, "name": name}
        def run_one(rgb, rec, suffix="", frame=None):
            res = auto_digitize(rgb, use_ocr=not args.no_ocr, tol=args.tol, frame=frame)
            rec["log"] = res.log
            rec["frame"] = list(res.frame.box)
            rec["calibrated"] = res.calibration is not None
            rec["curves"] = [{"name": c.name, "color": list(c.color), "points": len(c), **c.meta} for c in res.curves]
            check = os.path.join(args.outdir, name + suffix + "_check.png")
            save_preview(rgb, res, check)
            rec["check"] = check
            if res.curves:
                cal = res.calibration or AxisCalibration(0, 0, 1, 1, 0, 0, 1, 1)
                out = os.path.join(args.outdir, name + suffix + "_data.csv")
                export_csv(out, res.curves, cal, mode=args.mode, n_points=args.n)
                rec["csv"] = out
                rec["csv_is_pixels"] = res.calibration is None
            return rec

        try:
            img = Image.open(path).convert("RGB")
            if args.crop:
                img = img.crop(parse_crop(args.crop, img.size))
            rgb = np.asarray(img)
            panels = find_panels(rgb)
            rec["panels"] = [{"label": p.label, "box": list(p.box), "crop": list(p.crop)} for p in panels]
            if args.panels and panels:
                rec["panel_results"] = []
                for p in panels:
                    sub = np.ascontiguousarray(rgb[p.crop[1]:p.crop[3], p.crop[0]:p.crop[2]])
                    prec = run_one(sub, {"panel": p.label, "crop": list(p.crop)}, "_" + p.label, frame=p.frame())
                    rec["panel_results"].append(prec)
                rec["calibrated"] = all(r["calibrated"] for r in rec["panel_results"])
                rec["curves"] = [c for r in rec["panel_results"] for c in r["curves"]]
            else:
                run_one(rgb, rec)
        except Exception as ex:
            rec["error"] = str(ex)
        summary.append(rec)
        status = "ERR " + rec["error"] if "error" in rec else \
            f"cal={'OK' if rec['calibrated'] else 'NO'} curves={len(rec['curves'])}" + \
            (f" panels={len(rec['panels'])}" if len(rec.get('panels', [])) > 1 else "")
        print(f"{name:28s} {status}")
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
