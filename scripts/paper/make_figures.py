#!/usr/bin/env python3
"""
make_figures.py — Data figures, rebuilt from results/.

Output: results/figures/<name>.png (300 dpi) + .pdf

  F2a Fig02a_class_distribution  instances per class, train/val (needs the dataset)
  F3  Fig03_accuracy_vs_fps      mAP50-95 vs Jetson FPS (TensorRT FP16), 15 FPS line
  F4  Fig04_train_infer_matrix   heatmap: training x inference resolution (FP32, FP16)
  F5  Fig05_pytorch_vs_tensorrt  PyTorch vs TensorRT FPS per resolution and precision
  F6  Fig06_thermal_30min        GPU/CPU temperature and GPU load over 30 min
  F7  Fig07_confusion_matrix     deployed configuration (train 768, run 704)
  F8  Fig08_qualitative          ground truth vs prediction on val crops
  S1  FigS1_training_curves      mAP50-95 per epoch (train 640/704/768, seed 42)
  S2  FigS2_mismanaged_ap        AP50-95 of "mismanaged" per seed and configuration

Usage (from the repository root):
  python scripts/paper/make_figures.py                # all
  python scripts/paper/make_figures.py --only F3 F4   # some
F2a, F7 and F8 need the dataset; F7 and F8 also need the deployed weights
(weights/tandan-edge_yolo11n_train768.pt) and a GPU (~1 min). The rest read results/ only.
"""

import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import tandan_paths  # noqa: E402
from tandan_paths import P  # noqa: E402  (maps the original workspace paths to this repository)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

OUT = P("figures")
CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
LABELS = ["Dead", "Healthy", "Mismanaged", "Smallish", "Yellowish"]
CCOL = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4", "#e6c619"]   # same as fig_dataset_samples
INK, MUTED, GRID = "#222222", "#666666", "#e5e5e5"
C_OURS, C_BASE, C_RES = "#c0392b", "#7f7f7f", "#2c6e9b"
# deployed weights: release asset if present, otherwise your own R4 run
FINAL_W = str(tandan_paths.FINAL_WEIGHTS) if tandan_paths.FINAL_WEIGHTS.exists() \
    else str(tandan_paths.RUNS_DIR / "R4_res768_seed42" / "weights" / "best.pt")
DATA = str(tandan_paths.DATASET_YAML)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("  ->", os.path.join(OUT, name + ".png"))


def rcsv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- data loaders
def reeval_summary():
    return {r["group"]: r for r in rcsv(P("reeval/reeval_summary.csv"))}


def train_infer():
    s = {}
    for r in rcsv(P("reeval/train_infer_summary.csv")):
        s[(int(r["train_imgsz"]), int(r["infer_imgsz"]), r["precision"])] = r
    return s


def trt_benchmark():
    b = {}
    for r in rcsv(P("jetson_results/trt_results/benchmark_summary.csv")):
        b[(int(r["resolution"]), r["precision"])] = float(r["throughput_qps"])
    return b


def trt_models():
    out = {}
    for m in ("yolov8n", "yolo11n", "yolo12n", "yolo26n"):
        p = P("jetson_results/onnx_cmp_logs/%s_trt.log") % m
        fps = None
        if os.path.isfile(p):
            mt = re.search(r"Throughput: ([0-9.]+) qps", open(p, errors="replace").read())
            fps = float(mt.group(1)) if mt else None
        out[m] = fps
    return out



# ---------------------------------------------------------------- F2a
def fig_class_distribution():
    cnt = {sp: [0] * len(CLASSES) for sp in ("train", "val")}
    nimg = {}
    for sp in cnt:
        files = glob.glob(P("site_2_only_dataset/%s/labels/*.txt") % sp)
        nimg[sp] = len(files)
        for fp in files:
            for line in open(fp):
                p = line.split()
                if p:
                    cnt[sp][int(float(p[0]))] += 1
    with open(os.path.join(OUT, "Fig02a_class_counts.csv"), "w") as f:
        f.write("class,train,val,total\n")
        for k, c in enumerate(CLASSES):
            f.write("%s,%d,%d,%d\n" % (c, cnt["train"][k], cnt["val"][k], cnt["train"][k] + cnt["val"][k]))
    print("  images train/val:", nimg, "| instances:", cnt)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(CLASSES)); w = 0.38
    for off, sp, alpha, hatch in ((-w / 2, "train", 1.0, None), (w / 2, "val", 0.45, None)):
        bars = ax.bar(x + off, cnt[sp], w, color=CCOL, alpha=alpha, edgecolor="#333333", linewidth=0.4, zorder=3)
        for b, v in zip(bars, cnt[sp]):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, format(v, ","), ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.set_yscale("log")
    lo = min(min(cnt["train"]), min(cnt["val"]))
    ax.set_ylim(max(1, lo / 2.5), max(cnt["train"]) * 20)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.set_ylabel("Instances (log scale)")
    ax.grid(axis="x", visible=False)
    handles = [Patch(fc="#777777", ec="#333333", lw=0.4, label="Training (%d images)" % nimg["train"]),
               Patch(fc="#777777", alpha=0.45, ec="#333333", lw=0.4, label="Validation (%d images)" % nimg["val"])]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=6.2)
    save(fig, "Fig02a_class_distribution")

# ---------------------------------------------------------------- F3
def fig_accuracy_fps():
    rs, ti, tb, tm = reeval_summary(), train_infer(), trt_benchmark(), trt_models()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axvspan(15, 23.5, color="#eaf4ea", zorder=0)
    ax.axvline(15, color="#4a8a4a", lw=0.8, ls="--", zorder=1)
    ax.text(15.25, 0.9362, "real-time (≥ 15 FPS)", color="#4a8a4a", fontsize=6.3, va="bottom")

    base = [("yolov8n", "YOLOv8n"), ("yolo11n", "YOLO11n"), ("yolo12n", "YOLO12n")]
    for key, lab in base:
        if tm.get(key) is None:
            continue
        m, s = float(rs[key]["mAP50_95_mean"]), float(rs[key]["mAP50_95_std"])
        ax.errorbar(tm[key], m, yerr=s, fmt="o", ms=4, color=C_BASE, mfc="white", mew=1, capsize=2, lw=0.8, zorder=3)
        off = (5, -8) if key == "yolo11n" else (5, -3)
        ax.annotate(lab + " @640", (tm[key], m), xytext=off, textcoords="offset points", fontsize=6.3, color=MUTED)

    pts = [(704, 704, "train 704 → infer 704", C_RES, "s", (-6, -3)),
           (768, 768, "train 768 → infer 768", C_RES, "s", (-4, 6)),
           (768, 640, "train 768 → infer 640", C_RES, "s", (6, 2)),
           (768, 704, "train 768 → infer 704 (deployed)", C_OURS, "*", (6, 3))]
    for tr, inf, lab, col, mk, off in pts:
        r = ti[(tr, inf, "fp32")]
        m, s = float(r["mAP50_95_mean"]), float(r["mAP50_95_std"])
        fps = tb[(inf, "fp16")]
        ax.errorbar(fps, m, yerr=s, fmt=mk, ms=9 if mk == "*" else 4.5, color=col, capsize=2, lw=0.8, zorder=4)
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(lab, (fps, m), xytext=off, textcoords="offset points", fontsize=6.3, color=col, ha=ha,
                    fontweight="bold" if mk == "*" else "normal")

    ax.set_xlim(8.5, 23.5)
    ax.set_ylim(0.935, 0.976)
    ax.set_xlabel("Jetson Nano throughput, TensorRT FP16 (FPS)")
    ax.set_ylabel("mAP50-95 (mean ± SD, 3 seeds)")
    ax.text(0.03, 0.36, "YOLO26n not shown:\nTensorRT 8.2 conversion fails\n(unsupported Mod operator)",
            transform=ax.transAxes, fontsize=5.8, color=MUTED, style="italic", va="top")
    save(fig, "Fig03_accuracy_vs_fps")


# ---------------------------------------------------------------- F4
def fig_train_infer_matrix():
    ti, tb = train_infer(), trt_benchmark()
    res = [640, 704, 768]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"wspace": 0.28})
    vmin, vmax = 0.945, 0.972
    for ax, prec, ttl in zip(axes, ("fp32", "fp16"), ("(a) FP32", "(b) FP16")):
        M = np.array([[float(ti[(tr, inf, prec)]["mAP50_95_mean"]) for inf in res] for tr in res])
        S = np.array([[float(ti[(tr, inf, prec)]["mAP50_95_std"]) for inf in res] for tr in res])
        im = ax.imshow(M, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                dark = M[i, j] > (vmin + vmax) / 2 + 0.004
                ax.text(j, i, "%.4f\n± %.4f" % (M[i, j], S[i, j]), ha="center", va="center", fontsize=6.6,
                        color="white" if dark else INK)
        ax.add_patch(Rectangle((0.5, 1.5), 1, 1, fill=False, ec=C_OURS, lw=2))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["%d\n(%.1f FPS)" % (r, tb[(r, "fp16")]) for r in res])
        ax.set_yticks(range(3)); ax.set_yticklabels(res)
        ax.set_xlabel("Inference resolution (Jetson FP16 FPS)")
        if prec == "fp32":
            ax.set_ylabel("Training resolution")
        ax.set_title(ttl + " — mAP50-95", loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    save(fig, "Fig04_train_infer_matrix")


# ---------------------------------------------------------------- F5
def fig_pytorch_vs_trt():
    tb = trt_benchmark()
    pt = {(int(r["resolution"]), r["precision"]): float(r["model_fps"]) for r in rcsv(P("jetson_results/pytorch_benchmark.csv"))}
    res = [640, 704, 768]
    series = [("PyTorch FP32", lambda r: pt[(r, "fp32")], "#bdbdbd"),
              ("PyTorch FP16", lambda r: pt[(r, "fp16")], "#8c8c8c"),
              ("TensorRT FP32", lambda r: tb[(r, "fp32")], "#9ecae1"),
              ("TensorRT INT8*", lambda r: tb[(r, "int8")], "#6baed6"),
              ("TensorRT FP16", lambda r: tb[(r, "fp16")], C_OURS)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    w = 0.16
    x = np.arange(len(res))
    for k, (lab, fn, col) in enumerate(series):
        vals = [fn(r) for r in res]
        bars = ax.bar(x + (k - 2) * w, vals, w, color=col, label=lab, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, "%.1f" % v, ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.axhline(15, color="#4a8a4a", ls="--", lw=0.8, zorder=2)
    ax.text(-0.45, 15.3, "15 FPS", color="#4a8a4a", fontsize=6, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(["%d px" % r for r in res])
    ax.set_ylabel("Throughput on Jetson Nano (FPS)")
    ax.set_ylim(0, 25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, handlelength=1, columnspacing=0.8)
    ax.text(0.0, -0.24, "*INT8 engine built without calibration; the Maxwell GPU has no INT8\n acceleration, so INT8 runs at FP32 speed.",
            transform=ax.transAxes, fontsize=5.4, color=MUTED)
    save(fig, "Fig05_pytorch_vs_tensorrt")


# ---------------------------------------------------------------- F6
def fig_thermal():
    path = P("jetson_results/thermal_results_30min/tegrastats.log")
    gpu, cpu, util = [], [], []
    for line in open(path, errors="replace"):
        g = re.search(r"GPU@([0-9.]+)C", line)
        c = re.search(r"CPU@([0-9.]+)C", line)
        u = re.search(r"GR3D_FREQ (\d+)%", line)
        if g and c and u:
            gpu.append(float(g.group(1))); cpu.append(float(c.group(1))); util.append(int(u.group(1)))
    t = np.arange(len(gpu)) / 60.0          # tegrastats interval 1000 ms
    def smooth(v, k=15):
        v = np.asarray(v, float)
        return np.convolve(v, np.ones(k) / k, mode="same")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    a1.plot(t, gpu, color=C_OURS, lw=0.3, alpha=0.35)
    a1.plot(t, cpu, color=C_RES, lw=0.3, alpha=0.35)
    a1.plot(t[7:-7], smooth(gpu)[7:-7], color=C_OURS, lw=1.1, label="GPU")
    a1.plot(t[7:-7], smooth(cpu)[7:-7], color=C_RES, lw=1.1, label="CPU")
    a1.set_ylabel("Temperature (°C)")
    a1.legend(loc="lower right", frameon=False, ncol=2)
    a1.set_ylim(20, 45)
    a1.text(0.01, 0.95, "Sustained TensorRT FP16 @704 for 30 min: 17.0 FPS, no throttling",
            transform=a1.transAxes, fontsize=6.2, va="top", color=MUTED)
    a2.plot(t, util, color="#555555", lw=0.4)
    a2.set_ylabel("GPU util. (%)")
    a2.set_ylim(0, 105)
    a2.set_xlabel("Time (min)")
    a2.set_xlim(0, t[-1])
    save(fig, "Fig06_thermal_30min")


# ---------------------------------------------------------------- F7
def fig_confusion():
    from ultralytics import YOLO
    m = YOLO(FINAL_W).val(data=DATA, split="val", imgsz=704, batch=16, conf=0.001, iou=0.7, half=False,
                          device=0, plots=True, verbose=False, project=str(tandan_paths.RUNS_DIR / "_tmp_cm"), name="cm", exist_ok=True)
    M = np.asarray(m.confusion_matrix.matrix)            # [pred, gt], index 5 = background
    np.save(os.path.join(P("reeval"), "cm_final_R4seed42_infer704.npy"), M)
    nc = len(CLASSES)
    names = LABELS + ["Background"]
    N = M.copy().astype(float)
    colsum = N[:, :nc].sum(0)
    Nn = np.zeros_like(N)
    Nn[:, :nc] = N[:, :nc] / np.maximum(colsum, 1)          # normalised per GT class
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    show = Nn[:, :nc]
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(nc + 1):
        for j in range(nc):
            v, n = show[i, j], int(N[i, j])
            if n == 0:
                continue
            ax.text(j, i, ("%.2f\n(%s)" % (v, format(n, ","))) if v >= 0.005 else "<0.01\n(%d)" % n, ha="center", va="center",
                    fontsize=5.6, color="white" if v > 0.5 else INK)
    ax.set_xticks(range(nc)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(nc + 1)); ax.set_yticklabels(names)
    ax.set_xlabel("Ground truth"); ax.set_ylabel("Predicted")
    for sp in ax.spines.values():
        sp.set_visible(False)
    bg = N[:nc, nc]
    parts = ["%s %d" % (LABELS[k], int(bg[k])) for k in range(nc) if bg[k] > 0]
    txt = ("Columns normalised per ground-truth class; counts in brackets. Background false positives\n"
           "(predictions without matching GT, total %d): " % int(bg.sum())) + ", ".join(parts)
    ax.text(0, -0.34, txt, transform=ax.transAxes, fontsize=5.4, color=MUTED, va="top")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    save(fig, "Fig07_confusion_matrix")


# ---------------------------------------------------------------- F8
def fig_qualitative(n_rows=3, crop=480, seed=7):
    import random
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    rnd = random.Random(seed)
    model = YOLO(FINAL_W)
    imgs = sorted(glob.glob(P("site_2_only_dataset/val/images/*")))

    def labels(p, W, H):
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        out = []
        for line in open(lp):
            c, cx, cy, w, h = line.split()[:5]
            c, cx, cy, w, h = int(float(c)), float(cx), float(cy), float(w), float(h)
            out.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        return out

    # find crops that contain minority classes (mismanaged, dead, yellowish)
    targets = []
    for want in (2, 0, 4):
        rnd.shuffle(imgs)
        for p in imgs:
            W, H = Image.open(p).size
            L = labels(p, W, H)
            hits = [b for b in L if b[0] == want]
            if hits:
                b = rnd.choice(hits)
                cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
                x0 = int(min(max(cx - crop / 2, 0), W - crop)); y0 = int(min(max(cy - crop / 2, 0), H - crop))
                targets.append((p, x0, y0, L, want))
                break
    targets = targets[:n_rows]

    def draw(img, boxes, x0, y0, s, lw=7):
        d = ImageDraw.Draw(img)
        for c, a1, b1, a2, b2 in boxes:
            if a2 < x0 or a1 > x0 + crop or b2 < y0 or b1 > y0 + crop:
                continue
            d.rectangle([(a1 - x0) * s, (b1 - y0) * s, (a2 - x0) * s, (b2 - y0) * s],
                        outline=tuple(int(CCOL[c][i:i + 2], 16) for i in (1, 3, 5)), width=lw)
        return img

    fig, axes = plt.subplots(len(targets), 2, figsize=(3.5, 1.68 * len(targets) + 0.35),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.05})
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.07)
    axes = np.atleast_2d(axes)
    out_px = 640
    for r, (p, x0, y0, L, want) in enumerate(targets):
        im = Image.open(p).convert("RGB")
        res = model.predict(p, imgsz=704, conf=0.25, iou=0.7, verbose=False)[0]
        P = [(int(c), *map(float, xy)) for c, xy in zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist())]
        s = out_px / crop
        base = im.crop((x0, y0, x0 + crop, y0 + crop)).resize((out_px, out_px), Image.LANCZOS)
        for k, (bx, ttl) in enumerate(((L, "Ground truth"), (P, "Prediction (train 768 → infer 704)"))):
            ax = axes[r, k]
            ax.imshow(draw(base.copy(), bx, x0, y0, s)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(ttl, fontsize=7)
            if k == 0:
                ax.set_ylabel("contains: " + LABELS[want], fontsize=6.5)
    handles = [Patch(fc="none", ec=CCOL[c], lw=1.5, label=LABELS[c]) for c in range(len(CLASSES))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.53, 0.0),
               handlelength=1, columnspacing=0.9)
    with open(os.path.join(OUT, "Fig08_qualitative_selection.txt"), "w") as f:
        for p, x0, y0, _, want in targets:
            f.write("%s\tcrop_xy=(%d,%d)\tsize=%d\ttarget=%s\n" % (os.path.basename(p), x0, y0, crop, CLASSES[want]))
    save(fig, "Fig08_qualitative")


# ---------------------------------------------------------------- S1
def fig_training_curves():
    runs = [(P("exp-journal-1/multiseed_baseline/yolo11n_seed42"), "train 640", C_BASE),
            (P("tandan-yolo/runs/R5_res704_seed42"), "train 704", C_RES),
            (P("tandan-yolo/runs/R4_res768_seed42"), "train 768", C_OURS)]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for path, lab, col in runs:
        rows = rcsv(os.path.join(path, "results.csv"))
        k = [c for c in rows[0] if "mAP50-95" in c][0]
        ep = [int(float(r[[c for c in r if c.strip() == "epoch"][0]])) for r in rows]
        ax.plot(ep, [float(r[k]) for r in rows], color=col, lw=1, label=lab)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val. mAP50-95 (training log)")
    ax.set_ylim(0.6, 1.0); ax.legend(frameon=False, loc="lower right")
    save(fig, "FigS1_training_curves")


# ---------------------------------------------------------------- S2
def fig_mismanaged():
    rows = rcsv(P("reeval/train_infer_runs.csv"))
    cfg = [(640, 640, "640→640"), (704, 704, "704→704"), (768, 640, "768→640"), (768, 704, "768→704\n(deployed)"), (768, 768, "768→768")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, (tr, inf, lab) in enumerate(cfg):
        v = [float(r["ap5095_mismanaged"]) for r in rows if int(r["train_imgsz"]) == tr and int(r["infer_imgsz"]) == inf and r["precision"] == "fp32"]
        col = C_OURS if (tr, inf) == (768, 704) else C_RES
        ax.scatter([i] * len(v), v, color=col, s=14, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [st.mean(v)] * 2, color=col, lw=1.4)
    y12 = [float(r["ap5095_mismanaged"]) for r in rcsv(P("reeval/reeval_runs.csv")) if r["group"] == "yolo12n"]
    ax.scatter([5] * len(y12), y12, color=C_BASE, s=14, zorder=3)
    ax.plot([4.8, 5.2], [st.mean(y12)] * 2, color=C_BASE, lw=1.4)
    ax.set_xticks(range(6)); ax.set_xticklabels([c[2] for c in cfg] + ["YOLO12n\n640→640"], fontsize=6.3)
    ax.set_ylabel("AP50-95, mismanaged class")
    ax.set_xlabel("Training → inference resolution (FP32, 3 seeds)")
    save(fig, "FigS2_mismanaged_ap")


FIGS = {"F2a": fig_class_distribution, "F3": fig_accuracy_fps, "F4": fig_train_infer_matrix, "F5": fig_pytorch_vs_trt, "F6": fig_thermal,
        "F7": fig_confusion, "F8": fig_qualitative, "S1": fig_training_curves, "S2": fig_mismanaged}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(FIGS))
    a = ap.parse_args()
    for k in a.only:
        print(k)
        try:
            FIGS[k]()
        except Exception as e:  # one failing figure does not stop the others
            print("  FAILED %s: %s: %s" % (k, type(e).__name__, e))
