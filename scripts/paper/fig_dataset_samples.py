#!/usr/bin/env python3
"""
fig_dataset_samples.py — annotated samples of MOPAD Site 2

Panel (a): one full UAV tile with every bounding box (coloured by class),
           showing tree density and object scale.
Panel (b): 5 classes x N examples; each cell is a crop around one object of
           that class (thick box), neighbouring objects drawn thin for context.
Legend: class name + instance count (train+val, counted from the labels).

Reproducible: random choices use --seed; the selected files and objects are
written to <out>_selection.txt.

Usage (from the repository root; needs the dataset):
  python scripts/paper/fig_dataset_samples.py
Options:
  --per-class 4      examples per class
  --seed 42
  --tile <name.jpg>  force the tile of panel (a) (default: the val tile with
                     the most diverse classes)
"""

import argparse
import glob
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from tandan_paths import P  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
LABELS = ["Dead", "Healthy", "Mismanaged", "Smallish", "Yellowish"]
COLORS = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4", "#e6c619"]  # red, green, orange, blue, yellow


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


RGB = [hex2rgb(c) for c in COLORS]

# Box line widths in pixels (tile rendered at 1400 px, crops at 220 px)
LW_TILE, LW_CTX, LW_TARGET = 7, 3, 8


def read_labels(lbl_path, W, H):
    """YOLO txt -> list of (cls, x1, y1, x2, y2) in pixels."""
    out = []
    if not os.path.isfile(lbl_path):
        return out
    with open(lbl_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            out.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def label_path(img_path):
    d, f = os.path.split(img_path)
    return os.path.join(os.path.dirname(d), "labels", os.path.splitext(f)[0] + ".txt")


def index_split(root, split):
    imgs = sorted(glob.glob(os.path.join(root, split, "images", "*")))
    return [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]


def draw_boxes(img, boxes, scale, width, highlight=None, hl_width=None):
    d = ImageDraw.Draw(img)
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        if highlight is not None and i == highlight:
            continue
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=RGB[c], width=width)
    if highlight is not None:
        c, x1, y1, x2, y2 = boxes[highlight]
        r = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        d.rectangle(r, outline=RGB[c], width=hl_width)
    return img


def make_crop(img_path, boxes, idx, out_px, context):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    c, x1, y1, x2, y2 = boxes[idx]
    side = max(x2 - x1, y2 - y1) * context
    side = max(side, 64)
    side = min(side, W, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    crop = im.crop((int(left), int(top), int(left + side), int(top + side)))
    s = out_px / crop.size[0]
    crop = crop.resize((out_px, out_px), Image.LANCZOS)
    # boxes in crop coordinates (keep those overlapping the crop)
    local, new_idx = [], None
    for j, (cj, a1, b1, a2, b2) in enumerate(boxes):
        if a2 < left or a1 > left + side or b2 < top or b1 > top + side:
            continue
        if j == idx:
            new_idx = len(local)
        local.append((cj, a1 - left, b1 - top, a2 - left, b2 - top))
    return draw_boxes(crop, local, s, width=LW_CTX, highlight=new_idx, hl_width=LW_TARGET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=P("site_2_only_dataset"))
    ap.add_argument("--out", default=P("figures/Fig02b_dataset_samples"))
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile", default=None)
    ap.add_argument("--context", type=float, default=2.6, help="crop side = context x longest box side")
    ap.add_argument("--min-size", type=float, default=14, help="ignore objects smaller than this (px)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # ---- index every object (train + val) ----
    counts = Counter()
    objs = defaultdict(list)          # cls -> [(img_path, box_idx)]
    val_imgs = index_split(a.root, "val")
    all_imgs = index_split(a.root, "train") + val_imgs
    size_cache, label_cache = {}, {}
    for p in all_imgs:
        with Image.open(p) as im:
            W, H = im.size
        size_cache[p] = (W, H)
        bx = read_labels(label_path(p), W, H)
        label_cache[p] = bx
        for i, (c, x1, y1, x2, y2) in enumerate(bx):
            counts[c] += 1
            w, h = x2 - x1, y2 - y1
            if min(w, h) >= a.min_size and x1 > 1 and y1 > 1 and x2 < W - 1 and y2 < H - 1:
                objs[c].append((p, i))
    print("Instances per class (train+val):", {CLASSES[k]: counts[k] for k in sorted(counts)})

    # ---- panel (a): the val tile with the most diverse classes ----
    if a.tile:
        tile = [p for p in val_imgs + all_imgs if os.path.basename(p) == a.tile][0]
    else:
        def score(p):
            cls = Counter(b[0] for b in label_cache[p])
            minority = sum(1 for k in (0, 2, 4) if cls[k] > 0)
            return (len(cls), minority, -abs(len(label_cache[p]) - 80))
        cand = sorted(val_imgs, key=score, reverse=True)
        top = [p for p in cand if score(p)[:2] == score(cand[0])[:2]]
        tile = rnd.choice(top[:10])
    tile_boxes = label_cache[tile]
    W, H = size_cache[tile]
    tile_px = 1400
    s = tile_px / max(W, H)
    tile_img = Image.open(tile).convert("RGB").resize((int(W * s), int(H * s)), Image.LANCZOS)
    tile_img = draw_boxes(tile_img, tile_boxes, s, width=LW_TILE)

    # ---- panel (b): crops per class ----
    chosen = {}
    for c in range(len(CLASSES)):
        pool = list(objs[c])
        rnd.shuffle(pool)
        picked, used_imgs = [], set()
        for p, i in pool:                      # prefer different images
            if p in used_imgs:
                continue
            picked.append((p, i))
            used_imgs.add(p)
            if len(picked) == a.per_class:
                break
        chosen[c] = picked

    # ---- layout ----
    n = a.per_class
    rows = len(CLASSES)
    fig_w, left, right, top, bottom, gap = 7.2, 0.008, 0.955, 0.935, 0.085, 0.04
    cell = fig_w * (right - left) / (rows + n + gap * (n + 1))       # cell side (inches)
    fig_h = cell * (rows + gap * (rows - 1)) / (top - bottom)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = fig.add_gridspec(rows, 1 + n, width_ratios=[rows + gap * (rows - 1)] + [1] * n,
                          left=left, right=right, top=top, bottom=bottom, wspace=gap, hspace=gap)
    ax_tile = fig.add_subplot(gs[:, 0])
    ax_tile.imshow(tile_img)
    ax_tile.set_anchor("N")
    ax_tile.set_xticks([]); ax_tile.set_yticks([])
    ax_tile.set_title("(a) Full UAV tile with all annotations (%d objects)" % len(tile_boxes),
                      fontsize=7.5, loc="left", pad=3)
    for sp in ax_tile.spines.values():
        sp.set_linewidth(0.5)

    for r, c in enumerate(range(len(CLASSES))):
        for k in range(n):
            ax = fig.add_subplot(gs[r, 1 + k])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(COLORS[c]); sp.set_linewidth(1.2)
            if k < len(chosen[c]):
                p, i = chosen[c][k]
                ax.imshow(make_crop(p, label_cache[p], i, 220, a.context))
            if k == n - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(LABELS[c], fontsize=7, color="black", labelpad=3, rotation=270, va="bottom")
            if r == 0 and k == 0:
                ax.set_title("(b) Examples per class (target object in bold)", fontsize=7.5, loc="left", pad=3)

    handles = [Patch(facecolor="none", edgecolor=COLORS[c], linewidth=1.5,
                     label="%s (n = %s)" % (LABELS[c], format(counts[c], ","))) for c in range(len(CLASSES))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False,
               handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(a.out + ".png", dpi=300)
    fig.savefig(a.out + ".pdf")
    with open(a.out + "_selection.txt", "w") as f:
        f.write("seed=%d per_class=%d context=%.2f min_size=%.0f\n" % (a.seed, n, a.context, a.min_size))
        f.write("panel_a_tile=%s (%dx%d, %d objects)\n" % (os.path.basename(tile), W, H, len(tile_boxes)))
        for c in range(len(CLASSES)):
            for p, i in chosen[c]:
                f.write("%s\t%s\tbox_idx=%d\n" % (CLASSES[c], os.path.basename(p), i))
    print("Tile panel (a):", os.path.basename(tile), "(%dx%d, %d objects)" % (W, H, len(tile_boxes)))
    print("Done:", a.out + ".png", a.out + ".pdf", a.out + "_selection.txt")


if __name__ == "__main__":
    main()
