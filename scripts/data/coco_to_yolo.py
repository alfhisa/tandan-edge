#!/usr/bin/env python3
"""
coco_to_yolo.py — Convert the official MOPAD release (COCO format) to the YOLO
layout used by this repository.

MOPAD (https://github.com/rs-dl/MOPAD) ships each site as
    <site>/train2017/*.jpg   <site>/val2017/*.jpg
    <site>/annotations/instances_train2017.json, instances_val2017.json
This script writes
    <out>/train/images, <out>/train/labels   (from train2017)
    <out>/val/images,   <out>/val/labels     (from val2017)
with one "class cx cy w h" line per box (normalised to the image size) and
class ids matched BY NAME to: 0 dead, 1 healthy, 2 mismanaged, 3 smallish, 4 yellowish.
Images are copied (or symlinked with --link). Images without boxes get an
empty label file.

Usage (from the repository root):
  python scripts/data/coco_to_yolo.py --src /path/to/MOPAD/site2 --out datasets/mopad_site2

Check the printed counts against the dataset used here (Site 2):
  images  train 1,803 / val 500
  boxes   train: dead 234, healthy 89,958, mismanaged 431, smallish 26,644, yellowish 1,473
          val total 32,846
"""

import argparse
import json
import os
import shutil
from collections import Counter

CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]


def convert(src, out, split_in, split_out, link):
    ann_path = os.path.join(src, "annotations", "instances_%s.json" % split_in)
    coco = json.load(open(ann_path))
    cat = {}
    for c in coco["categories"]:
        name = c["name"].strip().lower()
        if name not in CLASSES:
            raise SystemExit("unknown category %r in %s (expected %s)" % (c["name"], ann_path, CLASSES))
        cat[c["id"]] = CLASSES.index(name)

    img_dir = os.path.join(out, split_out, "images")
    lbl_dir = os.path.join(out, split_out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    lines = {im["id"]: [] for im in coco["images"]}
    images = {im["id"]: im for im in coco["images"]}
    counts = Counter()
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = images[a["image_id"]]
        W, H = float(im["width"]), float(im["height"])
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        k = cat[a["category_id"]]
        lines[a["image_id"]].append("%d %.6f %.6f %.6f %.6f" % (k, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        counts[k] += 1

    for iid, im in images.items():
        fn = os.path.basename(im["file_name"])
        s = os.path.join(src, split_in, fn)
        d = os.path.join(img_dir, fn)
        if not os.path.exists(d):
            if link:
                os.symlink(os.path.abspath(s), d)
            else:
                shutil.copy2(s, d)
        with open(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines[iid]) + ("\n" if lines[iid] else ""))

    print("%-5s images %5d | boxes %6d | %s" % (
        split_out, len(images), sum(counts.values()),
        ", ".join("%s %d" % (CLASSES[k], counts[k]) for k in range(len(CLASSES)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MOPAD site folder with train2017/, val2017/, annotations/")
    ap.add_argument("--out", default="datasets/mopad_site2")
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    a = ap.parse_args()
    convert(a.src, a.out, "train2017", "train", a.link)
    convert(a.src, a.out, "val2017", "val", a.link)
    print("YOLO dataset ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
