#!/usr/bin/env python3
"""
train_infer_matrix.py — Training resolution x inference resolution (FP32 and FP16)

Produces the data behind Fig. 4 and Table 6 (the key finding: train at 768 px,
run at 704 px). Same protocol as reeval_all.py (Ultralytics 8.4.48, val Site 2,
batch 16, conf 0.001, NMS IoU 0.7). FP16 = PyTorch half precision on the GPU.

Models per training resolution (3 seeds each):
  640 -> runs/yolo11n_seed{42,123,456}     (train_base_models.py)
  704 -> runs/R5_res704_seed{42,123,456}   (train_tandan.py)
  768 -> runs/R4_res768_seed{42,123,456}   (train_tandan.py; seed 42 = deployed)
Inference: 640, 704, 768 x FP32, FP16 -> 54 evaluations (~15 min)

Also writes the validation plots (PR curve, confusion matrix) of the deployed
configuration R4_res768_seed42 @ 704 FP32 to runs/val/R4_res768_seed42_infer704/.

Output (results/reeval/):
  train_infer_runs.csv     one row per seed x inference size x precision (+ per-class AP)
  train_infer_summary.csv  mean ± SD per training size x inference size x precision

Usage (from the repository root):
  nohup python -u scripts/eval/train_infer_matrix.py > train_infer.log 2>&1 &
"""

import csv
import os
import statistics
import sys

from ultralytics import YOLO

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import tandan_paths  # noqa: E402

DATA = str(tandan_paths.DATASET_YAML)
CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
SEEDS = (42, 123, 456)
_R = str(tandan_paths.RUNS_DIR)
TRAIN = {
    640: os.path.join(_R, "yolo11n_seed{s}"),
    704: os.path.join(_R, "R5_res704_seed{s}"),
    768: os.path.join(_R, "R4_res768_seed{s}"),
}
INFER = (640, 704, 768)
OUT = os.path.join(str(tandan_paths.RESULTS_DIR), "reeval")


def evaluate(weights, imgsz, half, plots=False, name=None):
    kw = dict(data=DATA, split="val", imgsz=imgsz, batch=16, conf=0.001, iou=0.7,
              half=half, device=0, workers=4, verbose=False, plots=plots)
    if plots:
        kw.update(project=os.path.join(_R, "val"), name=name, exist_ok=True)
    m = YOLO(weights).val(**kw)
    maps = list(m.box.maps)
    return m, maps


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    total = len(TRAIN) * len(SEEDS) * len(INFER) * 2
    k = 0
    for tr, pat in TRAIN.items():
        for s in SEEDS:
            w = os.path.join(pat.format(s=s), "weights", "best.pt")
            for inf in INFER:
                for half in (False, True):
                    k += 1
                    m, maps = evaluate(w, inf, half)
                    r = {"train_imgsz": tr, "infer_imgsz": inf, "precision": "fp16" if half else "fp32",
                         "seed": s, "weights": os.path.relpath(w, REPO),
                         "mAP50_95": round(float(m.box.map), 4), "mAP50": round(float(m.box.map50), 4),
                         "P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4)}
                    for i, c in enumerate(CLASSES):
                        r["ap5095_" + c] = round(float(maps[i]), 4)
                    rows.append(r)
                    print("[%2d/%d] train%d infer%d %s seed%d  mAP50-95=%.4f  mismanaged=%.4f" % (
                        k, total, tr, inf, r["precision"], s, r["mAP50_95"], r["ap5095_mismanaged"]), flush=True)

    with open(os.path.join(OUT, "train_infer_runs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Ringkasan mean ± std
    summ = []
    for tr in TRAIN:
        for inf in INFER:
            for p in ("fp32", "fp16"):
                g = [r for r in rows if r["train_imgsz"] == tr and r["infer_imgsz"] == inf and r["precision"] == p]
                row = {"train_imgsz": tr, "infer_imgsz": inf, "precision": p, "n": len(g)}
                for key in ["mAP50_95", "mAP50", "ap5095_mismanaged", "ap5095_dead", "ap5095_yellowish", "ap5095_smallish"]:
                    v = [x[key] for x in g]
                    row[key + "_mean"] = round(statistics.mean(v), 4)
                    row[key + "_std"] = round(statistics.stdev(v), 4)
                summ.append(row)
    with open(os.path.join(OUT, "train_infer_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        wr.writeheader()
        wr.writerows(summ)

    print("\n=== RINGKASAN mAP50-95 (mean ± std, 3 seed) ===")
    print("train  infer   FP32               FP16")
    for tr in TRAIN:
        for inf in INFER:
            a = [x for x in summ if x["train_imgsz"] == tr and x["infer_imgsz"] == inf]
            f32 = [x for x in a if x["precision"] == "fp32"][0]
            f16 = [x for x in a if x["precision"] == "fp16"][0]
            print("%5d  %5d   %.4f ± %.4f    %.4f ± %.4f" % (
                tr, inf, f32["mAP50_95_mean"], f32["mAP50_95_std"], f16["mAP50_95_mean"], f16["mAP50_95_std"]))

    # Plot val untuk konfigurasi final
    print("\nValidation plots for the deployed configuration (R4 seed42 @704)...", flush=True)
    evaluate(os.path.join(TRAIN[768].format(s=42), "weights", "best.pt"), 704, False,
             plots=True, name="R4_res768_seed42_infer704")
    print("Done: results/reeval/train_infer_runs.csv, results/reeval/train_infer_summary.csv")


if __name__ == "__main__":
    main()
