"""
train_cb_focal.py — T1: YOLO11n + class-balanced focal loss (3 seeds, 640 px)
=============================================================================

Replaces the classification BCE with a class-balanced focal loss
(effective-number weights, Cui et al. 2019). Box (CIoU) and DFL losses stay
at the YOLO11 defaults. Compared against the yolo11n_seed* base runs.

Usage (from the repository root):
  python scripts/train/train_cb_focal.py --seeds 42 --epochs 1   # smoke test
  python scripts/train/train_cb_focal.py                         # 3 seeds (reported runs)
  python scripts/train/train_cb_focal.py --beta 0.999 --gamma 2.0

Output: runs/T1_lossmod_seed{42,123,456}/
Python 3.8 compatible.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

RUNS_DIR = tandan_paths.RUNS_DIR

# ============================================================
# CONFIG
# ============================================================
DATASET_YAML = tandan_paths.DATASET_YAML
PRETRAINED = tandan_paths.pretrained("yolo11n.pt")

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]
# Training instance counts (MOPAD Site 2) — for effective number weights.
# Kept exactly as used for the reported T1 runs. The verified count for "dead"
# is 234 (see data/mopad_site2.yaml); with beta = 0.999 the "dead" class
# weight differs by about 1 %.
CLASS_COUNTS     = [231, 89958, 431, 26644, 1473]
SEEDS            = [42, 123, 456]

BASE_TRAIN_ARGS = {
    "data"       : str(DATASET_YAML),
    "epochs"     : 150,
    "imgsz"      : 640,
    "batch"      : 16,
    "workers"    : 8,
    "device"     : 0,
    "exist_ok"   : False,
    "verbose"    : True,
    "patience"   : 50,
    "save"       : True,
    "save_period": 10,
    "val"        : True,
    "plots"      : True,
    "cache"      : False,
    "amp"        : True,
}
PROFILE_IMGSZ = 640


def print_section(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def extract_background_fp(vr, class_names):
    try:
        cm = vr.confusion_matrix.matrix
    except Exception:
        return None
    nc = len(class_names)
    bg_col = cm[:, nc]
    total = bg_col[:nc].sum()
    if total <= 0:
        return None
    res = {"bg_fp_total_abs": float(total)}
    for i, name in enumerate(class_names):
        res[f"bg_as_{name}_abs"] = float(bg_col[i])
        res[f"bg_as_{name}_frac"] = float(bg_col[i] / total)
    res["dominant_fp_class"] = class_names[int(np.argmax(bg_col[:nc]))]
    return res


def extract_metrics(results):
    ap50, ap5095 = results.box.ap50, results.box.maps
    p, r = results.box.p, results.box.r
    rows, min_r, min_a = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        a50 = float(ap50[i]) if i < len(ap50) else 0.0
        a95 = float(ap5095[i]) if i < len(ap5095) else 0.0
        pr = float(p[i]) if i < len(p) else 0.0
        rc = float(r[i]) if i < len(r) else 0.0
        if name in MINORITY_CLASSES:
            min_r.append(rc); min_a.append(a50)
        rows.append({"class": name, "is_minority": name in MINORITY_CLASSES,
                     "AP@.5": round(a50, 4), "AP@.5:.95": round(a95, 4),
                     "Precision": round(pr, 4), "Recall": round(rc, 4)})
    return (rows, float(ap50.mean()), float(ap5095.mean()),
            sum(min_r)/len(min_r) if min_r else 0.0,
            sum(min_a)/len(min_a) if min_a else 0.0)


def profile_model(mb):
    prof = {}
    try:
        prof["params_M"] = round(sum(p.numel() for p in mb.model.parameters())/1e6, 3)
    except Exception:
        prof["params_M"] = None
    try:
        from thop import profile as tp
        d = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(next(mb.model.parameters()).device)
        macs, _ = tp(mb.model, inputs=(d,), verbose=False)
        prof["GFLOPs"] = round(macs*2/1e9, 2)
    except Exception:
        prof["GFLOPs"] = None
    try:
        mb.model.eval()
        d = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(next(mb.model.parameters()).device)
        with torch.no_grad():
            for _ in range(10): _ = mb.model(d)
        torch.cuda.synchronize()
        ts = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.perf_counter(); _ = mb.model(d); torch.cuda.synchronize()
                ts.append((time.perf_counter()-t0)*1000)
        prof["gpu_fps"] = round(1000/(sum(ts)/len(ts)), 1)
    except Exception:
        prof["gpu_fps"] = None
    return prof


def run_single(seed, epochs, beta, gamma):
    from ultralytics import YOLO
    from patch_cb_focal import patch_cb_focal, verify_cb_focal

    run_name = f"T1_lossmod_seed{seed}"
    run_dir = RUNS_DIR / run_name
    print(f"\n{'#'*72}\n  T1: CB-Focal Loss (Cui beta={beta}, gamma={gamma})")
    print(f"  Run: {run_name} | seed={seed} epochs={epochs}\n{'#'*72}\n")

    # Patch CB-Focal BEFORE building/training
    patch_cb_focal(CLASS_COUNTS, beta=beta, gamma=gamma, verbose=True)

    model = YOLO(str(PRETRAINED))
    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed}
    model.train(**args, project=str(RUNS_DIR), name=run_name)

    verify_cb_focal(model, verbose=True)

    best = run_dir / "weights" / "best.pt"
    mb = YOLO(str(best))
    vr = mb.val(data=str(DATASET_YAML), imgsz=640, batch=16, device=0,
                verbose=True, plots=True, save_json=True)
    rows, m50, m5095, mr, ma = extract_metrics(vr)
    bgfp = extract_background_fp(vr, CLASS_NAMES)
    prof = profile_model(mb)

    out = {"experiment": "T1_lossmod", "run_name": run_name, "seed": seed,
           "epochs": epochs, "cui_beta": beta, "focal_gamma": gamma,
           "dataset": "site-2-only",
           "macro_mAP50": round(m50, 4), "macro_mAP50_95": round(m5095, 4),
           "minority_recall": round(mr, 4), "minority_AP50": round(ma, 4),
           "background_fp": bgfp, "profile": prof, "per_class": rows}
    with open(run_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  [OK] {run_name}: mAP50-95={m5095:.4f} min_recall={mr:.4f} min_AP50={ma:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"])
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()

    print_section("T1: CLASS-BALANCED FOCAL LOSS")
    print(f"  Seeds: {args.seeds} | Epochs: {args.epochs}")
    print(f"  Cui beta: {args.beta} | Focal gamma: {args.gamma}")
    if not DATASET_YAML.exists():
        print(f"ERROR: dataset missing"); return
    if not PRETRAINED.exists():
        print(f"ERROR: pretrained missing"); return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.seeds:
        try:
            runs.append(run_single(s, args.epochs, args.beta, args.gamma))
        except Exception as e:
            print(f"  [FAIL] seed{s}: {e}")
            import traceback; traceback.print_exc()

    if runs:
        print_section("SUMMARY — T1")
        print(f"  {'Seed':<8}{'mAP50-95':>10}{'min_recall':>12}{'min_AP50':>10}")
        for r in runs:
            print(f"  {r['seed']:<8}{r['macro_mAP50_95']:>10.4f}"
                  f"{r['minority_recall']:>12.4f}{r['minority_AP50']:>10.4f}")
        if len(runs) > 1:
            import statistics
            m = [r["macro_mAP50_95"] for r in runs]
            mr = [r["minority_recall"] for r in runs]
            print(f"\n  mAP50-95 mean±std: {statistics.mean(m):.4f} ± {statistics.stdev(m):.4f}")
            print(f"  min_recall mean±std: {statistics.mean(mr):.4f} ± {statistics.stdev(mr):.4f}")
        print(f"\n  COMPARE with the yolo11n base runs (run train_base_models.py first)")
        print(f"  H1: the loss change improves minority-class metrics over the base runs")


if __name__ == "__main__":
    main()
