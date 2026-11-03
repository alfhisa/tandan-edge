"""
train_vfl.py — Phase 1: Varifocal Loss ablation (B0 baseline vs B1 VFL)
========================================================================
Trains YOLOv12n with Varifocal Loss (B1) for the BAE-YOLOv12n Phase 1
ablation, 3-seed protocol, Site 2 only. Auto-documents results including
the KEY metric: background false-positive rate (from confusion matrix).

B0 (baseline, BCE) = the yolo12n_seed* runs of train_base_models.py.
This script runs B1 (VFL) so B0 vs B1 can be compared apple-to-apple.

Usage (from the repository root):
  python scripts/train/train_vfl.py --seeds 42 --epochs 1   # smoke test
  python scripts/train/train_vfl.py                         # 3 seeds
  python scripts/train/train_vfl.py --alpha 0.75 --gamma 2.0

Output: runs/B1_vfl_seed{42,123,456}/

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
_BAE_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_BAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAE_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ============================================================
# CONFIG
# ============================================================
DATASET_YAML = tandan_paths.DATASET_YAML
PRETRAINED = tandan_paths.pretrained("yolo12n.pt")
RUNS_DIR = tandan_paths.RUNS_DIR

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]
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
    "plots"      : True,   # generates confusion_matrix.png for background-FP metric
    "cache"      : False,
    "amp"        : True,
}

PROFILE_IMGSZ = 640


# ============================================================
# HELPERS
# ============================================================
def print_section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def extract_background_fp(val_results, class_names):
    """
    Extract background false-positive metrics from the confusion matrix.

    Ultralytics confusion matrix convention: columns = True, rows = Predicted.
    The last row/col is 'background'. The 'background' COLUMN (true=background)
    holds predictions where the model detected an object in a true-background
    region -> these are the background false positives we care about (the 69%
    healthy problem).

    Returns dict with per-class background-FP fractions and the dominant one.
    """
    try:
        cm = val_results.confusion_matrix.matrix  # shape (nc+1, nc+1)
    except Exception as e:
        print(f"  [bg-fp] confusion matrix unavailable: {e}")
        return None

    nc = len(class_names)
    # background column index = nc (last)
    bg_col = cm[:, nc]  # predictions for true-background regions, length nc+1
    total_bg = bg_col.sum()

    result = {"background_col_total": float(total_bg)}
    if total_bg > 0:
        for i, name in enumerate(class_names):
            result[f"bg_as_{name}"] = float(bg_col[i])
            result[f"bg_as_{name}_frac"] = float(bg_col[i] / total_bg)
        # background correctly ignored = cm[nc, nc]
        result["bg_as_background"] = float(bg_col[nc])
        result["bg_as_background_frac"] = float(bg_col[nc] / total_bg)
        # dominant false positive class
        fp_only = bg_col[:nc]
        dom_idx = int(np.argmax(fp_only))
        result["dominant_fp_class"] = class_names[dom_idx]
        result["dominant_fp_frac"] = float(fp_only[dom_idx] / total_bg)
        # overall background false-positive fraction (any foreground pred on bg)
        result["total_bg_fp_frac"] = float(fp_only.sum() / total_bg)
    return result


def extract_metrics(results):
    ap50   = results.box.ap50
    ap5095 = results.box.maps
    p      = results.box.p
    r      = results.box.r
    rows, min_r, min_a = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        a50  = float(ap50[i])   if i < len(ap50)   else 0.0
        a5095= float(ap5095[i]) if i < len(ap5095) else 0.0
        pr   = float(p[i])      if i < len(p)      else 0.0
        rc   = float(r[i])      if i < len(r)      else 0.0
        if name in MINORITY_CLASSES:
            min_r.append(rc); min_a.append(a50)
        rows.append({"class": name, "is_minority": name in MINORITY_CLASSES,
                     "AP@.5": round(a50, 4), "AP@.5:.95": round(a5095, 4),
                     "Precision": round(pr, 4), "Recall": round(rc, 4)})
    m50   = float(ap50.mean())
    m5095 = float(ap5095.mean())
    mr    = sum(min_r) / len(min_r) if min_r else 0.0
    ma    = sum(min_a) / len(min_a) if min_a else 0.0
    return rows, m50, m5095, mr, ma


def profile_model(model_best):
    profile = {}
    best_pt = model_best.ckpt_path if hasattr(model_best, "ckpt_path") else None
    if best_pt and Path(best_pt).exists():
        profile["model_size_mb"] = round(Path(best_pt).stat().st_size / 1e6, 2)
    try:
        n = sum(p.numel() for p in model_best.model.parameters())
        profile["params_M"] = round(n / 1e6, 3)
    except Exception:
        profile["params_M"] = None
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(
            next(model_best.model.parameters()).device)
        macs, _ = thop_profile(model_best.model, inputs=(dummy,), verbose=False)
        profile["GFLOPs"] = round(macs * 2 / 1e9, 2)
    except Exception:
        profile["GFLOPs"] = None
    try:
        model_best.model.eval()
        dummy = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(
            next(model_best.model.parameters()).device)
        with torch.no_grad():
            for _ in range(10):
                _ = model_best.model(dummy)
        torch.cuda.synchronize()
        ts = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.perf_counter()
                _ = model_best.model(dummy)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        profile["gpu_fps"] = round(1000 / (sum(ts) / len(ts)), 1)
    except Exception:
        profile["gpu_fps"] = None
    return profile


def run_single(seed, epochs, alpha, gamma):
    from ultralytics import YOLO
    from patch_vfl import patch_vfl, verify_vfl

    run_name = f"B1_vfl_seed{seed}"
    run_dir  = RUNS_DIR / run_name

    print(f"\n{'#'*72}")
    print(f"  B1 — VFL (alpha={alpha}, gamma={gamma})")
    print(f"  Run  : {run_name}  |  seed={seed}  epochs={epochs}")
    print(f"{'#'*72}\n")

    # Patch VFL BEFORE building/training
    patch_vfl(alpha=alpha, gamma=gamma, verbose=True)

    # Build from pretrained (same as baseline for fair comparison)
    model = YOLO(str(PRETRAINED))

    # Confirm no leftover class_weights (keep VFL isolated)
    if hasattr(model.model, "class_weights"):
        print("  [warn] model has class_weights attr -- removing for clean VFL ablation")
        delattr(model.model, "class_weights")

    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed}
    model.train(**args, project=str(RUNS_DIR), name=run_name)

    # Verify VFL was actually used
    verify_vfl(model, verbose=True)

    # Validate best
    best_pt = run_dir / "weights" / "best.pt"
    model_best = YOLO(str(best_pt))
    val_results = model_best.val(data=str(DATASET_YAML), imgsz=640, batch=16,
                                 device=0, verbose=True, plots=True, save_json=True)

    rows, m50, m5095, mr, ma = extract_metrics(val_results)
    bg_fp = extract_background_fp(val_results, CLASS_NAMES)
    profile = profile_model(model_best)

    output = {
        "experiment"     : "B1_vfl",
        "run_name"       : run_name,
        "seed"           : seed,
        "epochs"         : epochs,
        "vfl_alpha"      : alpha,
        "vfl_gamma"      : gamma,
        "dataset"        : "site-2-only",
        "macro_mAP50"    : round(m50, 4),
        "macro_mAP50_95" : round(m5095, 4),
        "minority_recall": round(mr, 4),
        "minority_AP50"  : round(ma, 4),
        "background_fp"  : bg_fp,     # <-- KEY metric for H1
        "profile"        : profile,
        "per_class"      : rows,
    }
    with open(str(run_dir / f"{run_name}_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  [OK] {run_name}")
    print(f"  mAP50-95 = {m5095:.4f} | min_recall = {mr:.4f} | GPU FPS = {profile.get('gpu_fps')}")
    if bg_fp:
        print(f"  background->healthy FP frac = {bg_fp.get('bg_as_healthy_frac')}")
        print(f"  dominant FP class = {bg_fp.get('dominant_fp_class')} "
              f"({bg_fp.get('dominant_fp_frac')})")
        print(f"  total background FP frac = {bg_fp.get('total_bg_fp_frac')}")

    return output


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 1 VFL training (B1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    print_section("PHASE 1 — VARIFOCAL LOSS ABLATION (B1)")
    print(f"  Seeds  : {args.seeds}")
    print(f"  Epochs : {args.epochs}")
    print(f"  VFL    : alpha={args.alpha}, gamma={args.gamma}")
    print(f"  Output : {RUNS_DIR}")
    if args.epochs == 1:
        print(f"  >> SMOKE TEST MODE <<")

    if not DATASET_YAML.exists():
        print(f"ERROR: dataset not found: {DATASET_YAML}")
        return
    if not PRETRAINED.exists():
        print(f"ERROR: pretrained not found: {PRETRAINED}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for seed in args.seeds:
        try:
            r = run_single(seed, args.epochs, args.alpha, args.gamma)
            all_runs.append(r)
        except Exception as e:
            print(f"\n  [FAIL] seed{seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_runs:
        print_section("PHASE 1 SUMMARY — B1 (VFL)")
        print(f"  {'Seed':<8} {'mAP50-95':>10} {'bg->healthy':>12} {'total bg-FP':>12}")
        print(f"  {'-'*46}")
        for r in all_runs:
            bg = r.get("background_fp") or {}
            print(f"  {r['seed']:<8} {r['macro_mAP50_95']:>10.4f} "
                  f"{str(round(bg.get('bg_as_healthy_frac', 0), 3)):>12} "
                  f"{str(round(bg.get('total_bg_fp_frac', 0), 3)):>12}")
        if len(all_runs) > 1:
            import statistics
            m = [r['macro_mAP50_95'] for r in all_runs]
            print(f"\n  mAP50-95 mean +/- std: {statistics.mean(m):.4f} "
                  f"+/- {statistics.stdev(m):.4f}")
        print(f"\n  COMPARE against baseline B0 (multiseed_baseline):")
        print(f"    B0 mAP50-95 = 0.9696 +/- 0.0026")
        print(f"    B0 bg->healthy FP = 0.69 (69%)  <-- target: reduce this")


if __name__ == "__main__":
    main()
