"""
train_repc3k2.py — T2: YOLO11n with C3k2 -> RepC3k2 in the backbone (640 px)
============================================================================
(originally `exp-journal-1/final_using_yolo11/scripts/train_repc3k2.py`)

Hypothesis tested: a re-parameterisable block (multi-branch in training,
single conv after fuse) raises mAP at identical inference cost.
Default is one seed (screening filter), as reported in the paper.

Usage (from the repository root):
  python scripts/train/train_repc3k2.py --seeds 42 --epochs 1   # smoke test
  python scripts/train/train_repc3k2.py                         # seed 42 (paper)

Output: runs/T2_repc3k2_seed42/
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

DATASET_YAML = tandan_paths.DATASET_YAML
PRETRAINED = tandan_paths.pretrained("yolo11n.pt")
DEFAULT_CONFIG = _ROOT / "model_configs" / "yolo11n-repc3k2.yaml"

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]
SEEDS_DEFAULT    = [42]  # 1-seed quick filter by default

BASE_TRAIN_ARGS = {
    "data": str(DATASET_YAML), "epochs": 150, "imgsz": 640, "batch": 16,
    "workers": 8, "device": 0, "exist_ok": False, "verbose": True,
    "patience": 50, "save": True, "save_period": 10, "val": True,
    "plots": True, "cache": False, "amp": True,
}
PROFILE_IMGSZ = 640


def print_section(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def extract_background_fp(vr, names):
    try:
        cm = vr.confusion_matrix.matrix
    except Exception:
        return None
    nc = len(names)
    bg_col = cm[:, nc]
    total = bg_col[:nc].sum()
    if total <= 0:
        return None
    res = {"bg_fp_total_abs": float(total)}
    for i, name in enumerate(names):
        res[f"bg_as_{name}_frac"] = float(bg_col[i] / total)
    res["dominant_fp_class"] = names[int(np.argmax(bg_col[:nc]))]
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


def run_single(seed, epochs, config):
    from ultralytics import YOLO
    from register_repc3k2 import register_all
    register_all(verbose=True)

    run_name = f"T2_repc3k2_seed{seed}"
    run_dir = RUNS_DIR / run_name
    print(f"\n{'#'*72}\n  TAHAP 2 — Track B1: RepC3k2\n  Run: {run_name} | seed={seed} epochs={epochs}\n{'#'*72}\n")

    model = YOLO(str(config))
    if PRETRAINED.exists():
        model.load(str(PRETRAINED))
        print(f"  [pretrained] transferred from {PRETRAINED.name} (partial match)")
    else:
        print(f"  [pretrained] WARNING not found")

    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed}
    model.train(**args, project=str(RUNS_DIR), name=run_name)

    best = run_dir / "weights" / "best.pt"
    mb = YOLO(str(best))
    vr = mb.val(data=str(DATASET_YAML), imgsz=640, batch=16, device=0,
                verbose=True, plots=True, save_json=True)
    rows, m50, m5095, mr, ma = extract_metrics(vr)
    bgfp = extract_background_fp(vr, CLASS_NAMES)
    prof = profile_model(mb)

    fuse_ok = False
    try:
        fm = YOLO(str(best)); fm.model.fuse(); fuse_ok = True
        print("  [fuse] OK -- RepConv fused, TensorRT-ready")
    except Exception as e:
        print(f"  [fuse] FAILED: {e}")

    out = {"experiment": "T2_repc3k2", "run_name": run_name, "seed": seed,
           "epochs": epochs, "config": Path(config).name, "dataset": "site-2-only",
           "macro_mAP50": round(m50, 4), "macro_mAP50_95": round(m5095, 4),
           "minority_recall": round(mr, 4), "minority_AP50": round(ma, 4),
           "fuse_ok": fuse_ok, "background_fp": bgfp, "profile": prof, "per_class": rows}
    with open(run_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  [OK] {run_name}: mAP50-95={m5095:.4f} min_recall={mr:.4f} "
          f"GFLOPs={prof.get('GFLOPs')} GPU_FPS={prof.get('gpu_fps')} fuse={fuse_ok}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"])
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()

    print_section("TAHAP 2 — TRACK B1: RepC3k2")
    print(f"  Seeds: {args.seeds} | Epochs: {args.epochs}")
    print(f"  Config: {Path(args.config).name}")
    if len(args.seeds) == 1:
        print("  >> 1-SEED QUICK FILTER (3-seed nanti untuk model final) <<")
    if not DATASET_YAML.exists() or not PRETRAINED.exists():
        print("ERROR: dataset/pretrained missing"); return
    if not Path(args.config).exists():
        print(f"ERROR: config missing: {args.config}"); return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.seeds:
        try:
            runs.append(run_single(s, args.epochs, args.config))
        except Exception as e:
            print(f"  [FAIL] seed{s}: {e}")
            import traceback; traceback.print_exc()

    if runs:
        print_section("SUMMARY — TAHAP 2 (RepC3k2)")
        print(f"  {'Seed':<8}{'mAP50-95':>10}{'min_recall':>12}{'GFLOPs':>8}{'GPU_FPS':>10}{'fuse':>6}")
        for r in runs:
            p = r["profile"]
            print(f"  {r['seed']:<8}{r['macro_mAP50_95']:>10.4f}{r['minority_recall']:>12.4f}"
                  f"{str(p.get('GFLOPs')):>8}{str(p.get('gpu_fps')):>10}{str(r['fuse_ok']):>6}")
        if len(runs) > 1:
            import statistics
            m = [r["macro_mAP50_95"] for r in runs]
            print(f"\n  mAP50-95 mean±std: {statistics.mean(m):.4f} ± {statistics.stdev(m):.4f}")
        print(f"\n  COMPARE vs baseline T0: 0.9619 ± 0.0023")
        print(f"  H2: RepC3k2 menaikkan mAP di atas noise (>0.0023) dgn latency identik setelah fuse")


if __name__ == "__main__":
    main()
