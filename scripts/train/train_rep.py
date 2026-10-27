"""
train_rep.py — Efficiency E1: A2C2fRep training (A2C2f -> RepConv)
==================================================================
Trains YOLOv12n with A2C2f replaced by A2C2fRep in the backbone (E1
conservative variant). Supports 1-seed quick-check or full 3-seed protocol.

Auto-documents: mAP50, mAP50-95, per-class, params, GFLOPs, GPU FPS, and
whether model.fuse() succeeds (critical for TensorRT deployment).

(originally `exp-journal-1/eff/scripts/train_rep.py`)

Usage (from the repository root):
  python scripts/train/train_rep.py --seeds 42 --epochs 1   # smoke test
  python scripts/train/train_rep.py                         # 3 seeds (paper)

Output: runs/E1_seed{42,123,456}/

Python 3.8 compatible.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_EFF_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_EFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_EFF_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ============================================================
# CONFIG
# ============================================================
DATASET_YAML = tandan_paths.DATASET_YAML
PRETRAINED = tandan_paths.pretrained("yolo12n.pt")
DEFAULT_CONFIG = _EFF_ROOT / "model_configs" / "yolo12n-rep-e1.yaml"
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
    "plots"      : True,
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


def profile_model(model_best, test_fuse=True):
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


def run_single(exp_name, config_path, seed, epochs):
    from ultralytics import YOLO
    from register_rep import register_all

    register_all(verbose=True)

    run_name = f"{exp_name}_seed{seed}"
    run_dir  = RUNS_DIR / run_name

    print(f"\n{'#'*72}")
    print(f"  {exp_name} — A2C2fRep (attention -> RepConv)")
    print(f"  Config : {Path(config_path).name}")
    print(f"  Run    : {run_name}  |  seed={seed}  epochs={epochs}")
    print(f"{'#'*72}\n")

    # Build from custom config
    model = YOLO(str(config_path))

    # Transfer pretrained weights (partial match: cv1/cv2/non-attention layers
    # inherit; RepConv blocks start random). Same approach as baseline/A1.
    if PRETRAINED.exists():
        model.load(str(PRETRAINED))
        print(f"  [pretrained] transferred from {PRETRAINED.name} (partial match expected)")
    else:
        print(f"  [pretrained] WARNING: not found, training from scratch!")

    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed}
    model.train(**args, project=str(RUNS_DIR), name=run_name)

    # Validate best
    best_pt = run_dir / "weights" / "best.pt"
    model_best = YOLO(str(best_pt))
    val_results = model_best.val(data=str(DATASET_YAML), imgsz=640, batch=16,
                                 device=0, verbose=True, plots=True, save_json=True)

    rows, m50, m5095, mr, ma = extract_metrics(val_results)
    profile = profile_model(model_best)

    # Test fuse (critical for deployment) on a fresh copy
    fuse_ok = False
    try:
        fuse_model = YOLO(str(best_pt))
        fuse_model.model.fuse()
        fuse_ok = True
        print("  [fuse] model.fuse() OK -- RepConv fused, TensorRT-ready")
    except Exception as e:
        print(f"  [fuse] FAILED: {e}")

    output = {
        "experiment"     : exp_name,
        "run_name"       : run_name,
        "config"         : Path(config_path).name,
        "seed"           : seed,
        "epochs"         : epochs,
        "dataset"        : "site-2-only",
        "macro_mAP50"    : round(m50, 4),
        "macro_mAP50_95" : round(m5095, 4),
        "minority_recall": round(mr, 4),
        "minority_AP50"  : round(ma, 4),
        "fuse_ok"        : fuse_ok,
        "profile"        : profile,
        "per_class"      : rows,
    }
    with open(str(run_dir / f"{run_name}_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  [OK] {run_name}")
    print(f"  mAP50-95 = {m5095:.4f} | min_recall = {mr:.4f} | "
          f"params = {profile.get('params_M')}M | GFLOPs = {profile.get('GFLOPs')} | "
          f"GPU FPS = {profile.get('gpu_fps')} | fuse = {fuse_ok}")

    return output


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Efficiency E1 training (A2C2fRep)")
    parser.add_argument("--exp", type=str, default="E1")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"])
    args = parser.parse_args()

    print_section(f"EFFICIENCY {args.exp} — A2C2fRep TRAINING")
    print(f"  Config : {args.config}")
    print(f"  Seeds  : {args.seeds}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Output : {RUNS_DIR}")
    if args.epochs == 1:
        print(f"  >> SMOKE TEST MODE <<")
    if len(args.seeds) == 1:
        print(f"  >> SINGLE-SEED QUICK CHECK (run full 3-seed after if promising) <<")

    if not DATASET_YAML.exists():
        print(f"ERROR: dataset not found: {DATASET_YAML}")
        return
    if not Path(args.config).exists():
        print(f"ERROR: config not found: {args.config}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for seed in args.seeds:
        try:
            r = run_single(args.exp, args.config, seed, args.epochs)
            all_runs.append(r)
        except Exception as e:
            print(f"\n  [FAIL] {args.exp} seed{seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_runs:
        print_section(f"SUMMARY — {args.exp}")
        print(f"  {'Seed':<8} {'mAP50-95':>10} {'min_recall':>12} {'GFLOPs':>8} {'GPU FPS':>10} {'fuse':>6}")
        print(f"  {'-'*58}")
        for r in all_runs:
            p = r["profile"]
            print(f"  {r['seed']:<8} {r['macro_mAP50_95']:>10.4f} {r['minority_recall']:>12.4f} "
                  f"{str(p.get('GFLOPs')):>8} {str(p.get('gpu_fps')):>10} {str(r['fuse_ok']):>6}")
        if len(all_runs) > 1:
            import statistics
            m = [r['macro_mAP50_95'] for r in all_runs]
            print(f"\n  mAP50-95 mean +/- std: {statistics.mean(m):.4f} "
                  f"+/- {statistics.stdev(m):.4f}")
        print(f"\n  COMPARE against baseline B0:")
        print(f"    B0 mAP50-95 = 0.9696 +/- 0.0026 | GFLOPs = 6.32 | Jetson FPS = 9.79")
        print(f"    Target: mAP within noise, GFLOPs <= baseline, Jetson FPS >= 15")
        print(f"\n  NEXT: export best.pt -> ONNX -> TensorRT FP16 -> Jetson benchmark (real FPS)")


if __name__ == "__main__":
    main()
