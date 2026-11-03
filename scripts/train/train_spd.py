"""
train_spd.py — A1: YOLO12n with SPD-Conv downsampling (3 seeds, 640 px)
=======================================================================
(originally `modifications/scripts/train_ablation.py`)

Trains modified YOLO12n architectures with the same settings as the base
models. Each run folder gets a config snapshot, metrics JSON and profiling.
Only A1 was run for the reported results; A2/A3 stay commented out in EXPERIMENTS.

Usage (from the repository root):
  python scripts/train/train_spd.py --exp A1                       # 3 seeds
  python scripts/train/train_spd.py --exp A1 --seeds 42 --epochs 1 # smoke test

Output: runs/A1_spd_seed{42,123,456}/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# --- Path setup: make custom_modules + register importable ---
_THIS_DIR    = Path(__file__).resolve().parent
_MODIF_ROOT  = _THIS_DIR.parents[1]  # repository root
_CONFIG_DIR  = _MODIF_ROOT / "model_configs"
if str(_MODIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODIF_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_RUNS_DIR    = tandan_paths.RUNS_DIR

# ============================================================
# CONFIG
# ============================================================
DATASET_YAML = tandan_paths.DATASET_YAML

# COCO-pretrained YOLOv12n weights — transferred to modified architectures so
# that all layers EXCEPT the new custom modules (SPDConv, etc.) start from the
# same pretrained foundation as the baseline. This ensures fair apple-to-apple
# comparison with the baseline (which used these pretrained weights).
PRETRAINED_WEIGHTS = tandan_paths.pretrained("yolo12n.pt")

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]
SEEDS            = [42, 123, 456]

# Experiment registry: each maps to a model config YAML.
# 'needs_nwd' flags experiments that require the NWD loss patch (A3+).
EXPERIMENTS = {
    "A1": {
        "name"      : "A1_spd",
        "config"    : _CONFIG_DIR / "yolo12n-spd.yaml",
        "desc"      : "YOLOv12n + SPD-Conv (backbone P3,P4,P5)",
        "needs_nwd" : False,
    },
    # "A2": {
    #     "name"   : "A2_spd_bifpn",
    #     "config" : _CONFIG_DIR / "yolo12n-spd-bifpn.yaml",
    #     "desc"   : "A1 + BiFPN neck",
    #     "needs_nwd": False,
    # },
    # "A3": {
    #     "name"   : "A3_spd_bifpn_nwd",
    #     "config" : _CONFIG_DIR / "yolo12n-spd-bifpn.yaml",
    #     "desc"   : "A2 + NWD Loss",
    #     "needs_nwd": True,
    # },
}

# Training args — IDENTICAL to baseline for fair comparison
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

PROFILE_WARMUP = 10
PROFILE_TIMING = 100
PROFILE_IMGSZ  = 640


# ============================================================
# HELPERS
# ============================================================
def print_section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check_environment():
    import ultralytics
    print_section("ENVIRONMENT CHECK")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  Ultralytics   : {ultralytics.__version__}")
    print(f"  CUDA          : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  DATASET_YAML  : {DATASET_YAML}")
    print(f"  YAML exists   : {DATASET_YAML.exists()}")
    if not DATASET_YAML.exists():
        print("  ERROR: dataset yaml not found!")
        return False
    return True


def profile_model(model_best):
    """Profile params, GFLOPs, size, GPU latency."""
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
            for _ in range(PROFILE_WARMUP):
                _ = model_best.model(dummy)
        torch.cuda.synchronize()
        timings = []
        with torch.no_grad():
            for _ in range(PROFILE_TIMING):
                t0 = time.perf_counter()
                _ = model_best.model(dummy)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1000)
        avg = sum(timings) / len(timings)
        profile["gpu_latency_ms"] = round(avg, 2)
        profile["gpu_fps"]        = round(1000 / avg, 1)
    except Exception:
        profile["gpu_fps"] = None

    return profile


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
    return pd.DataFrame(rows), m50, m5095, mr, ma


def run_single(exp_key, exp_cfg, seed, epochs):
    from ultralytics import YOLO

    run_name = f"{exp_cfg['name']}_seed{seed}"
    run_dir  = _RUNS_DIR / run_name

    print(f"\n{'#'*72}")
    print(f"  EXPERIMENT : {exp_key} — {exp_cfg['desc']}")
    print(f"  Run        : {run_name}")
    print(f"  Config     : {exp_cfg['config'].name}")
    print(f"  Seed       : {seed}  |  Epochs: {epochs}")
    print(f"{'#'*72}\n")

    # --- NWD loss patch (A3+) ---
    if exp_cfg.get("needs_nwd"):
        try:
            from custom_modules.nwd_loss import patch_nwd_loss
            patch_nwd_loss()
            print("  [NWD] loss patch applied")
        except Exception as e:
            print(f"  [NWD] WARNING: patch failed: {e}")

    # Build model from custom config
    model = YOLO(str(exp_cfg["config"]))

    # Transfer COCO-pretrained weights to all compatible (shape-matching) layers.
    # New custom modules (SPDConv) without matching pretrained weights are
    # skipped automatically and start from random init. This matches the
    # baseline's pretrained starting point for fair comparison.
    if PRETRAINED_WEIGHTS.exists():
        model.load(str(PRETRAINED_WEIGHTS))
        print(f"  [pretrained] transferred from {PRETRAINED_WEIGHTS.name}")
    else:
        print(f"  [pretrained] WARNING: {PRETRAINED_WEIGHTS} not found — training from scratch!")

    # Train
    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed}
    model.train(**args, project=str(_RUNS_DIR), name=run_name)

    # Validate best
    best_pt = run_dir / "weights" / "best.pt"
    model_best = YOLO(str(best_pt))
    val_results = model_best.val(data=str(DATASET_YAML), imgsz=640, batch=16,
                                 device=0, verbose=True, plots=True, save_json=True)

    df, m50, m5095, mr, ma = extract_metrics(val_results)
    profile = profile_model(model_best)

    # --- Auto-documentation: save everything to run folder ---
    # 1. config snapshot
    import shutil
    try:
        shutil.copy(str(exp_cfg["config"]), str(run_dir / "model_config_snapshot.yaml"))
    except Exception:
        pass

    # 2. metrics JSON
    output = {
        "experiment"     : exp_key,
        "run_name"       : run_name,
        "description"    : exp_cfg["desc"],
        "config_file"    : exp_cfg["config"].name,
        "seed"           : seed,
        "epochs"         : epochs,
        "dataset"        : "site-2-only",
        "macro_mAP50"    : round(m50, 4),
        "macro_mAP50_95" : round(m5095, 4),
        "minority_recall": round(mr, 4),
        "minority_AP50"  : round(ma, 4),
        "profile"        : profile,
        "per_class"      : df.to_dict(orient="records"),
    }
    with open(str(run_dir / f"{run_name}_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  [OK] {run_name}")
    print(f"  mAP50-95 = {m5095:.4f} | min_recall = {mr:.4f} | "
          f"params = {profile.get('params_M')}M | GFLOPs = {profile.get('GFLOPs')} | "
          f"GPU FPS = {profile.get('gpu_fps')}")

    return {"exp": exp_key, "seed": seed, "m50": m50, "m5095": m5095,
            "mr": mr, "ma": ma, "profile": profile, "df": df}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation training (Site 2 only)")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()),
                        help="Experiment key (A1, A2, ...)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds to run (default: 42 123 456)")
    parser.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"],
                        help="Override epochs (use 1 for smoke test)")
    args = parser.parse_args()

    # Register custom modules FIRST
    print_section("REGISTERING CUSTOM MODULES")
    from register_spd import register_all
    register_all(verbose=True)

    if not check_environment():
        return

    exp_cfg = EXPERIMENTS[args.exp]
    if not exp_cfg["config"].exists():
        print(f"\nERROR: config not found: {exp_cfg['config']}")
        return

    print_section(f"ABLATION PLAN — {args.exp}")
    print(f"  Description : {exp_cfg['desc']}")
    print(f"  Config      : {exp_cfg['config']}")
    print(f"  Seeds       : {args.seeds}")
    print(f"  Epochs      : {args.epochs}")
    if args.epochs == 1:
        print(f"  >> SMOKE TEST MODE (1 epoch) <<")

    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for seed in args.seeds:
        try:
            r = run_single(args.exp, exp_cfg, seed, args.epochs)
            all_runs.append(r)
        except Exception as e:
            print(f"\n  [FAIL] {args.exp} seed{seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Quick summary
    if all_runs:
        print_section(f"SUMMARY — {args.exp}")
        print(f"  {'Seed':<8} {'mAP50-95':>10} {'min_recall':>12} {'GPU FPS':>10}")
        print(f"  {'-'*44}")
        for r in all_runs:
            print(f"  {r['seed']:<8} {r['m5095']:>10.4f} {r['mr']:>12.4f} "
                  f"{str(r['profile'].get('gpu_fps')):>10}")
        if len(all_runs) > 1:
            import statistics
            m = [r['m5095'] for r in all_runs]
            print(f"\n  mAP50-95 mean +/- std: {statistics.mean(m):.4f} "
                  f"+/- {statistics.stdev(m):.4f}" if len(m) > 1 else "")
        print(f"\n  Next: run aggregate_ablation.py after all experiments done")


if __name__ == "__main__":
    main()