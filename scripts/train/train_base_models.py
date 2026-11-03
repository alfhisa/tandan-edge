"""
train_base_models.py — Base-model screening: 4 detectors x 3 seeds at 640 px
============================================================================

Models : YOLOv8n, YOLO11n, YOLO12n, YOLO26n (COCO-pretrained, Ultralytics)
Seeds  : 42, 123, 456
Dataset: MOPAD Site 2

Output: runs/{model}_seed{seed}/ (standard Ultralytics output + {name}_metrics.json)
        runs/multiseed_raw_results.csv
The yolo11n_seed* runs are also the "train 640" rows of the resolution study.

Usage (from the repository root):
  python scripts/train/train_base_models.py                     # all 12 runs
  python scripts/train/train_base_models.py --models yolo11n    # YOLO11n x 3 seeds
  python scripts/train/train_base_models.py --models yolo12n --seeds 42 123

Time: ~8-10 h on one Quadro RTX 6000 (12 runs x ~40 min).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

# ============================================================
# CONFIG
# ============================================================
DATASET_YAML = tandan_paths.DATASET_YAML
EXP_ROOT = tandan_paths.RUNS_DIR
EXP_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]

MODELS = [
    {"key": "yolov8n", "weights": "yolov8n.pt", "role": "Classical YOLO reference"},
    {"key": "yolo11n", "weights": "yolo11n.pt", "role": "Modern lightweight reference"},
    {"key": "yolo12n", "weights": "yolo12n.pt", "role": "Attention-heavy (PRIMARY BASE CANDIDATE)"},
    {"key": "yolo26n", "weights": "yolo26n.pt", "role": "Native STAL+ProgLoss reference"},
]

SEEDS = [42, 123, 456]

# Training args (kept identical to single-seed run for fair comparison)
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
    # seed is injected per-run
}

# Profiling config
PROFILE_WARMUP_RUNS  = 10
PROFILE_TIMING_RUNS  = 100
PROFILE_IMGSZ        = 640


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
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM          : {vram:.1f} GB")
        print(f"  CUDA version  : {torch.version.cuda}")
    else:
        print("  WARNING: CUDA not available!")
    print(f"\n  DATASET_YAML  : {DATASET_YAML}")
    print(f"  YAML exists   : {DATASET_YAML.exists()}")
    print(f"  EXP_ROOT      : {EXP_ROOT}")
    if not DATASET_YAML.exists():
        print("\n  YAML not found!")
        return False
    return True


def profile_model(model_best, run_name):
    """Profile model: params, GFLOPs, file size, GPU latency."""
    print_section(f"MODEL PROFILING - {run_name}")
    profile = {}

    # 1. File size
    best_pt = model_best.ckpt_path if hasattr(model_best, "ckpt_path") else None
    if best_pt and Path(best_pt).exists():
        size_mb = Path(best_pt).stat().st_size / 1e6
        profile["model_size_mb"] = round(size_mb, 2)
        print(f"  Model file size : {size_mb:.2f} MB")
    else:
        profile["model_size_mb"] = None

    # 2. Parameters
    try:
        n_params = sum(p.numel() for p in model_best.model.parameters())
        profile["params_M"] = round(n_params / 1e6, 3)
        print(f"  Parameters      : {profile['params_M']} M")
    except Exception as e:
        profile["params_M"] = None
        print(f"  Parameters      : N/A ({e})")

    # 3. GFLOPs
    try:
        from thop import profile as thop_profile
        dummy_input = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(
            next(model_best.model.parameters()).device
        )
        macs, _ = thop_profile(model_best.model, inputs=(dummy_input,), verbose=False)
        gflops = macs * 2 / 1e9
        profile["GFLOPs"] = round(gflops, 2)
        print(f"  GFLOPs @640^2   : {gflops:.2f}")
    except Exception as e:
        profile["GFLOPs"] = None
        print(f"  GFLOPs          : N/A ({e})")

    # 4. GPU latency
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_best.model.eval()
        dummy = torch.zeros(1, 3, PROFILE_IMGSZ, PROFILE_IMGSZ).to(device)
        with torch.no_grad():
            for _ in range(PROFILE_WARMUP_RUNS):
                _ = model_best.model(dummy)
        torch.cuda.synchronize()

        timings = []
        with torch.no_grad():
            for _ in range(PROFILE_TIMING_RUNS):
                t0 = time.perf_counter()
                _ = model_best.model(dummy)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1000)
        avg_ms = sum(timings) / len(timings)
        profile["gpu_latency_ms"] = round(avg_ms, 2)
        profile["gpu_fps"]        = round(1000 / avg_ms, 1)
        profile["gpu_device"]     = torch.cuda.get_device_name(0)
        print(f"  GPU latency     : {avg_ms:.2f} ms")
        print(f"  GPU FPS         : {profile['gpu_fps']}")
    except Exception as e:
        profile["gpu_latency_ms"] = None
        profile["gpu_fps"]        = None
        print(f"  GPU speed       : N/A ({e})")

    return profile


def extract_per_class_metrics(results, class_names, minority_classes):
    """Extract per-class metrics from validation results."""
    per_class_ap50   = results.box.ap50
    per_class_ap5095 = results.box.maps
    per_class_p      = results.box.p
    per_class_r      = results.box.r

    rows             = []
    minority_recalls = []
    minority_ap50s   = []

    for i, name in enumerate(class_names):
        ap50   = float(per_class_ap50[i])   if i < len(per_class_ap50)   else 0.0
        ap5095 = float(per_class_ap5095[i]) if i < len(per_class_ap5095) else 0.0
        prec   = float(per_class_p[i])      if i < len(per_class_p)      else 0.0
        rec    = float(per_class_r[i])      if i < len(per_class_r)      else 0.0
        if name in minority_classes:
            minority_recalls.append(rec)
            minority_ap50s.append(ap50)
        rows.append({
            "class"      : name,
            "is_minority": name in minority_classes,
            "AP@.5"      : round(ap50, 4),
            "AP@.5:.95"  : round(ap5095, 4),
            "Precision"  : round(prec, 4),
            "Recall"     : round(rec, 4),
        })

    macro_map50   = float(per_class_ap50.mean())
    macro_map5095 = float(per_class_ap5095.mean())
    avg_mr        = sum(minority_recalls) / len(minority_recalls) if minority_recalls else 0.0
    avg_ma        = sum(minority_ap50s)   / len(minority_ap50s)   if minority_ap50s   else 0.0

    return pd.DataFrame(rows), macro_map50, macro_map5095, avg_mr, avg_ma


def run_single(model_cfg, seed):
    """Run a single (model x seed) experiment."""
    from ultralytics import YOLO

    run_name = f"{model_cfg['key']}_seed{seed}"
    run_dir  = EXP_ROOT / run_name

    print(f"\n{'#'*72}")
    print(f"  RUN       : {run_name}")
    print(f"  Model     : {model_cfg['weights']} ({model_cfg['role']})")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {run_dir}")
    print(f"{'#'*72}\n")

    # Train
    model = YOLO(str(tandan_paths.pretrained(model_cfg["weights"])))
    train_args = {**BASE_TRAIN_ARGS, "seed": seed}
    model.train(
        **train_args,
        project = str(EXP_ROOT),
        name    = run_name,
    )

    # Validate best weights
    best_pt = run_dir / "weights" / "best.pt"
    print(f"\n  Validating: {best_pt}")
    model_best  = YOLO(str(best_pt))
    val_results = model_best.val(
        data      = str(DATASET_YAML),
        imgsz     = 640,
        batch     = 16,
        device    = 0,
        verbose   = True,
        plots     = True,
        save_json = True,
    )

    # Extract metrics
    df, m50, m5095, mr, ma = extract_per_class_metrics(
        val_results, CLASS_NAMES, MINORITY_CLASSES
    )

    # Profile
    profile = profile_model(model_best, run_name)

    # Save metrics JSON
    output = {
        "run_name"       : run_name,
        "model"          : model_cfg["key"],
        "seed"           : seed,
        "dataset"        : "site-2-only",
        "macro_mAP50"    : round(m50, 4),
        "macro_mAP50_95" : round(m5095, 4),
        "minority_recall": round(mr, 4),
        "minority_AP50"  : round(ma, 4),
        "profile"        : profile,
        "per_class"      : df.to_dict(orient="records"),
    }
    out_path = run_dir / f"{run_name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Metrics saved -> {out_path}")

    return {
        "model"  : model_cfg["key"],
        "seed"   : seed,
        "m50"    : m50,
        "m5095"  : m5095,
        "mr"     : mr,
        "ma"     : ma,
        "profile": profile,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-seed baseline training (Site 2 only)")
    parser.add_argument(
        "--models", nargs="+", type=str,
        default=[m["key"] for m in MODELS],
        choices=[m["key"] for m in MODELS],
        help="Models to run (default: all 4)"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=SEEDS,
        help="Seeds to run (default: 42 123 456)"
    )
    args = parser.parse_args()

    if not check_environment():
        return

    models_to_run = [m for m in MODELS if m["key"] in args.models]
    seeds_to_run  = args.seeds

    total_runs = len(models_to_run) * len(seeds_to_run)

    print_section(f"MULTI-SEED PLAN - {total_runs} total runs")
    print(f"  Models : {[m['key'] for m in models_to_run]}")
    print(f"  Seeds  : {seeds_to_run}")
    print(f"  Epochs : {BASE_TRAIN_ARGS['epochs']}")
    print(f"  Output : {EXP_ROOT}")
    print(f"\n  Estimated time: ~{total_runs * 40} minutes ({total_runs * 40 / 60:.1f} hours)")

    all_runs = []
    failed   = []
    run_idx  = 0

    for model_cfg in models_to_run:
        for seed in seeds_to_run:
            run_idx += 1
            print(f"\n{'='*72}")
            print(f"  [{run_idx}/{total_runs}] {model_cfg['key']} seed={seed}")
            print(f"{'='*72}")
            try:
                result = run_single(model_cfg, seed)
                all_runs.append(result)
                print(f"\n  [OK] {model_cfg['key']}_seed{seed} DONE")
            except Exception as e:
                print(f"\n  [FAIL] {model_cfg['key']}_seed{seed} FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed.append(f"{model_cfg['key']}_seed{seed}")
                continue

    # Final summary (raw, not aggregated - aggregation in separate script)
    print_section(f"ALL RUNS COMPLETE - {len(all_runs)}/{total_runs} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\n  Per-run summary:")
    print(f"  {'Run':<30} {'mAP50':>8} {'mAP50-95':>10} {'Min Rec':>9}")
    print(f"  {'-'*60}")
    for r in all_runs:
        run_label = f"{r['model']}_seed{r['seed']}"
        print(f"  {run_label:<30} {r['m50']:>8.4f} {r['m5095']:>10.4f} {r['mr']:>9.4f}")

    # Save raw results (per-run) as CSV
    raw_rows = []
    for r in all_runs:
        p = r["profile"]
        raw_rows.append({
            "model"         : r["model"],
            "seed"          : r["seed"],
            "macro_mAP50"   : round(r["m50"], 4),
            "macro_mAP50_95": round(r["m5095"], 4),
            "minority_AP50" : round(r["ma"], 4),
            "minority_Recall": round(r["mr"], 4),
            "params_M"      : p.get("params_M"),
            "GFLOPs"        : p.get("GFLOPs"),
            "model_size_mb" : p.get("model_size_mb"),
            "gpu_fps"       : p.get("gpu_fps"),
            "gpu_latency_ms": p.get("gpu_latency_ms"),
        })
    df_raw = pd.DataFrame(raw_rows)
    raw_csv = EXP_ROOT / "multiseed_raw_results.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"\n  Raw results saved -> {raw_csv}")
    print(f"\n  Next step: run aggregation script to compute mean +/- std")
    print(f"    python 03_aggregate_multiseed.py")


if __name__ == "__main__":
    main()
