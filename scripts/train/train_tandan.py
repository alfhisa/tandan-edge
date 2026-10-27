"""
train_tandan.py — YOLO11n runs R1–R6 (architecture screening + resolution study)
================================================================================
(originally `tandan-yolo/scripts/train_tandan.py`)

One script for every YOLO11n run built from a model YAML; COCO-pretrained
yolo11n.pt weights are transferred where layer shapes match.

Exact commands used for the paper (from the repository root):

  # R1-R3: architecture screening, 640 px, seed 42
  python scripts/train/train_tandan.py --config model_configs/tandan-r1-scshead.yaml --name R1_scshead
  python scripts/train/train_tandan.py --config model_configs/tandan-r2-simam.yaml   --name R2_simam
  python scripts/train/train_tandan.py --config model_configs/tandan-r3-bifpn.yaml   --name R3_bifpn

  # R4 / R5: resolution study, plain YOLO11n, 3 seeds
  #   R4_res768_seed42 is the deployed model
  python scripts/train/train_tandan.py --config model_configs/yolo11n-base.yaml --name R4_res768 --imgsz 768 --seeds 42 123 456
  python scripts/train/train_tandan.py --config model_configs/yolo11n-base.yaml --name R5_res704 --imgsz 704 --seeds 42 123 456

  # R6: optimiser check at 768 px
  python scripts/train/train_tandan.py --config model_configs/yolo11n-base.yaml --name R6_adamw768 --imgsz 768 --optimizer AdamW --lr0 0.001

  # smoke test
  python scripts/train/train_tandan.py --config model_configs/yolo11n-base.yaml --name smoke --epochs 1

Output: runs/<name>_seed<seed>/ (+ <run>_metrics.json and best.onnx)
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

CLASS_NAMES      = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
MINORITY_CLASSES = ["dead", "mismanaged", "yellowish"]

BASE_TRAIN_ARGS = {
    "data": str(DATASET_YAML), "epochs": 150, "imgsz": 640, "batch": 16,
    "workers": 8, "device": 0, "exist_ok": False, "verbose": True,
    "patience": 50, "save": True, "save_period": 10, "val": True,
    "plots": True, "cache": False, "amp": True,
}
PROFILE_IMGSZ = 640
BASELINE_REF = {"mAP50_95": 0.9619, "std": 0.0023, "gflops": 6.32,
                "params_M": 2.583, "gpu_fps": 153.4}


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


def profile_model(mb, imgsz=PROFILE_IMGSZ):
    prof = {}
    try:
        prof["params_M"] = round(sum(p.numel() for p in mb.model.parameters())/1e6, 3)
    except Exception:
        prof["params_M"] = None
    try:
        from thop import profile as tp
        d = torch.zeros(1, 3, imgsz, imgsz).to(next(mb.model.parameters()).device)
        macs, _ = tp(mb.model, inputs=(d,), verbose=False)
        prof["GFLOPs"] = round(macs*2/1e9, 2)
    except Exception:
        prof["GFLOPs"] = None
    try:
        mb.model.eval()
        d = torch.zeros(1, 3, imgsz, imgsz).to(next(mb.model.parameters()).device)
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


def run_single(seed, epochs, config, name, imgsz=640, optimizer='auto', lr0=None):
    from ultralytics import YOLO
    from register_tandan import register_all
    register_all(verbose=True)

    run_name = f"{name}_seed{seed}"
    run_dir = RUNS_DIR / run_name
    print(f"\n{'#'*72}\n  TANDAN-YOLO | {run_name}\n  config={Path(config).name} seed={seed} epochs={epochs} imgsz={imgsz}\n  optimizer={optimizer} lr0={lr0 if lr0 else 'default'}\n{'#'*72}\n")

    model = YOLO(str(config))
    if PRETRAINED.exists():
        model.load(str(PRETRAINED))
        print(f"  [pretrained] transfer dari {PRETRAINED.name} (partial match; "
              f"head baru train dari awal)")

    args = {**BASE_TRAIN_ARGS, "epochs": epochs, "seed": seed, "imgsz": imgsz}
    if optimizer and optimizer != "auto":
        args["optimizer"] = optimizer
    if lr0 is not None:
        args["lr0"] = lr0
    model.train(**args, project=str(RUNS_DIR), name=run_name)

    best = run_dir / "weights" / "best.pt"
    mb = YOLO(str(best))
    vr = mb.val(data=str(DATASET_YAML), imgsz=imgsz, batch=16, device=0,
                verbose=True, plots=True, save_json=True)
    rows, m50, m5095, mr, ma = extract_metrics(vr)
    bgfp = extract_background_fp(vr, CLASS_NAMES)
    prof = profile_model(mb, imgsz)

    export_ok = False
    try:
        mb2 = YOLO(str(best))
        mb2.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
        export_ok = True
        print("  [export] ONNX opset 12 OK")
    except Exception as e:
        print(f"  [export] FAILED: {e}")

    out = {"experiment": name, "run_name": run_name, "seed": seed,
           "epochs": epochs, "imgsz": imgsz, "config": Path(config).name,
           "optimizer": optimizer, "lr0": lr0, "dataset": "site-2-only",
           "macro_mAP50": round(m50, 4), "macro_mAP50_95": round(m5095, 4),
           "minority_recall": round(mr, 4), "minority_AP50": round(ma, 4),
           "export_onnx_ok": export_ok, "background_fp": bgfp,
           "profile": prof, "per_class": rows}
    with open(run_dir / f"{run_name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  [OK] {run_name}: mAP50-95={m5095:.4f} min_recall={mr:.4f} "
          f"params={prof.get('params_M')}M GFLOPs={prof.get('GFLOPs')} "
          f"GPU_FPS={prof.get('gpu_fps')} export={export_ok}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--name", type=str, required=True,
                    help="nama eksperimen, mis. R1_scshead")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=BASE_TRAIN_ARGS["epochs"])
    ap.add_argument("--imgsz", type=int, default=640,
                    help="resolusi input (640 default; 704/768/896 untuk studi resolusi)")
    ap.add_argument("--optimizer", type=str, default="auto",
                    choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
                    help="'auto' = default Ultralytics; 'AdamW' utk replikasi setelan Thapa 2025")
    ap.add_argument("--lr0", type=float, default=None,
                    help="initial learning rate (mis. 0.001 utk AdamW). None = default Ultralytics")
    args = ap.parse_args()

    print_section(f"TANDAN-YOLO — {args.name}")
    print(f"  Config: {args.config}")
    print(f"  Seeds : {args.seeds} | Epochs: {args.epochs} | imgsz: {args.imgsz}")
    print(f"  Optimizer: {args.optimizer} | lr0: {args.lr0 if args.lr0 else 'default'}")
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = _ROOT / cfg
    if not cfg.exists():
        print(f"ERROR: config tidak ditemukan: {cfg}"); return
    if not DATASET_YAML.exists() or not PRETRAINED.exists():
        print("ERROR: dataset/pretrained missing"); return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for s in args.seeds:
        try:
            runs.append(run_single(s, args.epochs, str(cfg), args.name, args.imgsz,
                                   args.optimizer, args.lr0))
        except Exception as e:
            print(f"  [FAIL] seed{s}: {e}")
            import traceback; traceback.print_exc()

    if runs:
        print_section(f"SUMMARY — {args.name}")
        print(f"  {'Seed':<7}{'mAP50-95':>10}{'min_rec':>9}{'params':>8}"
              f"{'GFLOPs':>8}{'GPU_FPS':>9}{'export':>7}")
        for r in runs:
            p = r["profile"]
            print(f"  {r['seed']:<7}{r['macro_mAP50_95']:>10.4f}"
                  f"{r['minority_recall']:>9.4f}{str(p.get('params_M')):>8}"
                  f"{str(p.get('GFLOPs')):>8}{str(p.get('gpu_fps')):>9}"
                  f"{str(r['export_onnx_ok']):>7}")
        if len(runs) > 1:
            import statistics
            m = [r["macro_mAP50_95"] for r in runs]
            print(f"\n  mAP50-95 mean±std: {statistics.mean(m):.4f} ± {statistics.stdev(m):.4f}")
        b = BASELINE_REF
        print(f"\n  BASELINE T0: mAP {b['mAP50_95']} ± {b['std']} | "
              f"{b['params_M']}M | {b['gflops']} GFLOPs | GPU {b['gpu_fps']} FPS")
        print("  Gate R1: mAP drop <= 0.005 DAN GFLOPs turun -> LANJUT")


if __name__ == "__main__":
    main()