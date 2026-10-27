"""
export_base_models_onnx.py — Export the 4 base detectors to ONNX at 640 px
==========================================================================
(originally `exp-journal-1/jetson_deployment/scripts/export_to_onnx.py`)

Exports YOLOv8n, YOLO11n, YOLO12n and YOLO26n (from train_base_models.py) for
the Jetson speed comparison in Table 4 (jetson/benchmark_base_models.sh).
Throughput does not depend on the weights; the seeds below are the ones used.
YOLO26n exports, but TensorRT 8.2 cannot parse its ONNX graph (Mod operator).

Usage (from the repository root):
  python scripts/export/export_base_models_onnx.py
  python scripts/export/export_base_models_onnx.py --models yolo12n

Output: deploy/onnx/{model}_640.onnx + export_manifest_base_models.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(BASE_DIR))
import tandan_paths  # noqa: E402

_RUNS = tandan_paths.RUNS_DIR
_ONNX = BASE_DIR / "deploy" / "onnx"

# ============================================================
# CONFIG
# ============================================================
MODELS = [
    {
        "key"    : "yolov8n",
        "pt"     : _RUNS / "yolov8n_seed123" / "weights" / "best.pt",
        "onnx"   : _ONNX / "yolov8n_640.onnx",
    },
    {
        "key"    : "yolo11n",
        "pt"     : _RUNS / "yolo11n_seed42" / "weights" / "best.pt",
        "onnx"   : _ONNX / "yolo11n_640.onnx",
    },
    {
        "key"    : "yolo12n",
        "pt"     : _RUNS / "yolo12n_seed456" / "weights" / "best.pt",
        "onnx"   : _ONNX / "yolo12n_640.onnx",
    },
    {
        "key"    : "yolo26n",
        "pt"     : _RUNS / "yolo26n_seed42" / "weights" / "best.pt",
        "onnx"   : _ONNX / "yolo26n_640.onnx",
    },
]

EXPORT_CONFIG = {
    "imgsz"   : 640,
    "opset"   : 12,       # TensorRT 8.x on JetPack 4.6 compatible
    "simplify": True,     # onnxsim simplification
    "dynamic" : False,    # fixed batch=1 for Jetson
}


# ============================================================
# HELPERS
# ============================================================
def print_section(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def check_environment():
    print_section("ENVIRONMENT CHECK")
    try:
        import ultralytics
        import torch
        import onnx
        print(f"  Ultralytics : {ultralytics.__version__}")
        print(f"  PyTorch     : {torch.__version__}")
        print(f"  ONNX        : {onnx.__version__}")
        print(f"  CUDA        : {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  Missing dep : {e}")
        print("  Run: pip install onnx onnxsim")
        return False

    # Check onnxsim
    try:
        import onnxsim
        print(f"  onnxsim     : {onnxsim.__version__}")
    except ImportError:
        print("  onnxsim     : NOT FOUND (simplify will be skipped)")
        print("  Install via: pip install onnxsim")

    return True


def export_single(model_cfg, opset, simplify):
    from ultralytics import YOLO

    key  = model_cfg["key"]
    pt   = model_cfg["pt"]
    onnx_path = model_cfg["onnx"]

    print(f"\n  [{key}] Source  : {pt}")
    print(f"  [{key}] Target  : {onnx_path}")

    if not pt.exists():
        print(f"  [{key}] ERROR: .pt not found at {pt}")
        return {"model": key, "status": "failed", "error": "pt not found"}

    if onnx_path.exists():
        print(f"  [{key}] Already exists, skipping. Delete to re-export.")
        size_mb = onnx_path.stat().st_size / 1e6
        return {"model": key, "status": "skipped", "onnx_size_mb": round(size_mb, 2)}

    t0 = time.time()
    try:
        model = YOLO(str(pt))
        exported = model.export(
            format   = "onnx",
            imgsz    = EXPORT_CONFIG["imgsz"],
            opset    = opset,
            simplify = simplify,
            dynamic  = EXPORT_CONFIG["dynamic"],
        )

        # Ultralytics saves onnx alongside .pt — move to target location
        # Find the exported file
        candidate = pt.with_suffix(".onnx")
        if candidate.exists():
            candidate.rename(onnx_path)
        elif Path(str(exported)).exists():
            Path(str(exported)).rename(onnx_path)
        else:
            # Search in common locations
            for loc in [Path.cwd() / f"{key}_best.onnx",
                        pt.parent / f"{pt.stem}.onnx"]:
                if loc.exists():
                    loc.rename(onnx_path)
                    break

        elapsed = time.time() - t0
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1e6
            print(f"  [{key}] OK — {size_mb:.2f} MB in {elapsed:.1f}s")
            return {
                "model"       : key,
                "status"      : "success",
                "onnx_path"   : str(onnx_path),
                "onnx_size_mb": round(size_mb, 2),
                "export_time_s": round(elapsed, 1),
                "opset"       : opset,
                "imgsz"       : EXPORT_CONFIG["imgsz"],
            }
        else:
            return {"model": key, "status": "failed", "error": "onnx file not found after export"}

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{key}] FAILED: {e}")
        return {"model": key, "status": "failed", "error": str(e)[:200], "time_s": round(elapsed, 1)}


def verify_onnx(onnx_path):
    """Basic ONNX model verification."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        # Print input/output shape
        inputs  = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                   for i in model.graph.input]
        outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                   for o in model.graph.output]
        print(f"    Input  : {inputs}")
        print(f"    Output : {outputs}")
        return True
    except Exception as e:
        print(f"    Verification failed: {e}")
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Export YOLO best.pt to ONNX")
    parser.add_argument("--models", nargs="+", default=[m["key"] for m in MODELS],
                        choices=[m["key"] for m in MODELS])
    parser.add_argument("--opset", type=int, default=EXPORT_CONFIG["opset"])
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    if not check_environment():
        return

    # Ensure output dir exists
    _ONNX.mkdir(parents=True, exist_ok=True)

    models_to_run = [m for m in MODELS if m["key"] in args.models]
    simplify = not args.no_simplify

    print_section(f"EXPORT PLAN — {len(models_to_run)} model(s)")
    print(f"  Opset    : {args.opset}")
    print(f"  Simplify : {simplify}")
    print(f"  Image sz : {EXPORT_CONFIG['imgsz']}")
    print(f"  Dynamic  : {EXPORT_CONFIG['dynamic']}")

    results = []
    for m in models_to_run:
        print_section(f"EXPORTING {m['key'].upper()}")
        r = export_single(m, args.opset, simplify)
        results.append(r)

        if r["status"] == "success":
            print(f"  Verifying ONNX graph...")
            verify_onnx(m["onnx"])

    # Summary
    print_section("EXPORT SUMMARY")
    for r in results:
        status = r["status"]
        size   = r.get("onnx_size_mb", "N/A")
        t      = r.get("export_time_s", "N/A")
        print(f"  {r['model']:<12} {status:<10} size={size} MB  time={t}s")

    # Save manifest (useful for Jetson scripts to know filenames)
    manifest_path = _ONNX / "export_manifest_base_models.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Manifest saved -> {manifest_path}")

    print_section("NEXT STEP — copy to the Jetson")
    print(f"  scp {_ONNX}/*_640.onnx <user>@<jetson>:~/tandan-edge/deploy/onnx/")
    print("\n  Then on the Jetson:")
    print("  bash jetson/benchmark_base_models.sh")


if __name__ == "__main__":
    main()
