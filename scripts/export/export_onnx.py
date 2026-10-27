"""
export_onnx.py — Export the deployed model to ONNX at the three study resolutions
=================================================================================
(originally `exp-journal-1/final_using_yolo11/scripts/export_final_onnx.py`)

Static input, batch 1, opset 12, onnxsim — the settings that TensorRT 8.2
(JetPack 4.6) accepts. One ONNX file per inference resolution, because the
TensorRT engines are built with a fixed input size.

For the paper the weights were R4_res768_seed42 (trained at 768 px) and the
files were exported at 640, 704 and 768 px and copied to the Jetson as
best_640.onnx, best_704.onnx and best_768.onnx (the names expected by
jetson/benchmark_trt.sh).

Usage (from the repository root):
  python scripts/export/export_onnx.py                       # deployed weights
  python scripts/export/export_onnx.py --weights runs/R4_res768_seed42/weights/best.pt

Output: deploy/onnx/best_{640,704,768}.onnx + export_manifest.json
Python 3.8 compatible.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]  # repository root
sys.path.insert(0, str(_ROOT))
import tandan_paths  # noqa: E402

ONNX_DIR = _ROOT / "deploy" / "onnx"

DEFAULT_WEIGHTS = tandan_paths.FINAL_WEIGHTS

EXPORT_CONFIG = {
    "opset": 12,       # TensorRT 8.x / JetPack 4.6 compatible
    "simplify": True,
    "dynamic": False,  # fixed batch=1 for the Jetson
}
RESOLUTIONS = [640, 704, 768]


def print_section(t):
    print("\n" + "=" * 68 + f"\n  {t}\n" + "=" * 68)


def check_env():
    print_section("ENVIRONMENT CHECK")
    try:
        import ultralytics, torch, onnx
        print(f"  Ultralytics : {ultralytics.__version__}")
        print(f"  PyTorch     : {torch.__version__}")
        print(f"  ONNX        : {onnx.__version__}")
        try:
            import onnxsim
            print(f"  onnxsim     : {onnxsim.__version__}")
        except ImportError:
            print("  onnxsim     : NOT FOUND (simplify diskip). pip install onnxsim")
        return True
    except ImportError as e:
        print(f"  Missing dep : {e}\n  Run: pip install onnx onnxsim")
        return False


def verify_onnx(onnx_path):
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                  for i in m.graph.input]
        outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                   for o in m.graph.output]
        print(f"    Input  : {inputs}")
        print(f"    Output : {outputs}")
        return True
    except Exception as e:
        print(f"    Verify failed: {e}")
        return False


def export_one(weights, res, tag, opset, simplify):
    from ultralytics import YOLO
    onnx_target = ONNX_DIR / f"{tag}_{res}.onnx"

    print(f"\n  [{tag}@{res}] source: {weights}")
    if onnx_target.exists():
        size = onnx_target.stat().st_size / 1e6
        print(f"  [{tag}@{res}] sudah ada ({size:.2f} MB), skip. Hapus untuk re-export.")
        return {"tag": tag, "res": res, "status": "skipped",
                "onnx": str(onnx_target), "size_mb": round(size, 2)}

    t0 = time.time()
    try:
        model = YOLO(str(weights))
        exported = model.export(format="onnx", imgsz=res, opset=opset,
                                simplify=simplify, dynamic=EXPORT_CONFIG["dynamic"])
        # Ultralytics simpan onnx di samping .pt -> pindahkan
        cand = Path(weights).with_suffix(".onnx")
        if cand.exists():
            cand.rename(onnx_target)
        elif Path(str(exported)).exists():
            Path(str(exported)).rename(onnx_target)

        elapsed = time.time() - t0
        if onnx_target.exists():
            size = onnx_target.stat().st_size / 1e6
            print(f"  [{tag}@{res}] OK -- {size:.2f} MB in {elapsed:.1f}s")
            verify_onnx(onnx_target)
            return {"tag": tag, "res": res, "status": "success",
                    "onnx": str(onnx_target), "size_mb": round(size, 2),
                    "export_time_s": round(elapsed, 1), "opset": opset}
        return {"tag": tag, "res": res, "status": "failed", "error": "onnx not found"}
    except Exception as e:
        print(f"  [{tag}@{res}] FAILED: {e}")
        return {"tag": tag, "res": res, "status": "failed", "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--tag", type=str, default="best")
    ap.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    ap.add_argument("--opset", type=int, default=EXPORT_CONFIG["opset"])
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    if not check_env():
        return
    w = Path(args.weights)
    if not w.exists():
        print(f"\nERROR: weights tidak ditemukan: {w}")
        return

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    simplify = not args.no_simplify

    print_section(f"EXPORT PLAN — {args.tag}")
    print(f"  Weights     : {w}")
    print(f"  Resolutions : {args.resolutions}")
    print(f"  Opset       : {args.opset} | Simplify: {simplify} | batch=1 fixed")

    results = []
    for res in args.resolutions:
        print_section(f"EXPORT {args.tag} @ {res}")
        results.append(export_one(w, res, args.tag, args.opset, simplify))

    print_section("EXPORT SUMMARY")
    for r in results:
        print(f"  {r['tag']}@{r['res']:<5} {r['status']:<10} "
              f"size={r.get('size_mb','N/A')} MB")

    manifest = ONNX_DIR / "export_manifest.json"
    with open(manifest, "w") as f:
        json.dump({"weights": str(w), "tag": args.tag, "results": results}, f, indent=2)
    print(f"\n  Manifest -> {manifest}")

    print_section("NEXT — copy to the Jetson")
    print(f"  scp {ONNX_DIR}/best_*.onnx <user>@<jetson>:~/tandan-edge/deploy/onnx/")
    print("  then on the Jetson: bash jetson/benchmark_trt.sh  (see jetson/README.md)")


if __name__ == "__main__":
    main()
