"""
register_rep.py — Register A2C2fRep into Ultralytics
=====================================================
Injects A2C2fRep into the Ultralytics namespace and patches parse_model so the
YAML parser gives it correct (c1, c2) channel injection AND treats it as a
'repeat' module (n repeats), exactly like A2C2f.

Usage:
    from register_rep import register_all
    register_all()
    from ultralytics import YOLO
    model = YOLO("model_configs/yolo12n-rep-e1.yaml")

Approach (package untouched):
  A2C2f is in Ultralytics' parse_model base_modules AND repeat_modules
  frozensets. A2C2fRep has the same signature, so we intercept frozenset
  construction during parse_model: any frozenset containing A2C2f also gets
  A2C2fRep. This routes A2C2fRep through identical channel + repeat handling.

Python 3.8 compatible.
"""

import builtins
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EFF_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_EFF_ROOT) not in sys.path:
    sys.path.insert(0, str(_EFF_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

_PATCHED = False


def register_all(verbose: bool = True):
    """Register A2C2fRep + patch parse_model. Idempotent."""
    global _PATCHED
    from custom_modules.a2c2f_rep import A2C2fRep, RepBlock

    import ultralytics.nn.modules as _modules
    import ultralytics.nn.tasks as _tasks

    _modules.A2C2fRep = A2C2fRep
    _modules.RepBlock = RepBlock
    setattr(_tasks, "A2C2fRep", A2C2fRep)
    setattr(_tasks, "RepBlock", RepBlock)

    if not _PATCHED:
        _patch_parse_model(_tasks, verbose=verbose)
        _PATCHED = True

    if verbose:
        print("  [register] A2C2fRep registered")
    return ["A2C2fRep"]


def _patch_parse_model(_tasks, verbose: bool = True):
    """Intercept frozenset so A2C2fRep joins A2C2f in base_modules & repeat_modules."""
    from custom_modules.a2c2f_rep import A2C2fRep
    from ultralytics.nn.modules import A2C2f

    original_parse_model = _tasks.parse_model
    orig_frozenset = builtins.frozenset

    def smart_frozenset(iterable=()):
        s = orig_frozenset(iterable)
        # Both base_modules and repeat_modules contain A2C2f; add A2C2fRep to both.
        if A2C2f in s and A2C2fRep not in s:
            return orig_frozenset(set(s) | {A2C2fRep})
        return s

    def parse_model_patched(d, ch, verbose=True):
        builtins.frozenset = smart_frozenset
        try:
            return original_parse_model(d, ch, verbose=verbose)
        finally:
            builtins.frozenset = orig_frozenset

    _tasks.parse_model = parse_model_patched
    if verbose:
        print("  [register] parse_model patched (A2C2fRep -> base+repeat modules)")


if __name__ == "__main__":
    print("Registering A2C2fRep...")
    reg = register_all(verbose=True)
    print(f"\nRegistered: {reg}")

    print("\nSmoke test: building yolo12n-rep-e1.yaml...")
    try:
        from ultralytics import YOLO
        cfg = _EFF_ROOT / "model_configs" / "yolo12n-rep-e1.yaml"
        model = YOLO(str(cfg))
        print(f"  Model built OK from {cfg.name}")
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Total parameters: {n_params:,} ({n_params/1e6:.3f} M)")
        print(f"  (baseline YOLOv12n was 2.558 M)")

        # Report GFLOPs if possible
        try:
            model.model.info(imgsz=640)
        except Exception:
            pass

        # Test that fuse works (critical for TensorRT export)
        print("\n  Testing model.fuse() (RepConv fusion for deployment)...")
        model.model.fuse()
        print("  model.fuse() OK -- RepConv blocks fused to single conv")
    except Exception as e:
        print(f"  Model build FAILED: {e}")
        import traceback
        traceback.print_exc()
