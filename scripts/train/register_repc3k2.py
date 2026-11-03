"""
register_repc3k2.py — register RepC3k2 with Ultralytics (T2).

Adds RepC3k2 to the parser module sets used for C3k2, so the YAML parser
handles its channels (c1, c2) and repeats (n) the same way, without modifying
the installed Ultralytics package.

Usage:
    from register_repc3k2 import register_all
    register_all()
    from ultralytics import YOLO
    model = YOLO("model_configs/yolo11n-repc3k2.yaml")

Python 3.8 compatible.
"""

import builtins
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

_PATCHED = False


def register_all(verbose: bool = True):
    global _PATCHED
    from custom_modules.repc3k2 import RepC3k2

    import ultralytics.nn.modules as _modules
    import ultralytics.nn.tasks as _tasks

    _modules.RepC3k2 = RepC3k2
    setattr(_tasks, "RepC3k2", RepC3k2)

    if not _PATCHED:
        _patch(_tasks, verbose=verbose)
        _PATCHED = True
    if verbose:
        print("  [register] RepC3k2 registered")
    return ["RepC3k2"]


def _patch(_tasks, verbose=True):
    from custom_modules.repc3k2 import RepC3k2
    from ultralytics.nn.modules import C3k2

    original = _tasks.parse_model
    orig_fs = builtins.frozenset

    def smart_fs(iterable=()):
        s = orig_fs(iterable)
        # base_modules and repeat_modules contain C3k2; add RepC3k2 to both
        if C3k2 in s and RepC3k2 not in s:
            return orig_fs(set(s) | {RepC3k2})
        return s

    def patched(d, ch, verbose=True):
        builtins.frozenset = smart_fs
        try:
            return original(d, ch, verbose=verbose)
        finally:
            builtins.frozenset = orig_fs

    _tasks.parse_model = patched
    if verbose:
        print("  [register] parse_model patched (RepC3k2 -> base+repeat)")


if __name__ == "__main__":
    print("Registering RepC3k2...")
    register_all(verbose=True)

    print("\nSmoke test: building yolo11n-repc3k2.yaml...")
    try:
        from ultralytics import YOLO
        cfg = _ROOT / "model_configs" / "yolo11n-repc3k2.yaml"
        model = YOLO(str(cfg))
        print(f"  Model built OK from {cfg.name}")
        n = sum(p.numel() for p in model.model.parameters())
        print(f"  Params: {n:,} ({n/1e6:.3f} M)  [baseline YOLO11n = 2.583 M]")
        model.model.info(imgsz=640)

        print("\n  Testing model.fuse()...")
        model.model.fuse()
        print("  model.fuse() OK -- RepConv fused, TensorRT-ready")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
