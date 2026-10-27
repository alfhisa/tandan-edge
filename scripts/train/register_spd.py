"""
register_spd.py — Register SPD-Conv (A1) into Ultralytics
==========================================================
(originally `modifications/scripts/register_modules.py`)

Injects custom modules (SPDConv, ...) into the Ultralytics namespace and
patches parse_model so the YAML parser performs correct (c1, c2) channel
injection for custom downsampling modules.

Usage:
    from register_spd import register_all
    register_all()
    from ultralytics import YOLO
    model = YOLO("model_configs/yolo12n-spd.yaml")

Approach (clean, package untouched):
  1. Bind SPDConv into ultralytics.nn.modules and .tasks namespaces so the
     YAML parser can resolve the name "SPDConv".
  2. Patch parse_model: it builds a local `base_modules` frozenset listing the
     modules that receive (c1, c2) channel injection. We temporarily wrap the
     built-in frozenset during parse_model so that any frozenset containing
     Conv also contains SPDConv. This routes SPDConv through the same channel
     logic as Conv (both take c1, c2 as first two args).

Python 3.8 compatible.
"""

import builtins
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MODIF_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_MODIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODIF_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

_PATCHED = False


def register_all(verbose: bool = True):
    """Register custom modules and patch parse_model. Idempotent."""
    global _PATCHED
    registered = []

    from custom_modules.spd_conv import SPDConv, SpaceToDepth

    import ultralytics.nn.modules as _modules
    import ultralytics.nn.tasks as _tasks

    # 1. Bind names so YAML parser resolves them
    _modules.SPDConv = SPDConv
    _modules.SpaceToDepth = SpaceToDepth
    setattr(_tasks, "SPDConv", SPDConv)
    setattr(_tasks, "SpaceToDepth", SpaceToDepth)
    registered.append("SPDConv")

    # 2. Patch parse_model for channel injection (once)
    if not _PATCHED:
        _patch_channel_injection(_tasks, verbose=verbose)
        _PATCHED = True

    if verbose:
        print(f"  [register] Custom modules registered: {', '.join(registered)}")
    return registered


def _patch_channel_injection(_tasks, verbose: bool = True):
    """
    Wrap parse_model so SPDConv is added to the internal base_modules frozenset.

    parse_model constructs `base_modules = frozenset({Conv, ...})` locally.
    By temporarily replacing builtins.frozenset during the parse_model call, we
    intercept the construction of any frozenset that contains Conv and add
    SPDConv to it. This makes SPDConv receive the same (c1, c2) injection.
    """
    from custom_modules.spd_conv import SPDConv
    from ultralytics.nn.modules import Conv

    original_parse_model = _tasks.parse_model
    orig_frozenset = builtins.frozenset

    def smart_frozenset(iterable=()):
        s = orig_frozenset(iterable)
        # base_modules contains Conv-family classes. Add SPDConv to the set
        # that has Conv so it receives (c1, c2) injection like Conv.
        if Conv in s and SPDConv not in s:
            return orig_frozenset(set(s) | {SPDConv})
        return s

    def parse_model_patched(d, ch, verbose=True):
        builtins.frozenset = smart_frozenset
        try:
            return original_parse_model(d, ch, verbose=verbose)
        finally:
            builtins.frozenset = orig_frozenset

    _tasks.parse_model = parse_model_patched
    if verbose:
        print("  [register] parse_model patched (SPDConv channel injection)")


if __name__ == "__main__":
    print("Registering custom modules...")
    reg = register_all(verbose=True)
    print(f"\nRegistered: {reg}")

    print("\nSmoke test: building yolo12n-spd.yaml...")
    try:
        from ultralytics import YOLO
        cfg = _MODIF_ROOT / "model_configs" / "yolo12n-spd.yaml"
        model = YOLO(str(cfg))
        print(f"  Model built OK from {cfg.name}")
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Total parameters: {n_params:,} ({n_params/1e6:.3f} M)")
        print(f"  (baseline YOLOv12n was 2.558 M)")
    except Exception as e:
        print(f"  Model build FAILED: {e}")
        import traceback
        traceback.print_exc()