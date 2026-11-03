"""
register_tandan.py — register the custom modules (R1-R3) with Ultralytics.

Adds TandanDetect (SCS-Head, R1), C3k2SimAM (R2) and BiFPNConcat (R3) to the
Ultralytics model parser without modifying the installed package.

Usage:
    from register_tandan import register_all
    register_all()
    model = YOLO("model_configs/tandan-r2-simam.yaml")

CLI check (build, forward pass and ONNX opset 12 export):
    python scripts/train/register_tandan.py --config model_configs/tandan-r2-simam.yaml

Python 3.8 compatible.
"""

import argparse
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
    from custom_modules.tandan_head import TandanDetect
    from custom_modules.tandan_simam import C3k2SimAM, SimAM

    import ultralytics.nn.modules as _modules
    import ultralytics.nn.tasks as _tasks

    registered = {"TandanDetect": TandanDetect,
                  "C3k2SimAM": C3k2SimAM,
                  "SimAM": SimAM}

    # Optional module (available from R3 onwards)
    try:
        from custom_modules.tandan_bifpn import BiFPNConcat
        registered["BiFPNConcat"] = BiFPNConcat
    except Exception:
        pass

    for name, cls in registered.items():
        setattr(_modules, name, cls)
        setattr(_tasks, name, cls)

    if not _PATCHED:
        _patch_parse(_tasks, verbose=verbose)
        _PATCHED = True
    if verbose:
        print(f"  [register] {', '.join(registered.keys())}")
    return list(registered.keys())


def _patch_parse(_tasks, verbose=True):
    """
    parse_model checks module membership in several internal frozensets
    (base_modules, repeat_modules, head sets). Wrapping them lets each custom
    module join the sets that contain its parent class.
    """
    from custom_modules.tandan_head import TandanDetect
    from custom_modules.tandan_simam import C3k2SimAM
    from ultralytics.nn.modules import C3k2, Concat, Detect

    try:
        from custom_modules.tandan_bifpn import BiFPNConcat
    except Exception:
        BiFPNConcat = None

    original = _tasks.parse_model
    orig_fs = builtins.frozenset

    def smart_fs(iterable=()):
        s = orig_fs(iterable)
        add = set()
        if C3k2 in s and C3k2SimAM not in s:
            add.add(C3k2SimAM)
        if Detect in s and TandanDetect not in s:
            add.add(TandanDetect)
        if BiFPNConcat is not None and Concat in s and BiFPNConcat not in s:
            add.add(BiFPNConcat)
        return orig_fs(set(s) | add) if add else s

    def _rename_bifpn(d):
        """Replace 'BiFPNConcat' with 'Concat' in a YAML dict. Returns True if any was found."""
        found = False
        for sec in ("backbone", "head"):
            for layer in d.get(sec, []):
                # layer = [from, repeats, module_str, args]
                if len(layer) > 2 and layer[2] == "BiFPNConcat":
                    layer[2] = "Concat"
                    found = True
        return found

    def patched(d, ch, verbose=True):
        builtins.frozenset = smart_fs
        # parse_model checks Concat by identity (`m is Concat`), not via a set,
        # so the frozenset trick is not enough. Concat is swapped in the tasks
        # namespace during the call so that BiFPNConcat passes the identity check
        # and c2 = sum(ch[x] for x in f) is computed correctly.
        swapped = False
        orig_concat = None
        if BiFPNConcat is not None and _rename_bifpn(d):
            orig_concat = _tasks.Concat
            _tasks.Concat = BiFPNConcat
            swapped = True
        try:
            return original(d, ch, verbose=verbose)
        finally:
            builtins.frozenset = orig_fs
            if swapped:
                _tasks.Concat = orig_concat

    _tasks.parse_model = patched
    if verbose:
        print("  [register] parse_model patched (custom -> parent sets)")


def gate_check(cfg_path):
    """Build, run a forward pass and export to ONNX opset 12 (check before training)."""
    import torch
    from ultralytics import YOLO

    print(f"\nGate check: {Path(cfg_path).name}")
    model = YOLO(str(cfg_path))
    n = sum(p.numel() for p in model.model.parameters())
    print(f"  Build OK | params: {n/1e6:.3f} M  (baseline YOLO11n = 2.583 M)")
    model.model.info(imgsz=640)

    print("\n  Forward 640...")
    model.model.eval()
    with torch.no_grad():
        _ = model.model(torch.zeros(1, 3, 640, 640))
    print("  Forward OK")

    print("\n  Exporting ONNX opset 12...")
    model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    print("  Export OK -- structurally TensorRT-ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default=str(_ROOT / "model_configs" / "tandan-r2-simam.yaml"))
    args = ap.parse_args()

    print("Registering TANDAN modules...")
    register_all(verbose=True)

    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = _ROOT / cfg
    if not cfg.exists():
        print(f"ERROR: config not found: {cfg}")
        sys.exit(1)

    try:
        gate_check(str(cfg))
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()