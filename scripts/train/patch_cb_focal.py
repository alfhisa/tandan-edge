"""
patch_cb_focal.py — use the class-balanced focal loss in v8DetectionLoss (T1).

Wraps v8DetectionLoss.__init__ so that self.bce is replaced by
ClassBalancedFocalLoss (same signature and output shape) without modifying
the installed Ultralytics package. Box (CIoU) and DFL losses are unchanged.

Usage:
    from patch_cb_focal import patch_cb_focal
    patch_cb_focal(class_counts=[231,89958,431,26644,1473], beta=0.999, gamma=2.0)
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(...)

Python 3.8 compatible.
"""

import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

_PATCHED = False


def patch_cb_focal(class_counts, beta: float = 0.999, gamma: float = 2.0,
                   verbose: bool = True):
    """
    Patch v8DetectionLoss.__init__ so self.bce becomes ClassBalancedFocalLoss.
    Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        if verbose:
            print("  [CB-Focal] already patched")
        return

    from custom_modules.cb_focal_loss import (
        ClassBalancedFocalLoss, compute_effective_number_weights,
    )
    import ultralytics.utils.loss as _loss_mod

    weights = compute_effective_number_weights(class_counts, beta=beta)

    DetLoss = _loss_mod.v8DetectionLoss
    original_init = DetLoss.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        device = getattr(self, "device", None)
        cb = ClassBalancedFocalLoss(weights.clone(), gamma=gamma)
        if device is not None:
            cb = cb.to(device)
        self.bce = cb
        self._cb_focal_active = True
        self._cb_beta = beta
        self._cb_gamma = gamma

    DetLoss.__init__ = patched_init
    _PATCHED = True

    if verbose:
        print(f"  [CB-Focal] v8DetectionLoss patched "
              f"(Cui beta={beta}, focal gamma={gamma})")
        print(f"  [CB-Focal] class weights: {[round(w,4) for w in weights.tolist()]}")


def verify_cb_focal(model, verbose=True):
    try:
        crit = model.model.criterion if hasattr(model.model, "criterion") else None
        if crit is None:
            if verbose:
                print("  [CB-Focal] criterion not initialized yet")
            return None
        active = getattr(crit, "_cb_focal_active", False)
        bce_type = type(crit.bce).__name__
        if verbose:
            print(f"  [CB-Focal] criterion.bce = {bce_type} | active = {active}")
        return active
    except Exception as e:
        if verbose:
            print(f"  [CB-Focal] verify failed: {e}")
        return None


if __name__ == "__main__":
    counts = [231, 89958, 431, 26644, 1473]
    print("Testing CB-Focal patch...")
    patch_cb_focal(counts, beta=0.999, gamma=2.0, verbose=True)

    print("\nForcing criterion init on YOLO11n...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        crit = model.model.init_criterion()
        bce_type = type(crit.bce).__name__
        active = getattr(crit, "_cb_focal_active", False)
        print(f"  criterion.bce = {bce_type} | active = {active}")
        if bce_type == "ClassBalancedFocalLoss":
            print("  [OK] CB-Focal active")
        else:
            print("  [WARN] not detected")
    except Exception as e:
        print(f"  (criterion inits at train time) note: {e}")
