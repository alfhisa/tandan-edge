"""
patch_vfl.py — Inject Varifocal Loss into Ultralytics v8DetectionLoss
======================================================================
Replaces the classification BCE in v8DetectionLoss with Varifocal Loss,
WITHOUT editing the installed ultralytics package.

How it works:
  In v8DetectionLoss.__call__, the classification loss is computed as:
      bce_loss = self.bce(pred_scores, target_scores)   # (bs, anchors, nc)
  self.bce is an nn.BCEWithLogitsLoss(reduction="none") instance.
  VarifocalLoss has the SAME call signature and returns the same shape, so we
  simply swap the instance attribute `self.bce` on every v8DetectionLoss that
  gets constructed. We do this by wrapping v8DetectionLoss.__init__ so that
  after the original init runs, self.bce is replaced with a VarifocalLoss.

  target_scores from the Task-Aligned Assigner already serves as VFL's
  IoU-aware target q (0 for background, in (0,1] for positives), so no other
  change is needed.

Usage:
    from patch_vfl import patch_vfl
    patch_vfl(alpha=0.75, gamma=2.0)   # call BEFORE model.train()
    from ultralytics import YOLO
    model = YOLO("yolo12n.pt")
    model.train(...)

Python 3.8 compatible.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BAE_ROOT = _THIS_DIR.parents[1]  # repository root
if str(_BAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BAE_ROOT))
import tandan_paths  # noqa: E402  (repository paths, see tandan_paths.py)

_VFL_PATCHED = False


def patch_vfl(alpha: float = 0.75, gamma: float = 2.0, verbose: bool = True):
    """
    Patch v8DetectionLoss so its classification loss uses Varifocal Loss.
    Idempotent: safe to call multiple times.
    """
    global _VFL_PATCHED
    if _VFL_PATCHED:
        if verbose:
            print("  [VFL] already patched")
        return

    from custom_modules.varifocal_loss import VarifocalLoss
    import ultralytics.utils.loss as _loss_mod

    DetLoss = _loss_mod.v8DetectionLoss
    original_init = DetLoss.__init__

    def patched_init(self, *args, **kwargs):
        # Run the original constructor (sets up self.bce, assigner, etc.)
        original_init(self, *args, **kwargs)
        # Swap the classification BCE for Varifocal Loss (same call signature)
        device = getattr(self, "device", None)
        vfl = VarifocalLoss(alpha=alpha, gamma=gamma)
        if device is not None:
            vfl = vfl.to(device)
        self.bce = vfl
        # Tag for verification / logging
        self._vfl_active = True
        self._vfl_alpha = alpha
        self._vfl_gamma = gamma

    DetLoss.__init__ = patched_init
    _VFL_PATCHED = True

    if verbose:
        print(f"  [VFL] v8DetectionLoss patched with Varifocal Loss "
              f"(alpha={alpha}, gamma={gamma})")


def verify_vfl(model, verbose: bool = True):
    """
    Verify VFL is active on a built model's loss criterion.
    Call after model is built / training has initialized the criterion.
    """
    try:
        crit = model.model.criterion if hasattr(model.model, "criterion") else None
        if crit is None:
            if verbose:
                print("  [VFL] criterion not yet initialized (build/train first)")
            return None
        active = getattr(crit, "_vfl_active", False)
        bce_type = type(crit.bce).__name__
        if verbose:
            print(f"  [VFL] criterion.bce type: {bce_type} | vfl_active={active}")
        return active
    except Exception as e:
        if verbose:
            print(f"  [VFL] verify failed: {e}")
        return None


if __name__ == "__main__":
    print("Testing VFL patch...")
    patch_vfl(alpha=0.75, gamma=2.0, verbose=True)

    # Build a YOLOv12n model and check the criterion after a forward setup
    print("\nBuilding YOLOv12n and initializing criterion...")
    try:
        from ultralytics import YOLO
        import torch

        model = YOLO("yolo12n.pt")
        # Trigger criterion init by running a tiny forward/loss on dummy data is
        # complex; instead we check that the class-level init patch is in place.
        import ultralytics.utils.loss as _loss_mod
        # Instantiate directly to verify swap (needs a model ref; use the built one)
        det_model = model.model
        # The criterion is lazily created; force it:
        det_model.args = det_model.args if hasattr(det_model, "args") else None
        try:
            crit = det_model.init_criterion()
            bce_type = type(crit.bce).__name__
            active = getattr(crit, "_vfl_active", False)
            print(f"  criterion.bce = {bce_type} | vfl_active = {active}")
            if bce_type == "VarifocalLoss":
                print("  [OK] VFL is active in the criterion")
            else:
                print("  [WARN] VFL not detected -- check patch")
        except Exception as e:
            print(f"  Could not force criterion init: {e}")
            print("  (This is fine -- criterion initializes at train time.)")
    except Exception as e:
        print(f"  Build test skipped/failed: {e}")
        import traceback
        traceback.print_exc()
