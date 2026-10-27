"""
cb_focal_loss.py — Class-Balanced Weighted Focal Loss (Tahap 1, Track A)
=========================================================================
Rebuild + perbaikan dari loss modification paper IEEE (Saputra et al.,
ICITEE 2025) untuk versi jurnal.

Perbaikan berbasis teori dari paper IEEE:
  1. Class weighting: Effective Number of Samples (Cui et al. 2019, CVPR)
     yang BENAR, menggantikan inverse-frequency ekstrem (rasio 389:1 -> 4.8:1).
     Paper IEEE menyebut Cui 2019 tapi formulanya inverse-freq biasa; ini
     memperbaiki konsistensi + stabilitas training.
  2. Weighted Focal Loss (gamma=2.0) dipertahankan (sudah benar), hanya
     source weight-nya diperbaiki ke Cui effective number.
  3. Box loss: TETAP CIoU default YOLO11 (paper IEEE pakai SmoothL1 yang
     merupakan langkah mundur; CIoU scale-invariant & lebih baik).
  4. DFL: TETAP DFL asli YOLO11 (paper IEEE pakai MSE approx; DFL penuh
     lebih baik & tidak berat).

Jadi modifikasi INTI di jurnal = class-balanced weighted focal loss pada
komponen klasifikasi, menggantikan BCE default. Box & DFL dikembalikan ke
default YOLO11 yang optimal.

Effective Number (Cui 2019):
    E_n = (1 - beta^n) / (1 - beta)
    w_i = (1 - beta) / (1 - beta^n_i)      # = 1 / E_n
    dinormalisasi agar sum(w) = num_classes (menjaga skala loss ~1)

Drop-in replacement untuk self.bce (nn.BCEWithLogitsLoss(reduction="none"))
di v8DetectionLoss: signature (pred_logits, target_scores) -> per-element
loss, shape (bs, anchors, nc), reduction="none".

Python 3.8 compatible.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_effective_number_weights(
    class_counts: List[int],
    beta: float = 0.999,
    normalize_to_num_classes: bool = True,
) -> torch.Tensor:
    """
    Cui et al. (2019) class-balanced weights via effective number of samples.

    Args:
        class_counts: list of per-class instance counts (training set)
        beta: reweighting hyperparameter (0.999 recommended; higher = more extreme)
        normalize_to_num_classes: scale weights so they sum to num_classes

    Returns:
        1D tensor of per-class weights.
    """
    counts = torch.tensor(class_counts, dtype=torch.float64)
    eff_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))  # = 1 / eff_num
    if normalize_to_num_classes:
        weights = weights / weights.sum() * len(class_counts)
    return weights.to(torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Weighted Focal Loss, drop-in for BCEWithLogitsLoss(reduction='none').

    For each element:
        p   = sigmoid(pred_logits)
        p_t = p if target>0.5 else (1-p)     [handled via BCE formulation]
        focal = alpha_weight * (1 - p_t)^gamma * BCE(pred, target)
    where alpha_weight is the per-class effective-number weight broadcast over
    the class dimension.

    Target here is the TAL soft target_scores (IoU-aware, 0 for background),
    consistent with how Ultralytics v8DetectionLoss calls self.bce.

    Args:
        class_weights: 1D tensor [nc], per-class effective-number weights
        gamma: focal focusing parameter (default 2.0, from IEEE paper)
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        # Register as buffer so it moves with .to(device)
        self.register_buffer("class_weights", class_weights.view(1, 1, -1))
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        # pred_logits, target_scores: (bs, num_anchors, nc)
        # Per-element BCE (stable, with logits)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target_scores, reduction="none"
        )
        # Focal modulation: (1 - p_t)^gamma
        p = torch.sigmoid(pred_logits)
        p_t = p * target_scores + (1.0 - p) * (1.0 - target_scores)
        focal_mod = torch.pow(1.0 - p_t, self.gamma)

        # Class-balanced weight (broadcast over class dim)
        cw = self.class_weights.to(pred_logits.dtype)

        return cw * focal_mod * bce


# Self-test
if __name__ == "__main__":
    counts = [231, 89958, 431, 26644, 1473]  # MOPAD Site 2 training
    names = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]

    print("Effective Number weights (Cui 2019, beta=0.999):")
    w = compute_effective_number_weights(counts, beta=0.999)
    for name, wi in zip(names, w.tolist()):
        print(f"  {name:12s}: {wi:.4f}")
    print(f"  ratio max/min: {w.max().item()/w.min().item():.1f}")

    print("\nTesting ClassBalancedFocalLoss...")
    torch.manual_seed(0)
    bs, anchors, nc = 2, 100, 5
    pred = torch.randn(bs, anchors, nc, requires_grad=True)
    target = torch.zeros(bs, anchors, nc)
    target[0, 5, 1] = 0.85   # a healthy positive
    target[0, 12, 0] = 0.90  # a dead positive
    target[1, 8, 2] = 0.70   # a mismanaged positive

    loss_fn = ClassBalancedFocalLoss(w, gamma=2.0)
    loss = loss_fn(pred, target)
    print(f"  output shape: {tuple(loss.shape)} (expect (2,100,5))")
    assert loss.shape == (bs, anchors, nc)

    loss.sum().backward()
    assert pred.grad is not None
    print("  gradient flow: OK")

    # Compare vs plain BCE
    bce = nn.BCEWithLogitsLoss(reduction="none")(pred, target)
    print(f"  CB-Focal total: {loss.sum().item():.4f}")
    print(f"  plain BCE total: {bce.sum().item():.4f}")
    print("\nAll tests passed.")
