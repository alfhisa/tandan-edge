"""
tandan_bifpn.py — BiFPNConcat (R3)
===================================
Komponen K3 dari TANDAN-YOLO: weighted feature fusion di neck.

Concat bawaan menggabungkan cabang dengan bobot implisit SAMA. BiFPN
(Tan et al., EfficientDet CVPR 2020) memberi tiap cabang bobot LEARNABLE
dengan fast normalized fusion:

    w_i' = ReLU(w_i) / (sum_j ReLU(w_j) + eps)
    out  = Concat([w_i' * x_i])

Bobot: 1 skalar per cabang (2 skalar per titik fusi) -> penambahan parameter
praktis nol, tapi model bisa belajar cabang mana (jalur backbone vs jalur
top-down) yang lebih informatif di tiap skala.

Berbeda dari SimAM (R2) yang MENSKALAKAN ULANG aktivasi di tengah backbone
sehingga menggeser distribusi feature pretrained: bobot di sini diinisialisasi
1.0 sehingga saat awal training fusion setara Concat ternormalisasi — gangguan
terhadap bobot pretrained jauh lebih kecil.

Op yang dipakai: relu, sum, div, mul, cat -> semua opset 12 & TensorRT 8.2 safe.

Referensi: Tan et al., "EfficientDet: Scalable and Efficient Object Detection",
CVPR 2020.

Python 3.8 compatible.
"""

import torch
import torch.nn as nn


class BiFPNConcat(nn.Module):
    """
    Weighted concat, drop-in untuk Concat.

    Args:
        dimension: dim concat (1 = channel), sama seperti Concat
        n: jumlah cabang input (YOLO11n neck: semua titik fusi = 2)

    YAML usage (menggantikan Concat):
        - [[-1, 6], 1, BiFPNConcat, [1]]      # n default 2
        - [[-1, 6, 4], 1, BiFPNConcat, [1, 3]]  # eksplisit bila 3 cabang
    """

    def __init__(self, dimension: int = 1, n: int = 2):
        super().__init__()
        self.d = dimension
        self.n = n
        # init 1.0 -> awalnya setara Concat ternormalisasi (minim gangguan)
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, x):
        w = torch.relu(self.w)
        norm = w.sum() + self.eps
        return torch.cat([w[i] / norm * x[i] for i in range(len(x))], self.d)


if __name__ == "__main__":
    print("Testing BiFPNConcat...")
    m = BiFPNConcat(1, 2)
    a = torch.randn(1, 64, 40, 40)
    b = torch.randn(1, 128, 40, 40)
    y = m([a, b])
    print(f"  [{tuple(a.shape)}, {tuple(b.shape)}] -> {tuple(y.shape)} "
          f"(expect (1, 192, 40, 40))")
    assert y.shape == (1, 192, 40, 40)

    n_params = sum(p.numel() for p in m.parameters())
    print(f"  params: {n_params} (expect 2)")

    y.sum().backward()
    print(f"  grad on weights: {m.w.grad is not None}")

    # Cek ekuivalensi awal dengan Concat ternormalisasi
    with torch.no_grad():
        expected = torch.cat([0.5 * a, 0.5 * b], 1)
        out = m([a, b])
    diff = (out - expected).abs().max().item()
    print(f"  max diff vs Concat*0.5 pada init: {diff:.6f} (harus ~0)")

    print("\nAll tests passed.")
