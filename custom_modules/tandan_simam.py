"""
tandan_simam.py — SimAM + C3k2SimAM (R2)
=========================================
Komponen K1 dari TANDAN-YOLO: attention PARAMETER-FREE di dalam bottleneck
C3k2 backbone.

SimAM (Yang et al., ICML 2021) menurunkan bobot attention 3D dari energy
function berbasis "spatial inhibition" neuroscience — TANPA parameter
belajar sama sekali:

    e_t* = 4(sigma^2 + lambda) / ((t - mu)^2 + 2 sigma^2 + 2 lambda)
    X~   = sigmoid(1 / e_t*) . X

Implementasi memakai mean/pow/sum/div/sigmoid/mul saja -> semua op standar,
aman untuk ONNX opset 12 dan TensorRT 8.2 di Maxwell (tidak ada matmul QKV,
tidak ada softmax spatial, tidak ada fragmentasi channel).

Kontras dengan A2C2f YOLO12n yang gugur di Maxwell: SimAM tidak menambah
parameter maupun memory-bound matmul, hanya elementwise pass.

Penempatan mengikuti pola attn=True bawaan C3k2 (Bottleneck -> modul
attention), tapi PSABlock diganti SimAM yang jauh lebih murah.

Referensi: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
for Convolutional Neural Networks", ICML 2021. Adopsi di YOLO11: C3k2SA
(MDPI Electronics 2025), EDGE-MSE-YOLOv11 (Frontiers Plant Sci 2025).

Python 3.8 compatible.
"""

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.block import C3k2, C3k, Bottleneck
except Exception:  # pragma: no cover
    C3k2 = object
    C3k = None
    Bottleneck = None


class SimAM(nn.Module):
    """
    Parameter-free 3D attention (ICML 2021).

    Args:
        e_lambda: stabilizer pada denominator (default 1e-4 sesuai paper)
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = w * h - 1
        # (t - mu)^2 per channel
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # v = sigma^2 (unbiased-ish, sesuai implementasi resmi)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.act(e_inv)


class C3k2SimAM(C3k2):
    """
    C3k2 dengan SimAM di ujung tiap unit bottleneck.

    Signature identik C3k2 (c1, c2, n, c3k, e, attn, g, shortcut) agar
    drop-in di YAML dan kompatibel dengan parse_model.

    YAML usage:
        - [-1, 2, C3k2SimAM, [256, False, 0.25]]
        - [-1, 2, C3k2SimAM, [512, True]]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if Bottleneck is None:
            raise ImportError("Ultralytics block modules tidak tersedia.")

        # Bungkus tiap unit dengan SimAM (0 parameter tambahan)
        if not attn:
            self.m = nn.ModuleList(
                nn.Sequential(
                    C3k(self.c, self.c, 2, shortcut, g) if c3k
                    else Bottleneck(self.c, self.c, shortcut, g),
                    SimAM(),
                )
                for _ in range(n)
            )


# Self-test
if __name__ == "__main__":
    print("Testing SimAM...")
    sa = SimAM()
    x = torch.randn(2, 32, 20, 20)
    y = sa(x)
    assert y.shape == x.shape
    n_params = sum(p.numel() for p in sa.parameters())
    print(f"  output shape: {tuple(y.shape)} | params: {n_params} (expect 0)")
    assert n_params == 0, "SimAM harus parameter-free"

    if Bottleneck is None:
        print("\nultralytics tidak tersedia -- test C3k2SimAM dilewati.")
    else:
        print("\nTesting C3k2SimAM...")
        for c3k in (False, True):
            m = C3k2SimAM(64, 128, n=2, c3k=c3k, e=0.5)
            xi = torch.randn(1, 64, 40, 40)
            yo = m(xi)
            print(f"  c3k={c3k}: {tuple(xi.shape)} -> {tuple(yo.shape)} | "
                  f"params {sum(p.numel() for p in m.parameters()):,}")
            assert yo.shape == (1, 128, 40, 40)
            yo.sum().backward()

        # Bandingkan params dengan C3k2 asli (harus IDENTIK)
        base = C3k2(64, 128, n=2, c3k=False, e=0.5)
        mod = C3k2SimAM(64, 128, n=2, c3k=False, e=0.5)
        pb = sum(p.numel() for p in base.parameters())
        pm = sum(p.numel() for p in mod.parameters())
        print(f"\n  C3k2 params    : {pb:,}")
        print(f"  C3k2SimAM      : {pm:,}  (selisih {pm - pb} -> harus 0)")
        assert pb == pm, "SimAM tidak boleh menambah parameter"

    print("\nAll tests passed.")
