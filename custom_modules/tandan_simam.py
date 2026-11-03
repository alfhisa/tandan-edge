"""
tandan_simam.py — SimAM and C3k2SimAM (R2).

Parameter-free 3D attention (SimAM) added after each bottleneck unit of the
C3k2 blocks in the backbone. Uses only element-wise ops, so it adds no
parameters and stays compatible with ONNX opset 12 and TensorRT 8.2.

Reference: Yang et al., SimAM, ICML 2021. Python 3.8 compatible.
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
    Parameter-free 3D attention (Yang et al., ICML 2021).

    Args:
        e_lambda: stabiliser in the denominator (default 1e-4, as in the SimAM paper)
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
        # v = sigma^2 (as in the official implementation)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.act(e_inv)


class C3k2SimAM(C3k2):
    """
    C3k2 with SimAM after each bottleneck unit.

    Same signature as C3k2 (c1, c2, n, c3k, e, attn, g, shortcut), so it is a
    drop-in replacement in model YAMLs.

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
            raise ImportError("Ultralytics block modules are not available.")

        # Wrap each unit with SimAM (no extra parameters)
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
    assert n_params == 0, "SimAM must be parameter-free"

    if Bottleneck is None:
        print("\nUltralytics is not available -- C3k2SimAM test skipped.")
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

        # Compare parameter count with the original C3k2 (must be identical)
        base = C3k2(64, 128, n=2, c3k=False, e=0.5)
        mod = C3k2SimAM(64, 128, n=2, c3k=False, e=0.5)
        pb = sum(p.numel() for p in base.parameters())
        pm = sum(p.numel() for p in mod.parameters())
        print(f"\n  C3k2 params    : {pb:,}")
        print(f"  C3k2SimAM      : {pm:,}  (difference {pm - pb}, must be 0)")
        assert pb == pm, "SimAM must not add parameters"

    print("\nAll tests passed.")
