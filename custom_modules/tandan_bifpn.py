"""
tandan_bifpn.py — BiFPNConcat: weighted feature fusion for the neck (R3).

Replaces Concat with a learnable, fast-normalised weighting of the input
branches (one scalar per branch), as in BiFPN. Weights start at 1.0, so the
module initially behaves like a normalised Concat. Uses only ONNX opset 12 /
TensorRT 8.2 compatible ops.

Reference: Tan et al., EfficientDet, CVPR 2020. Python 3.8 compatible.
"""

import torch
import torch.nn as nn


class BiFPNConcat(nn.Module):
    """
    Weighted concat, drop-in replacement for Concat.

    Args:
        dimension: concat dimension (1 = channels), as in Concat
        n: number of input branches (2 at every fusion point of the YOLO11n neck)

    YAML usage (instead of Concat):
        - [[-1, 6], 1, BiFPNConcat, [1]]        # n defaults to 2
        - [[-1, 6, 4], 1, BiFPNConcat, [1, 3]]  # three branches
    """

    def __init__(self, dimension: int = 1, n: int = 2):
        super().__init__()
        self.d = dimension
        self.n = n
        # initialised to 1.0, i.e. a normalised Concat at the start
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

    # Check that it initially matches a normalised Concat
    with torch.no_grad():
        expected = torch.cat([0.5 * a, 0.5 * b], 1)
        out = m([a, b])
    diff = (out - expected).abs().max().item()
    print(f"  max diff vs Concat*0.5 at init: {diff:.6f} (should be ~0)")

    print("\nAll tests passed.")
