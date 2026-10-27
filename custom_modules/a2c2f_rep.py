"""
a2c2f_rep.py — A2C2fRep: A2C2f with attention replaced by re-parameterizable conv
==================================================================================
Reference / motivation (BAE efficiency / Journal 1):
  YOLOv12n's A2C2f block uses area-attention (ABlock -> AAttn) which is
  memory-access-bound and SLOW on the Jetson Nano's Maxwell GPU, because
  FlashAttention (which mitigates attention's memory bottleneck) does NOT
  support Maxwell (only Turing/Ampere/Ada/Hopper). See YOLOv12 paper
  (Tian et al., NeurIPS 2025, arXiv:2502.12524) sec. 3.1 and 6.

  A2C2fRep keeps the C2f wrapper (cv1, cv2, split-concat structure) of A2C2f
  but replaces the heavy ABlock (attention) with a RepConv block
  (RepVGG/DBB-style): multi-branch at training time (3x3 + 1x1 + optional
  identity BN) for representational richness, fused into a SINGLE 3x3 conv at
  inference time (zero extra inference cost, fast on Maxwell).

  This is a drop-in replacement for A2C2f in the YAML: same (c1, c2, n, ...)
  signature, same input/output channels, same spatial size.

Design (mirrors A2C2f exactly except the inner block):
  A2C2f:     cv1 -> [ABlock x2] per n -> concat -> cv2   (+ optional gamma residual)
  A2C2fRep:  cv1 -> [RepConv]   per n -> concat -> cv2   (+ optional gamma residual)

RepConv (from ultralytics.nn.modules.conv) already provides fuse_convs()
and get_equivalent_kernel_bias(), so fusing before TensorRT export is handled
by Ultralytics' standard model.fuse().

Python 3.8 compatible.
"""

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.conv import Conv, RepConv
except Exception:  # pragma: no cover
    Conv = None
    RepConv = None


class RepBlock(nn.Module):
    """
    A residual re-parameterizable block used to replace ABlock (attention).

    Structure (training):
        x -> RepConv(dim, dim, 3x3 multi-branch) -> + x   (residual)
        followed by a lightweight MLP (1x1 convs), mirroring ABlock's
        attn + mlp two-stage residual structure but with conv instead of
        attention.

    At inference, each RepConv fuses to a single 3x3 conv (via fuse_convs()).
    """

    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        if RepConv is None:
            raise ImportError("ultralytics RepConv not available.")
        # Re-parameterizable 3x3 conv branch (replaces area-attention)
        self.rep = RepConv(dim, dim, k=3, s=1, p=1, act=True)
        # Lightweight channel-mixing MLP (1x1 convs), mirrors ABlock.mlp
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden, 1),
            Conv(mlp_hidden, dim, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual around the rep conv (replaces x = x + attn(x))
        x = x + self.rep(x)
        # Residual around the MLP (mirrors x = x + mlp(x))
        return x + self.mlp(x)


class A2C2fRep(nn.Module):
    """
    Drop-in replacement for A2C2f: same signature and I/O, but the inner
    attention blocks (ABlock) are replaced with RepBlock (re-parameterizable
    conv). Removes the Maxwell-unfriendly attention while keeping the C2f
    aggregation structure.

    YAML usage (identical arg positions to A2C2f):
        - [-1, 4, A2C2fRep, [512, True, 4]]
        - [-1, 4, A2C2fRep, [1024, True, 1]]
    (the 'a2' and 'area' args are accepted for signature compatibility but
     'a2' is ignored -- there is no attention path; RepBlock is always used.)

    Args mirror A2C2f:
        c1, c2, n, a2 (ignored), area (ignored), residual, mlp_ratio, e, g, shortcut
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,       # accepted for YAML compat; ignored (no attention)
        area: int = 1,         # accepted for YAML compat; ignored
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        if Conv is None:
            raise ImportError("ultralytics Conv not available.")
        c_ = int(c2 * e)  # hidden channels (same as A2C2f)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        # Residual scaling gamma (kept for parity with A2C2f residual mode)
        self.gamma = (
            nn.Parameter(0.01 * torch.ones(c2), requires_grad=True)
            if residual else None
        )

        # Replace ABlock stack with RepBlock stack.
        # A2C2f used 2 ABlocks per n; we mirror with 2 RepBlocks per n.
        self.m = nn.ModuleList(
            nn.Sequential(*(RepBlock(c_, mlp_ratio) for _ in range(2)))
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


# Self-test
if __name__ == "__main__":
    if Conv is None:
        print("ultralytics not available -- run on server.")
    else:
        print("Testing A2C2fRep...")
        # Mirror A2C2f example: c1=c2=512, n=4
        m = A2C2fRep(512, 512, n=4, a2=True, area=4)
        x = torch.randn(1, 512, 32, 32)
        y = m(x)
        print(f"  Input : {tuple(x.shape)}")
        print(f"  Output: {tuple(y.shape)}  (expect [1, 512, 32, 32])")
        assert y.shape == (1, 512, 32, 32), "shape mismatch"

        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Params: {n_params:,}")

        # Gradient check
        y.sum().backward()
        print("  Gradient flow: OK")

        # Test fuse path (RepConv.fuse_convs exists)
        from ultralytics.nn.modules.conv import RepConv
        rep = None
        for mod in m.modules():
            if isinstance(mod, RepConv):
                rep = mod
                break
        if rep is not None and hasattr(rep, "fuse_convs"):
            print("  RepConv.fuse_convs available: OK (fuses at model.fuse())")

        print("\nAll tests passed.")
