"""
repc3k2.py — RepC3k2: C3k2 with re-parameterizable bottlenecks (RepConv).

The multi-branch convolutions used in training are fused into a single 3x3
conv for inference, so latency matches C3k2. Same signature as C3k2, so it is
a drop-in replacement in model YAMLs (screening run T2).

Reference: Ding et al., RepVGG, CVPR 2021. Python 3.8 compatible.
"""

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.block import C3k2, C3k, RepBottleneck
except Exception:  # pragma: no cover
    C3k2 = object
    C3k = None
    RepBottleneck = None


class RepC3k2(C3k2):
    """
    Drop-in replacement for C3k2 with RepBottleneck instead of Bottleneck.

    Same signature as C3k2 (c1, c2, n, c3k, e, attn, g, shortcut). With
    c3k=False (the YOLO11n case) the bottlenecks become RepBottleneck;
    with c3k=True the standard C3k blocks are kept.

    YAML usage (same argument positions as C3k2):
        - [-1, 2, RepC3k2, [256, False, 0.25]]
        - [-1, 2, RepC3k2, [512, True]]
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
        # Build the standard C3k2 first (sets cv1, cv2 and the default self.m)
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if RepBottleneck is None:
            raise ImportError("RepBottleneck is not available in this Ultralytics version.")

        # Replace Bottleneck with RepBottleneck on the c3k=False path.
        # The c3k=True path keeps C3k; YOLO11n uses c3k=False.
        if not attn:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else RepBottleneck(self.c, self.c, shortcut, g)
                for _ in range(n)
            )
        # The attn=True path is left as is (not used in the YOLO11n backbone)


# Self-test
if __name__ == "__main__":
    if RepBottleneck is None:
        print("Ultralytics is not available -- run this on the training server.")
    else:
        print("Testing RepC3k2...")
        # Same usage as in YOLO11n: c3k=False
        m = RepC3k2(64, 128, n=2, c3k=False, e=0.5)
        x = torch.randn(1, 64, 80, 80)
        y = m(x)
        print(f"  Input : {tuple(x.shape)}")
        print(f"  Output: {tuple(y.shape)}  (expect [1,128,80,80])")
        assert y.shape == (1, 128, 80, 80), "shape mismatch"

        n_params = sum(p.numel() for p in m.parameters())
        print(f"  Params: {n_params:,}")

        y.sum().backward()
        print("  Gradient flow: OK")

        # Check that RepConv is present and can be fused
        from ultralytics.nn.modules.conv import RepConv
        reps = [mod for mod in m.modules() if isinstance(mod, RepConv)]
        print(f"  RepConv count: {len(reps)}")
        if reps and hasattr(reps[0], "fuse_convs"):
            print("  RepConv.fuse_convs available: OK (fuses at model.fuse())")

        print("\nAll tests passed.")
