"""
spd_conv.py — Space-to-Depth convolution (SPD-Conv, A1).

Replaces a strided downsampling conv with a space-to-depth rearrangement
followed by a non-strided conv, so no pixels are discarded. Drop-in for
`Conv(c, k, 2)` in the YOLOv12n backbone.

Reference: Sunkara & Luo, ECML PKDD 2022. Python 3.8 compatible.
"""

from typing import Optional

import torch
import torch.nn as nn

# Ultralytics standard Conv block (autopad-aware). Imported lazily so this
# module can be inspected without ultralytics installed.
try:
    from ultralytics.nn.modules.conv import Conv
except Exception:  # pragma: no cover
    Conv = None


class SpaceToDepth(nn.Module):
    """
    Space-to-Depth operation with configurable downscaling factor.

    Folds spatial information into the channel dimension without losing any
    pixels. For scale=2, a [B, C, H, W] tensor becomes [B, 4C, H/2, W/2].
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        s = self.scale
        # Interleave spatial pixels into channels.
        # Take every s-th pixel with offset, then concat along channels.
        # Result for s=2: concat of 4 sub-sampled tensors -> 4C channels.
        return torch.cat(
            [x[..., i::s, j::s] for i in range(s) for j in range(s)],
            dim=1,
        )


class SPDConv(nn.Module):
    """
    SPD-Conv block: Space-to-Depth followed by a non-strided convolution.

    Drop-in replacement for a strided Conv used for downsampling. The output
    spatial resolution is halved (for scale=2), matching the behavior of a
    stride-2 conv, but no spatial information is discarded.

    YAML usage (Ultralytics parser passes c2 as args[0]):
        - [-1, 1, SPDConv, [256]]        # downsample to 256 channels
        - [-1, 1, SPDConv, [256, 3]]     # explicit kernel size

    Args:
        c1: input channels (auto-filled by Ultralytics parser)
        c2: output channels
        k : kernel size of the post-SPD convolution (default 3)
        scale: space-to-depth downscaling factor (default 2)
    """

    def __init__(self, c1: int, c2: int, k: int = 3, scale: int = 2):
        super().__init__()
        if Conv is None:
            raise ImportError(
                "ultralytics Conv not available. Install ultralytics or run "
                "via the registration script."
            )
        self.scale = scale
        self.spd = SpaceToDepth(scale)
        # After SPD, channel count becomes c1 * scale^2
        c_spd = c1 * (scale ** 2)
        # Non-strided conv (stride=1) to adjust channels to c2
        self.conv = Conv(c_spd, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.spd(x))


# Self-test (runs only when executed directly)
if __name__ == "__main__":
    print("Testing SpaceToDepth...")
    sd = SpaceToDepth(scale=2)
    x = torch.randn(1, 64, 80, 80)
    y = sd(x)
    print(f"  Input : {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}  (expect [1, 256, 40, 40])")
    assert y.shape == (1, 256, 40, 40), "SpaceToDepth shape mismatch"

    # Verify no information loss: total elements preserved
    assert x.numel() == y.numel(), "Element count changed -- info lost!"
    print("  Element count preserved: OK (no spatial info lost)")

    if Conv is not None:
        print("\nTesting SPDConv (full block)...")
        block = SPDConv(c1=64, c2=128, k=3, scale=2)
        out = block(x)
        print(f"  Input : {tuple(x.shape)}")
        print(f"  Output: {tuple(out.shape)}  (expect [1, 128, 40, 40])")
        assert out.shape == (1, 128, 40, 40), "SPDConv shape mismatch"
        n_params = sum(p.numel() for p in block.parameters())
        print(f"  Params: {n_params:,}")
        print("  SPDConv block: OK")
    else:
        print("\nUltralytics not available -- skipping SPDConv block test.")

    print("\nAll tests passed.")
