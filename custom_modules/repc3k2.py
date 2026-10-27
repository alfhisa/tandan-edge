"""
repc3k2.py — RepC3k2: C3k2 dengan RepBottleneck (free accuracy via reparam)
============================================================================
Tahap 2 (Track B1) — BAE efficiency/free-accuracy untuk YOLO11n.

Konsep:
  YOLO11n SUDAH real-time di Jetson Nano (20 FPS). Jadi kita tidak mengejar
  kecepatan — kita mengejar AKURASI GRATIS pada budget latency yang sama.
  RepC3k2 menyisipkan reparameterizable conv (RepConv) ke dalam bottleneck
  C3k2: saat training multi-branch (3x3 + 1x1 + identity) memberi representasi
  lebih kaya; saat inference di-fuse jadi single 3x3 conv -> latency IDENTIK
  dengan C3k2 biasa. "Free accuracy" jika hipotesis benar.

Implementasi:
  C3k2 (Ultralytics) meng-override self.m (list of Bottleneck) dari parent C2f.
  RepC3k2 hanya mengganti Bottleneck -> RepBottleneck (keduanya modul bawaan
  Ultralytics yang sudah teruji). RepBottleneck memakai RepConv yang sudah
  terbukti fuse-nya di eksperimen E1 (fuse OK, TensorRT-ready).

  Signature RepC3k2 identik dengan C3k2 -> drop-in replacement di YAML.

Referensi:
  RepVGG/RepConv — Ding et al., CVPR 2021.
  RepConv fuse via get_equivalent_kernel_bias() + fuse_convs() (bawaan Ultralytics).

Python 3.8 compatible.
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

    Signature identik dengan C3k2 (c1, c2, n, c3k, e, attn, g, shortcut) agar
    kompatibel dengan YAML parser (base_modules + repeat_modules). Saat c3k=True
    tetap memakai C3k (untuk konsistensi dengan varian besar); saat c3k=False
    (kasus YOLO11n) memakai RepBottleneck -> di sinilah free-accuracy bekerja.

    YAML usage (identik posisi arg dengan C3k2):
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
        # Bangun C3k2 standar dulu (mengeset cv1, cv2, dan self.m default)
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if RepBottleneck is None:
            raise ImportError("RepBottleneck tidak tersedia di ultralytics ini.")

        # Override self.m: ganti Bottleneck -> RepBottleneck (c3k=False path).
        # Untuk c3k=True, biarkan C3k (varian besar); YOLO11n memakai c3k=False.
        if not attn:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else RepBottleneck(self.c, self.c, shortcut, g)
                for _ in range(n)
            )
        # attn=True path dibiarkan default (tidak relevan untuk YOLO11n backbone)


# Self-test
if __name__ == "__main__":
    if RepBottleneck is None:
        print("ultralytics tidak tersedia -- jalankan di server.")
    else:
        print("Testing RepC3k2...")
        # Mirror pemakaian YOLO11n: c3k=False
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

        # Cek RepConv ada + bisa fuse
        from ultralytics.nn.modules.conv import RepConv
        reps = [mod for mod in m.modules() if isinstance(mod, RepConv)]
        print(f"  RepConv count: {len(reps)}")
        if reps and hasattr(reps[0], "fuse_convs"):
            print("  RepConv.fuse_convs available: OK (fuses at model.fuse())")

        print("\nAll tests passed.")
