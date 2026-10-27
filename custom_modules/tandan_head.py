"""
tandan_head.py — TandanDetect: Shared-Conv Separate-BN Detection Head (R1)
===========================================================================
Komponen K2 dari TANDAN-YOLO. Mengganti Detect head YOLO11 dengan head
shared-convolution:

  Baseline Detect (per level P3/P4/P5):
    box branch : Conv3x3 -> Conv3x3 -> Conv2d(4*reg_max)   } masing-masing
    cls branch : DWConv stack       -> Conv2d(nc)          } level punya
                                                             bobot sendiri
  TandanDetect (SCS-Head):
    stem SHARED : unify 1x1 (per level) -> [convA 3x3 SHARED + BN per level]
                  -> [convB 3x3 SHARED + BN per level]
    box  : Conv2d(4*reg_max) SHARED -> Scale (learnable per level)
    cls  : Conv2d(nc) SHARED

  Prinsip "shared convolution, separate BN":
  - Bobot conv dibagi antar 3 level -> params turun signifikan
  - BN terpisah per level -> statistik feature tiap skala tetap terkalibrasi
  - Scale layer per level mengkompensasi perbedaan magnitudo antar skala
  - Stem dihitung SEKALI per level dan dipakai bersama box+cls
    (baseline menghitung dua branch terpisah) -> GFLOPs head turun

Semua op standar (conv/BN/SiLU/mul) -> ONNX opset 12 & TensorRT 8.2 safe.
Referensi: LSCD/SCSHead family (ELS-YOLO Sensors 2025; YOLOv8-PD; LEAD-YOLO).

Python 3.8 compatible.
"""

import math

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.conv import Conv
except Exception:  # pragma: no cover
    Detect = object
    Conv = None


class Scale(nn.Module):
    """Learnable scalar multiplier (per-level compensation)."""

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return x * self.scale


class TandanDetect(Detect):
    """
    SCS-Head: shared conv + separate BN detection head, drop-in Detect.

    Subclass Detect agar:
    - isinstance(m, Detect) tetap True (stride init & task detection jalan)
    - atribut warisan (nc, reg_max, no, stride, dfl, anchors) tetap dipakai
    Forward & bias_init di-override; cv2/cv3 warisan dikosongkan.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch=()):
        # Signature WAJIB mencerminkan Detect.__init__ Ultralytics 8.4.48:
        #   __init__(self, nc=80, reg_max=16, end2end=False, ch=())
        # parse_model memanggil m(*args) secara positional -> kalau signature
        # tidak sama persis, argumen bergeser / jumlahnya tidak cocok.
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        hidc = ch[0]  # 64 utk YOLO11n (P3) — dim bersama stem

        # Buang branch per-level bawaan (params dihemat)
        self.cv2 = nn.ModuleList(nn.Identity() for _ in ch)
        self.cv3 = nn.ModuleList(nn.Identity() for _ in ch)

        # Per-level: unify channel ke hidc (Conv = conv+BN+SiLU Ultralytics)
        self.unify = nn.ModuleList(Conv(c, hidc, 1, 1) for c in ch)

        # SHARED conv weights (tanpa bias; BN per level yang bawa bias)
        self.convA = nn.Conv2d(hidc, hidc, 3, 1, 1, bias=False)
        self.convB = nn.Conv2d(hidc, hidc, 3, 1, 1, bias=False)
        # Separate BN per level
        self.bnA = nn.ModuleList(nn.BatchNorm2d(hidc) for _ in ch)
        self.bnB = nn.ModuleList(nn.BatchNorm2d(hidc) for _ in ch)
        self.act = nn.SiLU()

        # SHARED final projections
        self.final_box = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.final_cls = nn.Conv2d(hidc, self.nc, 1)
        # Per-level scale utk box (kompensasi magnitudo antar skala)
        self.scales = nn.ModuleList(Scale(1.0) for _ in ch)

    def forward(self, x):
        """
        Ultralytics 8.4.48 Detect API: forward mengembalikan dict
            {"boxes": (bs, 4*reg_max, N), "scores": (bs, nc, N), "feats": list}
        dengan N = total anchor seluruh level (P3+P4+P5 di-flatten & concat).
        Format ini yang dikonsumsi v8DetectionLoss dan _inference.
        """
        bs = x[0].shape[0]
        boxes_l, scores_l = [], []
        for i in range(self.nl):
            s = self.unify[i](x[i])
            s = self.act(self.bnA[i](self.convA(s)))
            s = self.act(self.bnB[i](self.convB(s)))
            box = self.scales[i](self.final_box(s))   # (bs, 4*reg_max, H, W)
            cls = self.final_cls(s)                   # (bs, nc,        H, W)
            boxes_l.append(box.view(bs, 4 * self.reg_max, -1))
            scores_l.append(cls.view(bs, self.nc, -1))

        preds = dict(
            boxes=torch.cat(boxes_l, dim=-1),
            scores=torch.cat(scores_l, dim=-1),
            feats=x,  # dipakai _get_decode_boxes untuk shape & make_anchors
        )
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Init bias final conv (shared -> pakai stride tengah 16 sbg acuan)."""
        self.final_box.bias.data[:] = 1.0
        self.final_cls.bias.data[: self.nc] = math.log(
            5 / self.nc / (640 / 16) ** 2
        )


# Self-test ringan (butuh ultralytics; jalankan di server)
if __name__ == "__main__":
    if Conv is None:
        print("ultralytics tidak tersedia -- jalankan di server.")
    else:
        print("Testing TandanDetect (SCS-Head)...")
        ch = (64, 128, 256)  # P3,P4,P5 YOLO11n
        head = TandanDetect(nc=5, ch=ch)
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.bias_init()

        feats = [torch.randn(1, c, s, s) for c, s in zip(ch, (80, 40, 20))]
        n_anchors = 80 * 80 + 40 * 40 + 20 * 20

        head.train()
        out = head([f.clone() for f in feats])
        print("  train output keys:", sorted(out.keys()))
        print("  boxes :", tuple(out["boxes"].shape),
              f"(expect (1, {4*head.reg_max}, {n_anchors}))")
        print("  scores:", tuple(out["scores"].shape),
              f"(expect (1, 5, {n_anchors}))")
        assert out["boxes"].shape == (1, 4 * head.reg_max, n_anchors)
        assert out["scores"].shape == (1, 5, n_anchors)

        n_params = sum(p.numel() for p in head.parameters())
        print(f"  head params: {n_params/1e3:.1f} K  (baseline Detect ~430 K)")

        head.eval()
        with torch.no_grad():
            y = head([f.clone() for f in feats])
        pred = y[0] if isinstance(y, tuple) else y
        print("  eval output:", tuple(pred.shape),
              f"(expect (1, {4 + 5}, {n_anchors}))")
        assert pred.shape == (1, 4 + 5, n_anchors)
        print("All tests passed.")
