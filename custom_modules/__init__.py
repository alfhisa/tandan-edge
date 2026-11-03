"""
custom_modules — layers and losses tested during model screening.

None of these is used by the deployed model (plain YOLO11n trained at 768 px).
They are kept so that every screening run in results/ can be reproduced and
so that their best.pt checkpoints can be unpickled (the checkpoints reference
`custom_modules.<module>`).

  spd_conv.py        SPD-Conv downsampling                (A1, YOLO12n)
  varifocal_loss.py  Varifocal Loss                       (B1, YOLO12n)
  a2c2f_rep.py       A2C2fRep, attention-free A2C2f       (E1, YOLO12n)
  cb_focal_loss.py   class-balanced focal loss            (T1, YOLO11n)
  repc3k2.py         RepC3k2 re-parameterisable block     (T2, YOLO11n)
  tandan_head.py     SCS-Head                             (R1, YOLO11n)
  tandan_simam.py    C3k2 + SimAM attention               (R2, YOLO11n)
  tandan_bifpn.py    BiFPN weighted concatenation         (R3, YOLO11n)

Each group is registered into Ultralytics by its own helper in scripts/train/
(register_spd.py, register_rep.py, register_repc3k2.py, register_tandan.py,
patch_vfl.py, patch_cb_focal.py); nothing in the Ultralytics package is edited.
"""
