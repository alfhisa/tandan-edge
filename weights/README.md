# Weights

Checkpoints are not stored in git. Download them from the
[GitHub release](https://github.com/<github-user>/tandan-edge/releases) of this
repository and place them here.

| File | What it is | Used by |
|---|---|---|
| `tandan-edge_yolo11n_train768.pt` | YOLO11n trained at 768 px, seed 42 (run `R4_res768_seed42`, md5 `62ccdc14065b…`). Deployed at 704 px. | `scripts/export/export_onnx.py`, `scripts/paper/make_figures.py` (F7, F8), `jetson/bench_pytorch.py` |
| `tandan-edge_yolo11n_train768_704.onnx` | the same model exported at 704 px (opset 12, static batch 1) | build the TensorRT engine directly on a Jetson |

`pretrained/` is filled automatically with the COCO checkpoints from
Ultralytics (`yolo11n.pt`, `yolo12n.pt`, `yolov8n.pt`, `yolo26n.pt`) the first
time a training script runs.

Quick test:

```python
from ultralytics import YOLO
model = YOLO("weights/tandan-edge_yolo11n_train768.pt")
model.predict("path/to/uav_tile.jpg", imgsz=704, conf=0.25, save=True)
```
