#!/usr/bin/env python3
"""
bench_pytorch.py — PyTorch (standard practice) speed on the Jetson Nano,
the baseline against which the TensorRT engines are compared.

Two measurements per resolution (640/704/768) x precision (FP32/FP16):
  1. model_only : forward pass of the fused model only (comparable to the
                  trtexec "GPU compute" time), 1x3xSxS input, batch 1,
                  CUDA synchronisation every iteration
  2. end_to_end : YOLO.predict() on a real val image
                  (pre-processing + inference + NMS), as in normal use

Usage (on the Jetson, from the repository root, Python 3.8 venv with
PyTorch 1.11 + ultralytics 8.3.152, see requirements-jetson.txt):
  python3 -u jetson/bench_pytorch.py --weights weights/tandan-edge_yolo11n_train768.pt \
      --image datasets/mopad_site2/val/images/<any>.jpg \
      --out results/jetson/pytorch_benchmark.csv
Python 3.8 / PyTorch 1.11 compatible.
"""

import argparse
import csv
import glob
import statistics
import time

import torch
from ultralytics import YOLO


def pct(vals, p):
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def bench_model_only(net, size, half, warmup, iters):
    x = torch.rand(1, 3, size, size, device="cuda")
    if half:
        x = x.half()
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_end_to_end(yolo, img, size, half, warmup, iters):
    for _ in range(warmup):
        yolo.predict(img, imgsz=size, half=half, device=0, verbose=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        yolo.predict(img, imgsz=size, half=half, device=0, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/tandan-edge_yolo11n_train768.pt")
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 704, 768])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--e2e_iters", type=int, default=100)
    ap.add_argument("--image", default=None,
                    help="image for the end-to-end test (default: first val image)")
    ap.add_argument("--out", default="results/jetson/pytorch_benchmark.csv")
    a = ap.parse_args()

    img = a.image or sorted(glob.glob("datasets/mopad_site2/val/images/*"))[0]
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
    print("weights", a.weights, "| image", img, "\n")

    rows = []
    for half in (False, True):
        prec = "fp16" if half else "fp32"
        # model only: fused nn module (Conv+BN), eval mode
        net = YOLO(a.weights).model.fuse().cuda().eval()
        if half:
            net = net.half()
        yolo = YOLO(a.weights)  # separate object for predict()
        for s in a.sizes:
            t_m = bench_model_only(net, s, half, a.warmup, a.iters)
            t_e = bench_end_to_end(yolo, img, s, half, 10, a.e2e_iters)
            r = {
                "framework": "pytorch", "resolution": s, "precision": prec,
                "model_mean_ms": round(statistics.mean(t_m), 2),
                "model_median_ms": round(statistics.median(t_m), 2),
                "model_p95_ms": round(pct(t_m, 95), 2),
                "model_fps": round(1000.0 / statistics.mean(t_m), 2),
                "e2e_mean_ms": round(statistics.mean(t_e), 2),
                "e2e_fps": round(1000.0 / statistics.mean(t_e), 2),
            }
            rows.append(r)
            print("{:5s} {:4d}  model {:7.2f} ms ({:5.2f} FPS)   e2e {:7.2f} ms ({:5.2f} FPS)".format(
                prec, s, r["model_mean_ms"], r["model_fps"], r["e2e_mean_ms"], r["e2e_fps"]))
        del net, yolo
        torch.cuda.empty_cache()

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nDone ->", a.out)


if __name__ == "__main__":
    main()
