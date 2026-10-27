#!/bin/bash
# ==============================================================================
# benchmark_base_models.sh — TensorRT FP16 speed of the 4 base detectors @640
# (Table 4, column "Jetson FPS"; logs in results/jetson/base_models_640/)
# ==============================================================================
# Same trtexec call that produced the 27 Oct 2026 logs (first line of each log):
#   trtexec --onnx=<model>_640.onnx --fp16 --workspace=1024 \
#           --saveEngine=<model>_640_fp16.engine --warmUp=2000 --duration=60
# YOLO26n fails to build on TensorRT 8.2 (the ONNX parser rejects its Mod node);
# the failure is part of the result.
#
# Input : deploy/onnx/{yolo11n,yolo12n,yolov8n,yolo26n}_640.onnx
#         (scripts/export/export_base_models_onnx.py)
# Usage (on the Jetson, from the repository root, MAXN power mode):
#   nohup bash jetson/benchmark_base_models.sh > base_models_run.log 2>&1 &
# ==============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ONNX_DIR="$REPO/deploy/onnx"
ENGINE_DIR="$REPO/deploy/engines"
OUT_DIR="$REPO/results/jetson/base_models_640"
mkdir -p "$ENGINE_DIR" "$OUT_DIR"

for M in yolo11n yolo12n yolov8n yolo26n; do
    ONNX="$ONNX_DIR/${M}_640.onnx"
    if [ ! -f "$ONNX" ]; then
        echo "[SKIP] $ONNX not found"
        continue
    fi
    "$TRTEXEC" --onnx="$ONNX" --fp16 --workspace=1024 \
        --saveEngine="$ENGINE_DIR/${M}_640_fp16.engine" \
        --warmUp=2000 --duration=60 > "$OUT_DIR/${M}_trt.log" 2>&1
    QPS=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$OUT_DIR/${M}_trt.log" | tail -1)
    echo "$M done $(date +%H:%M)  throughput=${QPS:-FAILED} qps" | tee -a "$OUT_DIR/trt_cmp.log"
done
