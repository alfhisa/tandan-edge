#!/bin/bash
# ==============================================================================
# benchmark_trt.sh — build TensorRT engines and measure speed on the Jetson Nano
# (originally benchmark_jetson.sh; Table 7 / Fig. 5)
# ==============================================================================
# 3 input resolutions (640/704/768) x 3 precisions (FP32/FP16/INT8) = 9 configs.
#
# trtexec is a stand-alone C++ binary shipped with JetPack: no PyTorch,
# Ultralytics or TensorRT Python bindings are needed (the TensorRT 8.2 Python
# bindings do not exist for Python 3.8 on JetPack 4.6). One trtexec call builds
# the engine and then benchmarks it.
#
# INT8: without a calibration cache trtexec still builds an INT8 engine (random
# dynamic ranges). That is valid for measuring SPEED only, which is all it is
# used for here: the Maxwell GPU of the Nano has no INT8 acceleration, and the
# result (INT8 = FP32 speed) confirms it.
#
# Input : deploy/onnx/best_{640,704,768}.onnx  (scripts/export/export_onnx.py)
# Output: results/jetson/trt_resolution/benchmark_summary.csv + logs/
#         deploy/engines/r{res}_{prec}.engine
#
# Usage (on the Jetson, from the repository root, MAXN power mode):
#   sudo nvpmodel -m 0
#   nohup bash jetson/benchmark_trt.sh > trt_run.log 2>&1 &
# Takes 30-60+ min (each FP16 engine build ~15 min).
# ==============================================================================

set -uo pipefail  # no -e: one failing configuration must not stop the others

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ONNX_DIR="$REPO/deploy/onnx"
OUT_DIR="$REPO/results/jetson/trt_resolution"
ENGINE_DIR="$REPO/deploy/engines"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_CSV="$OUT_DIR/benchmark_summary.csv"

RESOLUTIONS=(640 704 768)
PRECISIONS=("fp32" "fp16" "int8")
WORKSPACE_MB=2048
AVG_RUNS=100
PERCENTILE=95   # trtexec always prints the median; --percentile adds p95

mkdir -p "$ENGINE_DIR" "$LOG_DIR"

if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    exit 1
fi

echo "======================================================================"
echo "  JETSON NANO BENCHMARK — TensorRT ${RESOLUTIONS[*]} x ${PRECISIONS[*]}"
echo "======================================================================"
echo "  ONNX dir    : $ONNX_DIR"
echo "  Output dir  : $OUT_DIR"
echo "  Workspace   : ${WORKSPACE_MB} MB | avgRuns: $AVG_RUNS | percentile: $PERCENTILE"
echo ""

echo "resolution,precision,build_ok,engine_size_mb,build_time_s,throughput_qps,latency_mean_ms,latency_median_ms,latency_p${PERCENTILE}_ms,gpu_compute_mean_ms,gpu_compute_median_ms,gpu_compute_p${PERCENTILE}_ms" > "$SUMMARY_CSV"

for RES in "${RESOLUTIONS[@]}"; do
    ONNX_FILE="$ONNX_DIR/best_${RES}.onnx"

    if [ ! -f "$ONNX_FILE" ]; then
        echo "  [SKIP] ONNX not found: $ONNX_FILE"
        continue
    fi

    for PREC in "${PRECISIONS[@]}"; do
        TAG="r${RES}_${PREC}"
        ENGINE_FILE="$ENGINE_DIR/${TAG}.engine"
        LOG_FILE="$LOG_DIR/${TAG}.log"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "  [$TAG] resolution=${RES}px  precision=${PREC}"
        echo "----------------------------------------------------------------------"

        PREC_FLAG=""
        case "$PREC" in
            fp32) PREC_FLAG="" ;;
            fp16) PREC_FLAG="--fp16" ;;
            int8) PREC_FLAG="--int8" ;;
        esac

        START=$(date +%s)
        $TRTEXEC \
            --onnx="$ONNX_FILE" \
            --saveEngine="$ENGINE_FILE" \
            --workspace=$WORKSPACE_MB \
            --avgRuns=$AVG_RUNS \
            --percentile=$PERCENTILE \
            $PREC_FLAG \
            > "$LOG_FILE" 2>&1
        RC=$?
        END=$(date +%s)
        BUILD_TIME=$((END - START))

        if [ $RC -ne 0 ] || [ ! -f "$ENGINE_FILE" ]; then
            echo "  [FAILED] $TAG -- see $LOG_FILE"
            tail -15 "$LOG_FILE" | sed 's/^/    /'
            echo "$RES,$PREC,FALSE,,,,,,,,,," >> "$SUMMARY_CSV"
            continue
        fi

        ENGINE_SIZE_MB=$(du -m "$ENGINE_FILE" | cut -f1)

        # --- parse trtexec output ---
        THROUGHPUT=$(grep -oP 'Throughput:\s*\K[0-9.]+' "$LOG_FILE" | tail -1)

        LAT_LINE=$(grep -oP 'GPU Compute Time:\s*\Kmin.*' "$LOG_FILE" | tail -1)
        GPU_MEAN=$(echo "$LAT_LINE" | grep -oP 'mean\s*=\s*\K[0-9.]+')
        GPU_MEDIAN=$(echo "$LAT_LINE" | grep -oP 'median\s*=\s*\K[0-9.]+')
        GPU_PCT=$(echo "$LAT_LINE" | grep -oP "percentile\(${PERCENTILE}%\)\s*=\s*\K[0-9.]+")

        LAT2_LINE=$(grep -oP '^\[I\] Latency:\s*\Kmin.*' "$LOG_FILE" | tail -1)
        LAT_MEAN=$(echo "$LAT2_LINE" | grep -oP 'mean\s*=\s*\K[0-9.]+')
        LAT_MEDIAN=$(echo "$LAT2_LINE" | grep -oP 'median\s*=\s*\K[0-9.]+')
        LAT_PCT=$(echo "$LAT2_LINE" | grep -oP "percentile\(${PERCENTILE}%\)\s*=\s*\K[0-9.]+")

        echo "  [OK] Throughput=${THROUGHPUT} qps | GPU mean=${GPU_MEAN}ms median=${GPU_MEDIAN}ms p${PERCENTILE}=${GPU_PCT}ms"
        echo "       Engine size=${ENGINE_SIZE_MB}MB | Build time=${BUILD_TIME}s"

        echo "$RES,$PREC,TRUE,$ENGINE_SIZE_MB,$BUILD_TIME,$THROUGHPUT,$LAT_MEAN,$LAT_MEDIAN,$LAT_PCT,$GPU_MEAN,$GPU_MEDIAN,$GPU_PCT" >> "$SUMMARY_CSV"
    done
done

echo ""
echo "======================================================================"
echo "  DONE"
echo "======================================================================"
echo "  Summary  : $SUMMARY_CSV"
echo "  Engines  : $ENGINE_DIR/"
echo "  Logs     : $LOG_DIR/"
echo ""
column -s, -t "$SUMMARY_CSV"
