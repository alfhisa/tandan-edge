#!/bin/bash
# ==============================================================================
# thermal_test.sh — sustained inference + temperature logging on the Jetson Nano
# ==============================================================================
# Runs tegrastats alongside continuous TensorRT inference of the deployed
# configuration (704 px, FP16) to check for:
#   1. thermal throttling (temperature climbing / GPU load dropping mid-run)
#   2. board power (VDD_IN, if this tegrastats variant reports it; the Nano
#      developer kit does not)
#   3. a throughput drop between the start and the end of the run
#
# trtexec --duration loops pure inference with no Python overhead, so the GPU
# load is heavier than in a real camera pipeline (a conservative test).
#
# Input : deploy/engines/r704_fp16.engine (built by jetson/benchmark_trt.sh)
# Output: results/jetson/thermal_30min/{idle_baseline.log, tegrastats.log,
#         trtexec_sustained.log, thermal_summary.txt}
#
# Usage (on the Jetson, from the repository root, MAXN power mode):
#   nohup bash jetson/thermal_test.sh > thermal_run.log 2>&1 &
#   STRESS_DURATION=300 OUT_NAME=thermal_5min bash jetson/thermal_test.sh   # 5-min variant
# ==============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ENGINE="${ENGINE:-$REPO/deploy/engines/r704_fp16.engine}"
OUT_DIR="$REPO/results/jetson/${OUT_NAME:-thermal_30min}"
STRESS_DURATION="${STRESS_DURATION:-1800}"   # seconds of continuous inference (1800 and 300 were used)
TEGRA_INTERVAL=1000                          # ms between tegrastats samples

mkdir -p "$OUT_DIR"

TEGRA_LOG="$OUT_DIR/tegrastats.log"
TRT_LOG="$OUT_DIR/trtexec_sustained.log"
SUMMARY="$OUT_DIR/thermal_summary.txt"

if [ ! -f "$ENGINE" ]; then
    echo "ERROR: engine not found: $ENGINE"
    echo "Build it with jetson/benchmark_trt.sh or set ENGINE=<path>."
    exit 1
fi

echo "======================================================================"
echo "  THERMAL & SUSTAINED INFERENCE TEST"
echo "======================================================================"
echo "  Engine   : $ENGINE"
echo "  Duration : ${STRESS_DURATION}s continuous inference"
echo "  Output   : $OUT_DIR"
echo ""

# --- idle baseline before the load (10 s) ---
echo "  [1/3] Idle baseline (10 s)..."
timeout 11 tegrastats --interval $TEGRA_INTERVAL > "$OUT_DIR/idle_baseline.log" 2>&1
sleep 1

# --- tegrastats in the background during the load ---
echo "  [2/3] tegrastats logging + continuous inference (${STRESS_DURATION}s)..."
tegrastats --interval $TEGRA_INTERVAL > "$TEGRA_LOG" 2>&1 &
TEGRA_PID=$!

sleep 2   # let tegrastats start before the load

START=$(date +%s)
$TRTEXEC \
    --loadEngine="$ENGINE" \
    --duration=$STRESS_DURATION \
    --warmUp=1000 \
    --percentile=95 \
    > "$TRT_LOG" 2>&1
END=$(date +%s)
ACTUAL_DURATION=$((END - START))

sleep 2   # capture the start of the cool-down
kill $TEGRA_PID 2>/dev/null
wait $TEGRA_PID 2>/dev/null

echo "  [3/3] Load finished (${ACTUAL_DURATION}s). Analysing logs..."
echo ""

# ==============================================================================
# Parsing & summary
# ==============================================================================
python3 - "$TEGRA_LOG" "$TRT_LOG" "$OUT_DIR/idle_baseline.log" "$SUMMARY" << 'PYEOF'
import re, sys, statistics

tegra_log, trt_log, idle_log, summary_path = sys.argv[1:5]

def parse_tegra_line(line):
    """Temperatures (GPU/CPU/thermal/...) and GR3D load from one tegrastats line."""
    out = {}
    for comp in ("GPU", "CPU", "PMIC", "thermal", "AO", "PLL"):
        m = re.search(rf'{comp}@([\d.]+)C', line)
        if m:
            out[f"temp_{comp}"] = float(m.group(1))
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        out["gr3d_freq_pct"] = int(m.group(1))
    m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
    m = re.search(r'VDD_IN\s+(\d+)/(\d+)', line)   # only on some Jetson models
    if m:
        out["power_now_mw"] = int(m.group(1))
    return out

def load_series(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                d = parse_tegra_line(line)
                if d:
                    rows.append(d)
    except FileNotFoundError:
        pass
    return rows

idle_rows = load_series(idle_log)
stress_rows = load_series(tegra_log)

lines = []
lines.append("=" * 70)
lines.append("  SUMMARY — THERMAL & SUSTAINED INFERENCE")
lines.append("=" * 70)

def series(rows, key):
    return [r[key] for r in rows if key in r]

lines.append(f"\n--- IDLE baseline (n={len(idle_rows)} samples) ---")
for key in ("temp_GPU", "temp_CPU", "temp_thermal", "temp_PMIC"):
    s = series(idle_rows, key)
    if s:
        lines.append(f"  {key:16s}: {statistics.mean(s):.1f}C "
                     f"(min {min(s):.1f}, max {max(s):.1f})")

lines.append(f"\n--- During STRESS (n={len(stress_rows)} samples) ---")
for key in ("temp_GPU", "temp_CPU", "temp_thermal", "temp_PMIC"):
    s = series(stress_rows, key)
    if s:
        lines.append(f"  {key:16s}: start={s[0]:.1f}C  end={s[-1]:.1f}C  "
                     f"mean={statistics.mean(s):.1f}C  max={max(s):.1f}C  "
                     f"delta(end-start)={s[-1]-s[0]:+.1f}C")

power_s = series(stress_rows, "power_now_mw")
if power_s:
    lines.append(f"\n  power_now_mw    : mean={statistics.mean(power_s):.0f}mW  "
                 f"max={max(power_s):.0f}mW")
else:
    lines.append("\n  [note] VDD_IN/power not reported by this tegrastats variant "
                 "(Jetson Nano developer kit has no on-board power sensor); an "
                 "external USB power meter is needed for watt figures.")

gr3d_s = series(stress_rows, "gr3d_freq_pct")
if gr3d_s:
    lines.append(f"\n  GR3D_FREQ (util GPU): mean={statistics.mean(gr3d_s):.0f}%  "
                 f"min={min(gr3d_s)}%  max={max(gr3d_s)}%")
    # rough throttling check: GPU load drops in the second half
    half = len(gr3d_s) // 2
    if half > 2:
        first_half = statistics.mean(gr3d_s[:half])
        second_half = statistics.mean(gr3d_s[half:])
        drop = first_half - second_half
        lines.append(f"  Util half-1={first_half:.0f}% vs half-2={second_half:.0f}% "
                     f"(difference {drop:+.1f} points)")
        if drop > 10:
            lines.append("  >> POSSIBLE THROTTLING: GPU load fell by >10 points "
                         "in the second half.")
        else:
            lines.append("  >> No sign of throttling in GPU load.")

lines.append(f"\n--- trtexec throughput ---")
try:
    with open(trt_log) as f:
        trt_text = f.read()
    thr = re.findall(r'Throughput:\s*([\d.]+)\s*qps', trt_text)
    if thr:
        lines.append(f"  Reported throughput: {thr[-1]} qps")
    lat = re.search(r'GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms', trt_text)
    if lat:
        lines.append(f"  GPU Compute Time mean: {lat.group(1)} ms")
except FileNotFoundError:
    lines.append("  [warn] trtexec log not found")

lines.append("\n" + "=" * 70)
lines.append("  INTERPRETATION")
lines.append("=" * 70)
gpu_stress = series(stress_rows, "temp_GPU")
if gpu_stress:
    delta = gpu_stress[-1] - gpu_stress[0]
    if delta > 5:
        lines.append(f"  GPU temperature rose {delta:+.1f}C during the test; check "
                     f"again for runs longer than {len(stress_rows)}s.")
    else:
        lines.append(f"  GPU temperature stable (delta {delta:+.1f}C) over "
                     f"{len(stress_rows)}s; no sign of overheating.")

text = "\n".join(lines)
print(text)
with open(summary_path, "w") as f:
    f.write(text + "\n")
PYEOF

echo ""
echo "  Summary        : $SUMMARY"
echo "  tegrastats log : $TEGRA_LOG"
echo "  trtexec log    : $TRT_LOG"
