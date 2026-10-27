# Jetson Nano scripts

Hardware and software used for every number in `results/jetson/`:

| | |
|---|---|
| Board | Jetson Nano 4 GB developer kit, `nvpmodel -m 0` (MAXN, 10 W), `jetson_clocks` not enabled |
| OS | L4T R32.7.6 (JetPack 4.6.6) |
| CUDA / cuDNN / TensorRT | 10.2.460 / 8.2.1 / 8.2.1 (`/usr/src/tensorrt/bin/trtexec`) |
| Python env (PyTorch benchmark only) | Python 3.8, PyTorch 1.11.0 (community aarch64 wheel), Ultralytics 8.3.152 |

The TensorRT measurements use `trtexec` only. The TensorRT 8.2 Python bindings
ship for the system Python 3.6, not for Python 3.8, so no Python is involved in
the engine benchmarks.

## Steps

```bash
# on the training machine
python scripts/export/export_onnx.py                  # deploy/onnx/best_{640,704,768}.onnx
python scripts/export/export_base_models_onnx.py      # deploy/onnx/{model}_640.onnx (optional)
scp -r deploy/onnx <user>@<jetson>:~/tandan-edge/deploy/

# on the Jetson (clone the repository there too), from the repository root
sudo nvpmodel -m 0
nohup bash jetson/benchmark_trt.sh > trt_run.log 2>&1 &          # Table 7: 9 engines, 30-60+ min
nohup bash jetson/thermal_test.sh > thermal_run.log 2>&1 &       # Table 8: 30 min at 704 px FP16
nohup bash jetson/benchmark_base_models.sh > base_run.log 2>&1 & # Table 4: 4 models @640 FP16
python3 -u jetson/bench_pytorch.py                                # Table 7: PyTorch rows
```

Engines are hardware- and TensorRT-version-specific, so they are rebuilt on the
device and never shipped. Results land in `results/jetson/`; compare with
`git diff results/jetson/`.

| Script | Output |
|---|---|
| `benchmark_trt.sh` | `results/jetson/trt_resolution/benchmark_summary.csv`, `logs/r{res}_{prec}.log` |
| `thermal_test.sh` | `results/jetson/thermal_30min/` (`STRESS_DURATION=300 OUT_NAME=thermal_5min` for the 5-min run) |
| `benchmark_base_models.sh` | `results/jetson/base_models_640/{model}_trt.log` |
| `bench_pytorch.py` | `results/jetson/pytorch_benchmark.csv` |

`results/jetson/base_models_640_june/` holds the first (June 2026) measurement of
the same four models with an earlier script; it agrees with the October
re-measurement within 1.5 %.
