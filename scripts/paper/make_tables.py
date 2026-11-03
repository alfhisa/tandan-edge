#!/usr/bin/env python3
"""
make_tables.py — Every data table, rebuilt from results/.

Output: results/tables/T*.csv + results/tables/ALL_TABLES.md

  T1 hardware/software environment     T5 modification screening (+ Welch t-test)
  T2 dataset statistics (from labels)  T6 training x inference resolution
  T3 hyperparameters                   T7 Jetson benchmark (PyTorch vs TensorRT)
  T4 base-model screening              T8 thermal summary, 5 vs 30 min

Usage (from the repository root):
  python scripts/paper/make_tables.py              # all (T2 needs the dataset)
  python scripts/paper/make_tables.py --only T4 T5 # only tables that read results/
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from tandan_paths import P  # noqa: E402  (maps the original workspace paths to this repository)

OUT = P("tables")
CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]
LABELS = ["Dead", "Healthy", "Mismanaged", "Smallish", "Yellowish"]
MD = []   # potongan markdown untuk ALL_TABLES.md


# ------------------------------------------------------------------ helpers
def rcsv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def f4(x):
    return "" if x is None or x == "" else "%.4f" % float(x)


def pm(m, s):
    if m in (None, ""):
        return ""
    return "%.4f ± %.4f" % (float(m), float(s)) if s not in (None, "") else "%.4f" % float(m)


def pfmt(p):
    if p is None:
        return ""
    return "< 0.001" if p < 0.001 else "%.3f" % p


def welch(a, b):
    try:
        from scipy.stats import ttest_ind
        if len(a) < 2 or len(b) < 2:
            return None
        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return None


def write(name, title, header, rows, note=""):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["## %s — %s" % (name, title), "", "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", "_%s_" % note]
    MD.append("\n".join(md) + "\n")
    print("  ->", os.path.join(OUT, name + ".csv"), "(%d rows)" % len(rows))


def metrics_json(run_path):
    js = glob.glob(os.path.join(P(run_path), "*_metrics.json"))
    if not js:
        return {}
    try:
        return json.load(open(js[0]))
    except Exception:
        return {}


def trt_fps(model):
    p = P("jetson_results/onnx_cmp_logs/%s_trt.log") % model
    if not os.path.isfile(p):
        return None
    m = re.search(r"Throughput: ([0-9.]+) qps", open(p, errors="replace").read())
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------ T1
def t1():
    rows = [
        ["Training / evaluation", "GPU", "NVIDIA Quadro RTX 6000 (24 GB)"],
        ["Training / evaluation", "Driver / CUDA", "450.248 / CUDA 11.7 runtime (PyTorch wheel)"],
        ["Training / evaluation", "Python", "3.8.10"],
        ["Training / evaluation", "PyTorch / torchvision", "2.0.1+cu117 / 0.15.2"],
        ["Training / evaluation", "Ultralytics", "8.4.48"],
        ["Edge device", "Board", "NVIDIA Jetson Nano Developer Kit 4 GB (Maxwell GPU, 128 CUDA cores)"],
        ["Edge device", "OS / JetPack", "L4T R32.7.6 / JetPack 4.6.6"],
        ["Edge device", "CUDA / cuDNN", "10.2.460 / 8.2.1"],
        ["Edge device", "TensorRT", "8.2.1 (trtexec)"],
        ["Edge device", "PyTorch / Ultralytics", "1.11.0 / 8.3.152 (Python 3.8)"],
        ["Edge device", "Power mode", "nvpmodel 0 (MAXN, 10 W); jetson_clocks not enabled"],
    ]
    write("T1_environment", "Hardware and software environment", ["Stage", "Component", "Specification"], rows)


# ------------------------------------------------------------------ T2
def t2():
    from PIL import Image
    res = {}
    sizes = set()
    for sp in ("train", "val"):
        cnt = [0] * 5
        side = [[] for _ in range(5)]
        imgs = sorted(glob.glob(P("site_2_only_dataset/%s/images/*") % sp))
        for ip in imgs:
            with Image.open(ip) as im:
                W, H = im.size
            sizes.add((W, H))
            lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if not os.path.isfile(lp):
                continue
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                cnt[c] += 1
                side[c].append((float(p[3]) * W + float(p[4]) * H) / 2)
        res[sp] = (len(imgs), cnt, side)
    tot_train = sum(res["train"][1])
    rows = []
    for k in range(5):
        tr, va = res["train"][1][k], res["val"][1][k]
        allside = res["train"][2][k] + res["val"][2][k]
        rows.append([LABELS[k], format(tr, ","), format(va, ","), format(tr + va, ","),
                     "%.2f%%" % (100.0 * tr / tot_train),
                     "%.1f ± %.1f" % (st.mean(allside), st.stdev(allside)) if len(allside) > 1 else ""])
    rows.append(["Total", format(sum(res["train"][1]), ","), format(sum(res["val"][1]), ","),
                 format(sum(res["train"][1]) + sum(res["val"][1]), ","), "100%", ""])
    hc = max(res["train"][1]) / max(1, min(res["train"][1]))
    note = ("Images: %d train / %d val; image size(s): %s. Imbalance (largest/smallest class, train) = %.0f:1. "
            "Box side = mean of width and height in pixels." % (
                res["train"][0], res["val"][0], ", ".join("%dx%d" % s for s in sorted(sizes)), hc))
    write("T2_dataset", "MOPAD Site 2 dataset statistics",
          ["Class", "Train instances", "Val instances", "Total", "Share of train", "Box side px (mean ± SD)"], rows, note)


# ------------------------------------------------------------------ T3
def t3():
    args = {}
    p = P("tandan-yolo/runs/R4_res768_seed42/args.yaml")
    if os.path.isfile(p):
        for line in open(p):
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                args[k.strip()] = v.strip()
    keys = [("epochs", "Epochs"), ("patience", "Early-stopping patience"), ("batch", "Batch size"),
            ("imgsz", "Training resolution (final model)"), ("optimizer", "Optimizer"), ("lr0", "Initial learning rate"),
            ("lrf", "Final LR factor"), ("momentum", "Momentum"), ("weight_decay", "Weight decay"),
            ("warmup_epochs", "Warm-up epochs"), ("cos_lr", "Cosine LR"), ("mosaic", "Mosaic probability"),
            ("close_mosaic", "Close mosaic (last epochs)"), ("mixup", "Mixup"), ("hsv_h", "HSV hue"),
            ("hsv_s", "HSV saturation"), ("hsv_v", "HSV value"), ("degrees", "Rotation (deg)"),
            ("translate", "Translate"), ("scale", "Scale"), ("fliplr", "Horizontal flip"), ("flipud", "Vertical flip"),
            ("amp", "Mixed precision (AMP)"), ("pretrained", "Pretrained weights")]
    if args.get("pretrained", "").endswith(".pt"):
        args["pretrained"] = os.path.basename(args["pretrained"]) + " (COCO-pretrained, Ultralytics)"
    rows = [["Training", lab, args.get(k, "")] for k, lab in keys if k in args]
    rows.insert(4, ["Training", "Training resolutions studied", "640, 704, 768"])
    rows.append(["Training", "Seeds", "42, 123, 456"])
    rows += [["Evaluation", "Weights", "best.pt (highest validation fitness)"],
             ["Evaluation", "Batch / conf / IoU (NMS)", "16 / 0.001 / 0.7"],
             ["Evaluation", "Confusion matrix thresholds", "conf 0.25, IoU 0.45"],
             ["Evaluation", "Precision", "FP32 and FP16 (PyTorch half)"],
             ["Export", "ONNX", "opset 12, simplified, static batch 1, input 1×3×S×S"],
             ["Export", "TensorRT build", "trtexec --fp16 (also FP32, INT8 without calibration), workspace 2048 MB"],
             ["Benchmark", "trtexec timing", "avgRuns 100, percentile 95; sustained test --duration 1800 s"],
             ["Benchmark", "Thermal logging", "tegrastats, 1 s interval"]]
    write("T3_hyperparameters", "Training, evaluation, export and benchmark settings", ["Stage", "Setting", "Value"], rows,
          "Training values read from results/runs/R4_res768_seed42/args.yaml (identical for all runs except imgsz/seed).")


# ------------------------------------------------------------------ T4
def t4():
    rs = {r["group"]: r for r in rcsv(P("reeval/reeval_summary.csv"))}
    runs = rcsv(P("reeval/reeval_runs.csv"))
    ref = [float(r["mAP50_95"]) for r in runs if r["group"] == "yolo11n"]
    rows = []
    for g, lab in (("yolov8n", "YOLOv8n"), ("yolo11n", "YOLO11n"), ("yolo12n", "YOLO12n"), ("yolo26n", "YOLO26n")):
        r = rs[g]
        path = [x["path"] for x in runs if x["group"] == g and x["run"].endswith("seed42")]
        prof = metrics_json(path[0]).get("profile", {}) if path else {}
        fps = trt_fps(g)
        vals = [float(x["mAP50_95"]) for x in runs if x["group"] == g]
        p = welch(vals, ref) if g != "yolo11n" else None
        rows.append([lab, pm(r["mAP50_95_mean"], r["mAP50_95_std"]), pfmt(p), f4(r["mAP50_mean"]),
                     f4(r["ap5095_mismanaged_mean"]), "%.0f" % float(r["bg_fp_total_mean"]),
                     prof.get("params_M", ""), prof.get("GFLOPs", ""),
                     "%.1f" % fps if fps else "—", "yes" if fps else "no (Mod op)",
                     "yes" if (fps or 0) >= 15 else "no"])
    write("T4_base_models", "Base-model screening at 640 px (3 seeds each)",
          ["Model", "mAP50-95", "p vs YOLO11n", "mAP50", "AP50-95 mismanaged", "Background FP",
           "Params (M)", "GFLOPs", "Jetson FPS (TRT FP16)", "TensorRT 8.2 build", "≥ 15 FPS"], rows,
          "Accuracy: unified re-evaluation (Ultralytics 8.4.48, val Site 2). p: Welch t-test over seeds. "
          "Jetson FPS: trtexec FP16, 640 px, 27 Oct 2026 (June measurements within 1.5%).")


# ------------------------------------------------------------------ T5
MODS = [  # group, base group, base label, component, change, decision
    ("A1_spd", "yolo12n", "YOLO12n", "Backbone", "SPD-Conv downsampling"),
    ("B1_vfl", "yolo12n", "YOLO12n", "Loss", "Varifocal Loss"),
    ("E1", "yolo12n", "YOLO12n", "Backbone", "A2C2f → A2C2fRep (attention removed)"),
    ("T1_lossmod", "yolo11n", "YOLO11n", "Loss", "Class-balanced focal loss"),
    ("T2_repc3k2", "yolo11n", "YOLO11n", "Backbone", "RepC3k2 (re-parameterisation)"),
    # T4a_sparsity (Network Slimming) is excluded: the L1 penalty was registered on a
    # callback event that Ultralytics never fires, so the run equals yolo11n_seed42.
    ("R1_scshead", "yolo11n", "YOLO11n", "Head", "SCS-Head"),
    ("R2_simam", "yolo11n", "YOLO11n", "Backbone", "SimAM attention"),
    ("R3_bifpn", "yolo11n", "YOLO11n", "Neck", "BiFPN weighted fusion"),
    ("R6_adamw768", "R4_res768", "YOLO11n @768", "Optimiser", "AdamW (lr0 0.001) instead of auto"),
]


def t5():
    rs = {r["group"]: r for r in rcsv(P("reeval/reeval_summary.csv"))}
    runs = rcsv(P("reeval/reeval_runs.csv"))
    per = {}
    for r in runs:
        per.setdefault(r["group"], []).append(float(r["mAP50_95"]))
    rows = []
    for g, base, blab, comp, change in MODS:
        if g not in rs:
            continue
        r, b = rs[g], rs[base]
        d = float(r["mAP50_95_mean"]) - float(b["mAP50_95_mean"])
        p = welch(per[g], per[base])
        dbg = float(r["bg_fp_total_mean"]) / max(1.0, float(b["bg_fp_total_mean"]))
        path = [x["path"] for x in runs if x["group"] == g]
        prof = metrics_json(path[0]).get("profile", {}) if path else {}
        noise = float(b["mAP50_95_std"] or 0.0026)
        fp_note = ", background FP ×%.1f" % dbg if dbg > 1.5 else ""
        bpath = [x["path"] for x in runs if x["group"] == base]
        bprof = metrics_json(bpath[0]).get("profile", {}) if bpath else {}
        try:
            dg = float(prof.get("GFLOPs")) / float(bprof.get("GFLOPs")) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            dg = 0.0
        if dg > 0.2:
            fp_note += ", +%.0f%% GFLOPs" % (dg * 100)
        if p is not None:
            if p < 0.05:
                dec = ("Significant gain" if d > 0 else "Not adopted: significant loss") + fp_note
            else:
                dec = "Not adopted: no significant difference" + fp_note
        elif abs(d) <= noise:
            dec = "Not adopted: within seed noise (single seed)" + fp_note
        else:
            dec = ("Not adopted: accuracy loss (single seed)" if d < 0 else "Gain (single seed, unconfirmed)") + fp_note
        rows.append([change, comp, blab, int(r["n_seeds"]), pm(r["mAP50_95_mean"], r["mAP50_95_std"]),
                     "%+.4f" % d, pfmt(p), f4(r["ap5095_mismanaged_mean"]), "%.0f" % float(r["bg_fp_total_mean"]),
                     prof.get("GFLOPs", ""), dec])
    write("T5_modifications", "Screening of architecture, loss and optimiser changes",
          ["Change", "Component", "Base", "Seeds", "mAP50-95", "Δ vs base", "p (Welch)", "AP50-95 mismanaged",
           "Background FP", "GFLOPs", "Decision"], rows,
          "Base references (3 seeds, 640 px): YOLO11n 0.9619 ± 0.0023, YOLO12n 0.9696 ± 0.0026; R6 compared with YOLO11n trained at 768 px. "
          "p not available for single-seed runs.")


# ------------------------------------------------------------------ T6
def t6():
    rows_in = rcsv(P("reeval/train_infer_runs.csv"))
    fps = {}
    for r in rcsv(P("jetson_results/trt_results/benchmark_summary.csv")):
        fps[(int(r["resolution"]), r["precision"])] = float(r["throughput_qps"])

    def vals(tr, inf, prec, key="mAP50_95"):
        return [float(r[key]) for r in rows_in if int(r["train_imgsz"]) == tr and int(r["infer_imgsz"]) == inf and r["precision"] == prec]
    final = vals(768, 704, "fp32")
    rows = []
    for tr in (640, 704, 768):
        for inf in (640, 704, 768):
            a32, a16, mis = vals(tr, inf, "fp32"), vals(tr, inf, "fp16"), vals(tr, inf, "fp32", "ap5095_mismanaged")
            p = None if (tr, inf) == (768, 704) else welch(final, a32)
            tag = "deployed" if (tr, inf) == (768, 704) else ("train = infer" if tr == inf else "")
            rows.append([tr, inf, tag, pm(st.mean(a32), st.stdev(a32)), pm(st.mean(a16), st.stdev(a16)),
                         "%+.4f" % (st.mean(a16) - st.mean(a32)), f4(st.mean(mis)),
                         "%.1f" % fps[(inf, "fp16")], "yes" if fps[(inf, "fp16")] >= 15 else "no", pfmt(p)])
    write("T6_resolution", "YOLO11n training × inference resolution (3 seeds each)",
          ["Train px", "Infer px", "Note", "mAP50-95 FP32", "mAP50-95 FP16", "Δ FP16", "AP50-95 mismanaged (FP32)",
           "Jetson FPS (TRT FP16)", "≥ 15 FPS", "p vs deployed (FP32)"], rows,
          "FP16 accuracy measured with PyTorch half precision on the server. p: Welch t-test against train 768 / infer 704.")


# ------------------------------------------------------------------ T7
def t7():
    tb = {(int(r["resolution"]), r["precision"]): r for r in rcsv(P("jetson_results/trt_results/benchmark_summary.csv"))}
    pt = {(int(r["resolution"]), r["precision"]): r for r in rcsv(P("jetson_results/pytorch_benchmark.csv"))}
    rows = []
    for res in (640, 704, 768):
        base = float(pt[(res, "fp32")]["model_fps"])
        for prec in ("fp32", "fp16"):
            r = pt[(res, prec)]
            rows.append([res, "PyTorch", prec.upper(), "%.2f" % float(r["model_fps"]), "%.1f" % float(r["model_mean_ms"]),
                         "%.2f" % float(r["e2e_fps"]), "—", "—", "%.1f×" % (float(r["model_fps"]) / base)])
        for prec in ("fp32", "fp16", "int8"):
            r = tb[(res, prec)]
            q = float(r["throughput_qps"])
            rows.append([res, "TensorRT", prec.upper() + ("*" if prec == "int8" else ""), "%.2f" % q,
                         "%.1f" % float(r["gpu_compute_mean_ms"]), "—", r["engine_size_mb"], r["build_time_s"],
                         "%.1f×" % (q / base)])
    write("T7_jetson_benchmark", "Inference speed on Jetson Nano 4 GB (YOLO11n, batch 1)",
          ["Resolution", "Runtime", "Precision", "Throughput (FPS)", "GPU latency (ms)", "End-to-end FPS",
           "Engine (MB)", "Build time (s)", "Speed-up vs PyTorch FP32"], rows,
          "PyTorch: fused model forward pass (model only) and YOLO.predict on a val image (end-to-end, incl. pre/post-processing). "
          "TensorRT: trtexec throughput and mean GPU compute time. *INT8 without calibration; no INT8 acceleration on Maxwell.")


# ------------------------------------------------------------------ T8
def parse_thermal(p):
    # accepts the English summary of jetson/thermal_test.sh and the Indonesian
    # wording of the original runs shipped in results/jetson/thermal_*/
    t = open(p, errors="replace").read()
    out = {}
    for comp in ("GPU", "CPU"):
        m = re.search(r"temp_%s\s*:\s*([0-9.]+)C \(min" % comp, t)
        out["idle_" + comp] = m.group(1) if m else ""
        m = re.search(r"temp_%s\s*:\s*(?:awal|start)=([0-9.]+)C\s+(?:akhir|end)=([0-9.]+)C\s+mean=([0-9.]+)C\s+max=([0-9.]+)C" % comp, t)
        if m:
            out.update({comp + "_start": m.group(1), comp + "_end": m.group(2), comp + "_mean": m.group(3), comp + "_max": m.group(4)})
    m = re.search(r"STRESS \(n=(\d+)", t); out["n"] = m.group(1) if m else ""
    m = re.search(r"GR3D_FREQ \(util GPU\): mean=(\d+)%", t); out["util"] = m.group(1) if m else ""
    m = re.search(r"(?:paruh|half)-1=(\d+)% vs (?:paruh|half)-2=(\d+)%", t); out["halves"] = "%s / %s" % m.groups() if m else ""
    m = re.search(r"(?:Throughput dilaporkan|Reported throughput): ([0-9.]+)", t); out["qps"] = m.group(1) if m else ""
    return out


def t8():
    rows = []
    for lab, p in (("5 min", P("jetson_results/thermal_results/thermal_summary.txt")),
                   ("30 min", P("jetson_results/thermal_results_30min/thermal_summary.txt"))):
        if not os.path.isfile(p):
            continue
        d = parse_thermal(p)
        rows.append([lab, d["n"], d["idle_GPU"], "%s → %s" % (d.get("GPU_start", ""), d.get("GPU_end", "")),
                     d.get("GPU_mean", ""), d.get("GPU_max", ""), d.get("CPU_mean", ""), d.get("CPU_max", ""),
                     d["util"], d["halves"], d["qps"]])
    write("T8_thermal", "Sustained inference on Jetson Nano (TensorRT FP16, 704 px)",
          ["Test", "Samples (1 s)", "Idle GPU °C", "GPU start → end °C", "GPU mean °C", "GPU max °C", "CPU mean °C",
           "CPU max °C", "GPU util. mean %", "Util. 1st / 2nd half %", "Throughput (FPS)"], rows,
          "Board power not measurable with tegrastats on the Jetson Nano developer kit (no on-board power sensor).")


TABLES = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7, "T8": t8}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(TABLES))
    a = ap.parse_args()
    for k in a.only:
        print(k)
        try:
            TABLES[k]()
        except Exception as e:
            print("  GAGAL %s: %s: %s" % (k, type(e).__name__, e))
    os.makedirs(OUT, exist_ok=True)
    # keep the sections of tables that were not rebuilt in this call (e.g. T2 without the dataset)
    md_path = os.path.join(OUT, "ALL_TABLES.md")
    sections = {}
    if os.path.isfile(md_path):
        for blk in re.split(r"\n(?=## )", open(md_path).read()):
            if blk.startswith("## "):
                sections[blk.split()[1]] = blk.rstrip("\n") + "\n"
    for blk in "\n".join(MD).split("\n## "):
        blk = blk if blk.startswith("## ") else "## " + blk
        if blk.strip() != "##":
            sections[blk.split()[1]] = blk.rstrip("\n") + "\n"
    with open(md_path, "w") as f:
        f.write("# Paper 1 — tables (generated by make_tables.py)\n\n" + "\n".join(sections[k] for k in sorted(sections)))
    print("Done:", os.path.join(OUT, "ALL_TABLES.md"))
