#!/usr/bin/env python3
"""
reeval_all.py — Re-evaluate every trained run with ONE protocol.

This is the single source of the accuracy numbers in the paper
(results/reeval/reeval_runs.csv and reeval_summary.csv); the values that
Ultralytics logs during training (results.csv) are not used.

Protocol (identical for all runs):
  - Ultralytics 8.4.48, MOPAD Site 2 val split (500 images)
  - imgsz = the run's training imgsz (read from its args.yaml)
  - best.pt, batch 16, conf 0.001, NMS IoU 0.7, FP32 (half=False)
  - confusion matrix at the Ultralytics defaults (conf 0.25, IoU 0.45),
    stored as matrix[pred, gt]; the last index is background

Usage (from the repository root):
  python scripts/eval/reeval_all.py              # every run in runs/ (~30-40 min for 36 runs)
  python scripts/eval/reeval_all.py --only R5_   # only runs whose name contains "R5_" (merged into the CSV)

Output (results/reeval/):
  reeval_runs.csv      one row per run (metrics, per-class AP, background FP)
  reeval_summary.csv   mean ± SD per experiment (runs grouped by name without _seedN)
  cm/<run>.npy         raw confusion matrix
  log/<run>.log        stdout/stderr of a failed run
Re-running overwrites the shipped CSVs, so `git diff results/` shows any
difference from the published numbers. Each run is evaluated in its own
subprocess so that the custom-module registrations cannot interfere.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import tandan_paths  # noqa: E402

DATA_YAML = str(tandan_paths.DATASET_YAML)
OUT = os.path.join(str(tandan_paths.RESULTS_DIR), "reeval")
CLASSES = ["dead", "healthy", "mismanaged", "smallish", "yellowish"]

# Every run folder with weights/best.pt directly under runs/
RUN_GLOBS = [os.path.join(str(tandan_paths.RUNS_DIR), "*")]


def find_runs(only=None):
    runs = []
    for g in RUN_GLOBS:
        for r in sorted(glob.glob(g)):
            if os.path.isfile(os.path.join(r, "weights", "best.pt")):
                if only is None or only in os.path.basename(r):
                    runs.append(r)
    return runs


def code_root(run):
    """Repository root: holds custom_modules/ (needed to unpickle modified models)."""
    return REPO


def read_imgsz(run):
    with open(os.path.join(run, "args.yaml")) as f:
        for line in f:
            if line.startswith("imgsz:"):
                return int(line.split(":")[1].strip())
    return 640


def group_name(run):
    name = os.path.basename(run)
    if "_seed" in name:
        return name.rsplit("_seed", 1)[0]
    return name


# ----------------------------------------------------------------------
# Mode --one: evaluasi satu run (dijalankan di subprocess)
# ----------------------------------------------------------------------
def eval_one(run):
    root = os.path.abspath(code_root(run))
    for p in (root, os.path.join(root, "scripts", "train")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import ultralytics
    from ultralytics import YOLO

    # Make the custom-module helpers importable (A1, E1, T2, R1-R3, ...)
    for mod in ("register_tandan", "register_spd", "register_rep", "register_repc3k2"):
        try:
            __import__(mod)
        except Exception:
            pass

    imgsz = read_imgsz(run)
    model = YOLO(os.path.join(run, "weights", "best.pt"))
    m = model.val(
        data=DATA_YAML, split="val", imgsz=imgsz, batch=16,
        conf=0.001, iou=0.7, half=False, device=0, workers=4,
        plots=True, verbose=False,
        project=os.path.join(str(tandan_paths.RUNS_DIR), "val"), name=os.path.basename(run),
        exist_ok=True,
    )

    res = {
        "run": os.path.basename(run),
        "group": group_name(run),
        "path": os.path.relpath(run, REPO),
        "ultralytics": ultralytics.__version__,
        "imgsz": imgsz,
        "mAP50_95": round(float(m.box.map), 4),
        "mAP50": round(float(m.box.map50), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "infer_ms": round(float(m.speed.get("inference", 0.0)), 2),
    }
    # mAP50-95 per kelas
    maps = list(m.box.maps)
    for i, c in enumerate(CLASSES):
        res[f"ap5095_{c}"] = round(float(maps[i]), 4) if i < len(maps) else ""

    # Confusion matrix -> FP background (prediksi tanpa GT) & FN (GT terlewat)
    cm = getattr(getattr(m, "confusion_matrix", None), "matrix", None)
    if cm is not None:
        cm = np.asarray(cm)
        nc = len(CLASSES)
        os.makedirs(os.path.join(OUT, "cm"), exist_ok=True)
        np.save(os.path.join(OUT, "cm", res["run"] + ".npy"), cm)
        bg_fp = cm[:nc, nc]          # prediksi kelas i, GT = background
        fn = cm[nc, :nc]             # GT kelas j, tidak terdeteksi
        res["bg_fp_total"] = int(bg_fp.sum())
        for i, c in enumerate(CLASSES):
            res[f"bg_fp_{c}"] = int(bg_fp[i])
        res["bg_fp_to_healthy_frac"] = round(float(bg_fp[1] / bg_fp.sum()), 4) if bg_fp.sum() else 0.0
        res["fn_total"] = int(fn.sum())
        res["gt_total"] = int(cm[:, :nc].sum())
    print("RESULT_JSON " + json.dumps(res))


# ----------------------------------------------------------------------
# Mode utama: loop semua run lewat subprocess, lalu rekap
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=str, default=None)
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args()

    if a.one:
        eval_one(a.one)
        return

    os.makedirs(os.path.join(OUT, "log"), exist_ok=True)
    runs = find_runs(a.only)
    print(f"{len(runs)} run akan dievaluasi\n")
    rows = []
    for k, run in enumerate(runs, 1):
        name = os.path.basename(run)
        p = subprocess.run([sys.executable, __file__, "--one", run],
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
        if line:
            r = json.loads(line[-1][len("RESULT_JSON "):])
            rows.append(r)
            print(f"[{k:2d}/{len(runs)}] {name:24s} mAP50-95={r['mAP50_95']:.4f} "
                  f"bgFP={r.get('bg_fp_total', '-')}")
        else:
            with open(os.path.join(OUT, "log", name + ".log"), "w") as f:
                f.write(p.stdout + "\n" + p.stderr)
            print(f"[{k:2d}/{len(runs)}] {name:24s} FAILED -> results/reeval/log/{name}.log")

    if not rows:
        return

    # --only: gabungkan dengan hasil lama (run yang sama diganti), jangan timpa
    old_csv = os.path.join(OUT, "reeval_runs.csv")
    if a.only and os.path.isfile(old_csv):
        new_names = {r["run"] for r in rows}
        with open(old_csv) as f:
            old = [r for r in csv.DictReader(f) if r["run"] not in new_names]
        for r in old:
            for k, v in list(r.items()):
                if k in ("run", "group", "path", "ultralytics"):
                    continue
                try:
                    r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except (ValueError, AttributeError):
                    pass
        rows = old + rows
        print(f"\n(digabung dengan {len(old)} run lama dari reeval_runs.csv)")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "reeval_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Rekap per eksperimen (mean ± std antar seed)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    def ms(vals):
        vals = [v for v in vals if v != "" and v is not None]
        if not vals:
            return "", ""
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else ""
        return round(mean, 4), (round(sd, 4) if sd != "" else "")

    summ = []
    for g, rs in groups.items():
        row = {"group": g, "n_seeds": len(rs), "imgsz": rs[0]["imgsz"]}
        for k in ["mAP50_95", "mAP50", "precision", "recall", "bg_fp_total",
                  "bg_fp_to_healthy_frac", "fn_total",
                  "ap5095_mismanaged", "ap5095_dead", "ap5095_yellowish"]:
            row[k + "_mean"], row[k + "_std"] = ms([r.get(k, "") for r in rs])
        summ.append(row)
    summ.sort(key=lambda x: -(x["mAP50_95_mean"] or 0))
    with open(os.path.join(OUT, "reeval_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    print("\n=== RINGKASAN (mean ± std) ===")
    print(f"{'group':24s} n  imgsz  mAP50-95          bgFP_total   mismanaged")
    for s in summ:
        sd = s["mAP50_95_std"]
        sd = f"±{sd:.4f}" if sd != "" else "       "
        print(f"{s['group']:24s} {s['n_seeds']}  {s['imgsz']:4d}   "
              f"{s['mAP50_95_mean']:.4f} {sd}   {s['bg_fp_total_mean']!s:>9}   "
              f"{s['ap5095_mismanaged_mean']}")
    print("\nFiles: results/reeval/reeval_runs.csv, results/reeval/reeval_summary.csv")


if __name__ == "__main__":
    main()