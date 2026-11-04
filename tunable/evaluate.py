#!/usr/bin/env python3
"""Evaluation helpers: combined loss overlays + train-vs-test gap wrapper."""
import glob
import json
import os
import re
import subprocess

import sys
sys.path.insert(0, ".")


def loss_overlay(run_dir, out_path=None):
    """Combined train-vs-val loss PNG from the last checkpoint history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                   key=lambda p: int(p.split("-")[-1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run_dir}")
    st = json.load(open(os.path.join(ckpts[-1], "trainer_state.json")))
    lh = st.get("log_history", [])
    tr = [(e["step"], e["loss"]) for e in lh if "loss" in e and "eval_loss" not in e]
    ev = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]
    out = out_path or os.path.join(run_dir, "loss_overlay.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([s for s, _ in tr], [v for _, v in tr], color="red", label="train loss")
    ax.plot([s for s, _ in ev], [v for _, v in ev], color="blue", label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("train vs val loss - " + os.path.basename(run_dir.rstrip("/")))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"{run_dir}: train pts {len(tr)}, val pts {len(ev)} -> {out}")
    return out


def run_gap(model_dir, alias, hf_split_dir, train_csv, test_csv, test_name,
            out_dir, batch_size=512, gpu=0, log_path=None):
    """Run scripts/generalization_gap.py as subprocess; parse gap AUC/MCC."""
    cmd = ["pixi", "run", "python", "scripts/generalization_gap.py",
           "--model", model_dir, "--alias", alias,
           "--hf-split-dir", hf_split_dir, "--train-csv", train_csv,
           "--test-csv", test_csv, "--test-name", test_name,
           "--out-dir", out_dir, "--batch-size", str(batch_size),
           "--gpu", str(gpu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if log_path:
        with open(log_path, "w") as f:
            f.write(out)
    if r.returncode != 0:
        raise SystemExit(f"gap failed rc={r.returncode} for {alias} (see {log_path})")
    m = re.search(r"gap AUC ([+-][0-9.]+).*?MCC ([+-][0-9.]+)", out.replace("\r", "\n"))
    res = {"alias": alias, "rc": 0}
    if m:
        res["gap_auc"] = float(m.group(1))
        res["gap_mcc"] = float(m.group(2))
    print(f"{alias}: gap AUC {res.get('gap_auc'):+.4f} MCC {res.get('gap_mcc'):+.4f}")
    return res
