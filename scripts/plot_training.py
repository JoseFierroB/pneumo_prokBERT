#!/usr/bin/env python3
"""Training curves from trainer_state.json — one metric per file.

Reads finetuning_outputs/*/checkpoint-*/trainer_state.json (last checkpoint
per run holds the full log_history) and draws, per run:
  loss_train__<run>.png, loss_eval__<run>.png, lr__<run>.png,
  mcc_eval__<run>.png, auc_eval__<run>.png
Missing series are skipped with a note (e.g. small runs have no per-step
train loss because logging_steps=50 exceeded their total steps).

Usage:
  pixi run python scripts/plot_training.py [--runs mini_223k_replica_2ep ...] [--out-dir reports/training_curves]
"""
import argparse
import glob
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Training curves from trainer_state.json")
    p.add_argument("--runs", nargs="*", default=None, help="subset of run dir names (default: all with checkpoints)")
    p.add_argument("--runs-root", type=str, default="finetuning_outputs")
    p.add_argument("--out-dir", type=str, default="reports/training_curves")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted([d for d in os.listdir(args.runs_root)
                       if os.path.isdir(os.path.join(args.runs_root, d))])
    if args.runs:
        run_dirs = [d for d in run_dirs if d in args.runs]
    index = []
    for run in run_dirs:
        ckpts = sorted(glob.glob(os.path.join(args.runs_root, run, "checkpoint-*")),
                       key=lambda p: int(p.split("-")[-1]))
        if not ckpts:
            print(f"{run}: no checkpoints, skipped")
            continue
        st = json.load(open(os.path.join(ckpts[-1], "trainer_state.json")))
        lh = st.get("log_history", [])
        train = [(e["step"], e["loss"]) for e in lh if "loss" in e and "eval_loss" not in e]
        grad = [(e["step"], e["grad_norm"]) for e in lh if "grad_norm" in e]
        lrs = [(e["step"], e["learning_rate"]) for e in lh if "learning_rate" in e]
        ev = [e for e in lh if "eval_loss" in e]
        made = []
        def one(xs, ys, title, yl, fn):
            if not xs:
                return
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, lw=1.5)
            ax.set(title=f"{title} — {run}", xlabel="step", ylabel=yl)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / fn, dpi=120)
            plt.close(fig)
            made.append(fn)
            print(f"{fn} OK")
        if train:
            one([s for s, _ in train], [v for _, v in train], "train loss", "loss", f"loss_train__{run}.png")
        else:
            print(f"{run}: no per-step train loss (logging_steps exceeded total steps)")
        if grad:
            one([s for s, _ in grad], [v for _, v in grad], "grad norm", "grad_norm", f"gradnorm__{run}.png")
        if lrs:
            one([s for s, _ in lrs], [v for _, v in lrs], "learning rate", "lr", f"lr__{run}.png")
        if ev:
            one([e["step"] for e in ev], [e["eval_loss"] for e in ev], "eval loss", "loss", f"loss_eval__{run}.png")
            for key, fn in [("eval_mcc", f"mcc_eval__{run}.png"), ("eval_roc_auc", f"auc_eval__{run}.png")]:
                vals = [(e["step"], e[key]) for e in ev if key in e]
                if vals:
                    one([s for s, _ in vals], [v for _, v in vals], key, key, fn)
        index.append({"run": run, "max_step": st.get("max_steps"), "files": made,
                      "note": "" if train else "no per-step train loss (see above)"})
    json.dump(index, open(out / "index.json", "w"), indent=2)
    print(f"index: {len(index)} runs")


if __name__ == "__main__":
    main()
